import argparse
import json
import os
import sys
from pathlib import Path
import torch
from data_utils import CRDataset, get_bev_folders, get_trajs_path
from torch.utils.data import DataLoader
import numpy as np
import tqdm
from scipy.stats import multivariate_normal
from random import randrange
mod_path = (Path(__file__).resolve().parents[2] / "Wale-Net").resolve()
sys.path.append(str(mod_path))

import mod_prediction

# Custom imports
from mod_prediction.utils.model import predictionNet
from mod_prediction.utils.neural_network import NLL, MSE, MSE2, NLL2
from mod_prediction.utils.cuda import cudanize
from mod_prediction.utils.visualization import draw_in_scene



def get_probabilities(fut_pred, fut):
    """This function calculates the predicted probabilites of the prediction for the actual ground truth points
    Be careful:
    * The probability is calculated for a 1m x 1m square
    * Due to high computation times we do not iterate over the whole batch but only take the first sample of each batch!

    Arguments:
        fut_pred {[torch tensor]} -- [Predicted trajectories]
        fut {[torch tensor]} -- [Ground truth trajectories]

    Returns:
        [np.array] -- [Array of probabilities for every time step.]
    """
    # iterate over batchsize
    batch_list = []
    # for b in range(fut_pred.shape[1]):
    b = 0
    # iterate over time
    prob_list = []
    for t in range(fut_pred.shape[0]):
        # Obtain prediction values
        mu_x = fut_pred[t, b, 0].cpu().detach().numpy()
        mu_y = fut_pred[t, b, 1].cpu().detach().numpy()
        sigma_x = 1 / fut_pred[t, b, 2].cpu().detach().numpy()
        sigma_y = 1 / fut_pred[t, b, 3].cpu().detach().numpy()
        rho = fut_pred[t, b, 4].cpu().detach().numpy()

        # Calculate mu and sigma matrix
        mu = np.array([mu_x, mu_y])
        sigma = np.array(
            [
                [sigma_x ** 2, rho * sigma_x * sigma_y],
                [rho * sigma_x * sigma_y, sigma_y ** 2],
            ]
        )

        # Calculate multivariate normal distribution
        F = multivariate_normal(mu, sigma)

        # Calculate probability for a 1m x 1m square
        cdf_plus = F.cdf([fut[t, b, 0] + 0.5, fut[t, b, 1] + 0.5])
        cdf_minus = F.cdf([fut[t, b, 0] - 0.5, fut[t, b, 1] - 0.5])
        cdf_combined_1 = F.cdf([fut[t, b, 0] - 0.5, fut[t, b, 1] + 0.5])
        cdf_combined_2 = F.cdf([fut[t, b, 0] + 0.5, fut[t, b, 1] - 0.5])
        prob_list.append(cdf_plus - cdf_combined_1 - cdf_combined_2 + cdf_minus)

    batch_list.append(prob_list)

    batch_list = np.array(batch_list)
    return np.mean(batch_list, axis=0)

def evaluate(dataloader, net, common_args, verbose=True):
    """This function calculates evaluation metrics on a given dataset and net.

    Arguments:
        dataloader {[torch Dataloader]} -- [pytorch Dataloader]
        net {[torch nn.module]} -- [pytorch neural network]
        common_args {[dict]} -- [network arguments]

    Returns:
        [rmse, nll, prob_list, img_list] -- [RMSE, NLL, Probabilites, Scene images for visualization]
    """

    # Initialize torch variables
    if common_args["use_cuda"]:
        lossVals_mse = torch.zeros(common_args["out_length"]).cuda()
        counts_mse = torch.zeros(common_args["out_length"]).cuda()
        lossVals_nll = torch.zeros(common_args["out_length"]).cuda()
        counts_nll = torch.zeros(common_args["out_length"]).cuda()
    else:
        lossVals_mse = torch.zeros(common_args["out_length"])
        counts_mse = torch.zeros(common_args["out_length"])
        lossVals_nll = torch.zeros(common_args["out_length"])
        counts_nll = torch.zeros(common_args["out_length"])

    img_list = []
    prob_list = []

    for i, data in enumerate(tqdm.tqdm(dataloader) if verbose else dataloader):
        # Unpack data
        smpl_id, hist, nbrs, fut, sc_img = data

        # Shrink fut to out_length
        fut = fut[: common_args["out_length"], :, :]

        # Initialize Variables
        if common_args["use_cuda"]:
            hist, nbrs, fut, sc_img = cudanize(hist, nbrs, fut, sc_img)

        # Predict
        fut_pred1 = net(hist, nbrs, sc_img)


        _, l_nll = NLL(fut_pred1, fut)
        _, l_mse = MSE(fut_pred1, fut)
        prob_list.append(get_probabilities(fut_pred1, fut))

        # Get average over batch
        counts_mse += l_mse.shape[1]
        lossVals_mse += l_mse.sum(axis=1).detach()
        counts_nll += l_nll.shape[1]
        lossVals_nll += l_nll.sum(axis=1).detach()

        # Return some random visualizations for tensorboard
        if i == 0 or i == 10 or i == 20 or i == 50 or i == 70 or i == 100:
            img_list.append(
                draw_in_scene(
                    fut,
                    sc_img,
                    fut_pred1=fut_pred1,
                    render_method=common_args["scene_image_method"],
                )
            )

        if common_args["debug"]:
            break

    # Get average over batch
    nll = lossVals_nll / counts_nll
    nll = nll.cpu().detach().numpy()

    if verbose:
        try:
            print("=" * 30)
            print("NLL 1s: {0:.2f}".format(nll[9]))
            print("NLL 2s: {0:.2f}".format(nll[19]))
            print("NLL 3s: {0:.2f}".format(nll[29]))
            print("NLL 4s: {0:.2f}".format(nll[39]))
            print("NLL 5s: {0:.2f}".format(nll[49]))
            print("=" * 30)
        except IndexError:
            pass

    if common_args["use_cuda"]:
        rmse = torch.pow(lossVals_mse / counts_mse, 0.5)
        rmse = np.array(rmse.cpu())
    else:
        rmse = np.array(torch.pow(lossVals_mse / counts_mse, 0.5))

    if verbose:
        try:
            print("=" * 30)
            print("RMSE 1s: {0:.2f}".format(rmse[9]))
            print("RMSE 2s: {0:.2f}".format(rmse[19]))
            print("RMSE 3s: {0:.2f}".format(rmse[29]))
            print("RMSE 4s: {0:.2f}".format(rmse[39]))
            print("RMSE 5s: {0:.2f}".format(rmse[49]))
            print("=" * 30)
        except IndexError:
            pass

    prob_list = np.mean(np.array(prob_list), axis=0)

    if verbose:
        try:
            print("=" * 30)
            print("PROB 1s: {0:.2f}".format(prob_list[9]))
            print("PROB 2s: {0:.2f}".format(prob_list[19]))
            print("PROB 3s: {0:.2f}".format(prob_list[29]))
            print("PROB 4s: {0:.2f}".format(prob_list[39]))
            print("PROB 5s: {0:.2f}".format(prob_list[49]))
            print("=" * 30)
        except IndexError:
            pass

    return rmse, nll, prob_list, img_list


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=f"{Path.home()}/EDRL/collision_risk/configs/default.json"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=f"{Path.home()}/EDRL/collision_risk/trained_models/default_4064.tar",
    )
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    # Read config file
    with open(args.config, "r") as f:
        eval_args = json.load(f)

    # Network Arguments
    eval_args["use_cuda"] = bool(eval_args["gpu"])
    eval_args["save_path"] = os.path.join(
        eval_args["save_path"], eval_args["dataset"]
    )
    eval_args["model_name"] = args.config.split("/")[6].split(".")[0]
    eval_args["debug"] = args.debug
    eval_args["online_layer"] = 0

    # Initialize network
    net = predictionNet(eval_args)
    if eval_args["use_cuda"]:
        net.load_state_dict(torch.load(args.model))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(args.model, map_location=torch.device("cpu")))


    datafolder_list = get_bev_folders(f"{Path.home()}/EDRL/collision_risk/data/raw_data/")
    trajs_path_list = get_trajs_path(datafolder_list, "processed_traj.pkl")

    SEED = randrange(7)

    # Initialize data loaders
    if eval_args["debug"]:
        tsSet = CRDataset(
            file_path = trajs_path_list[SEED], 
            img_path = datafolder_list[SEED],
            )
    else:
        tsSet = CRDataset(
            file_path = trajs_path_list[SEED: SEED+1], 
            img_path = datafolder_list[SEED: SEED+1],
            )
        # Create the DataLoader to load data in batches
    
    tsDataloader = DataLoader(tsSet, batch_size=1, shuffle=True, collate_fn=tsSet.collate_fn)

    # Call main evaulation function
    rmse, nll, probs, _ = evaluate(tsDataloader, net, eval_args)
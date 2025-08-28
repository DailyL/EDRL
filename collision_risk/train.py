import sys
import os
from pathlib import Path
import argparse
import json
import numpy as np
import pkbar
from random import randrange
from cr_utils.data_utils import CRDataset, get_data_folders, get_trajs_path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

mod_path = (Path(__file__).resolve().parents[2] / "Wale-Net").resolve()
sys.path.append(str(mod_path))
import mod_prediction

# Custom imports
from mod_prediction.utils.model import predictionNet
from mod_prediction.utils.neural_network import MSE, NLL, MSE2, NLL2, add_histograms
from mod_prediction.utils.cuda import cudanize
from mod_prediction.utils.visualization import TrainingsVisualization
from mod_prediction.evaluate import evaluate




def create_dataloader(args):
    """Create dataloader for training, validation and test.

    Args:
        args ([dict]): [args for dataloader creation, relevant for dataset path, batch_size and workers]
        verbose (bool, optional): [Verbose]. Defaults to True.

    Returns:
        [torch.utils.data.dataloader]: [for training, validation and test set]
    """

    imgfolder_list, trajs_path_list = get_data_folders(
        directory=f"{Path(__file__).resolve().parents[1]}/collision_risk/data/raw_data/",
        specific_file_name="processed_traj.pkl",
        random_sample=True,
        sample_num=1000
    )

    print(imgfolder_list)
    print(trajs_path_list)


    if args["debug"]:
        SEED = randrange(7)
        imgfolder_list = imgfolder_list[SEED]
        trajs_path_list = trajs_path_list[SEED]
    
    DataSet = CRDataset(file_path=trajs_path_list,
                        img_path=imgfolder_list,
                        )
    # Create the DataLoader to load data in batches
    dataloader = DataLoader(DataSet, 
                            batch_size=args["batch_size"], 
                            shuffle=True,
                            num_workers=args["worker"], 
                            collate_fn=DataSet.collate_fn,
                            )

    return dataloader


def main(args, verbose=True):

    """
    main function for training
    """
    save_path = f"{Path.home()}/EDRL/collision_risk/trained_models/"

    model_path = os.path.join(save_path,  args["model_name"] + ".tar")

    print(f"Save the model in {model_path}")

     # Initialize network
    net = predictionNet(args)
    if args["use_cuda"]:
        net = net.cuda()

    # Get number of parameters
    if verbose:
        pytorch_total_params = sum(
            p.numel() for p in net.parameters() if p.requires_grad
        )
        print("Model initialized with {} parameters".format(pytorch_total_params))

    # Continue training if demanded
    if args["continue_training_from"]:
        if verbose:
            print(
                "Loading weights from {}".format(args["continue_training_from"])
            )

        if args["use_cuda"]:
            net.load_state_dict(torch.load(args["continue_training_from"]))
            net = net.cuda()
        else:
            net.load_state_dict(
                torch.load(
                    args["continue_training_from"],
                    map_location=torch.device("cpu"),
                )
            )

    # Initialize tensorboard
    writer = SummaryWriter(
        os.path.join(args["tb_logs"], args["model_name"])
    )
    # Write histograms to tensorboards for initializatoin
    writer = add_histograms(writer, net, global_step=0)


    # Initialize optimizer
    optimizer_rmse = torch.optim.Adam(
        net.parameters(),
        lr=args["lr_rmse"],
        weight_decay=args["decay_rmse"],
    )
    optimizer_nll = torch.optim.Adam(
        net.parameters(),
        lr=args["lr_nll"],
        weight_decay=args["decay_nll"],
    )

    # Batch size
    if args["debug"]:
        args["batch_size"] = 2  # for faster computing

    # Create data loaders
    dataloader = create_dataloader(args)

    trDataloader = dataloader
    valDataloader = dataloader
    tsDataloader = dataloader
    allDataloader = dataloader
    

    # Add graph to tensorboard logs
    trainings_sample = next(iter(trDataloader))
    (smpl_id, hist, nbrs, fut, sc_img) = trainings_sample
    if args["use_cuda"]:
        hist, nbrs, fut, sc_img = cudanize(hist, nbrs, fut, sc_img)

    # Init Trainingsvisualization
    train_vis = TrainingsVisualization(trainings_sample, update_rate=100)

    # writer.add_graph(net, [hist, nbrs, sc_img])

    # Check output length
    if args["out_length"] > len(fut):
        raise IndexError(
            "Not enough ground truth time steps in dataset. Demanded: {}. In dataset: {}".format(
                args["out_length"], len(fut)
            )
        )
    elif args["out_length"] < len(fut) and verbose:
        print(
            "Shrinking ground truth to {} time steps".format(args["out_length"])
        )

    # Variables holding train and validation loss values
    best_val_loss = np.inf

    # Main loop for training
    for epoch_num in range(args["pretrainEpochs"] + args["trainEpochs"]):
        if epoch_num == 0:
            if verbose:
                print("Pre-training with MSE loss")
            optimizer = optimizer_rmse
        elif epoch_num == args["pretrainEpochs"]:
            if verbose:
                print("Training with NLL loss")
            optimizer = optimizer_nll
            if args["save_best"]:
                if verbose:
                    print("Loading best model from pre-training")
                if args["use_cuda"]:
                    net.load_state_dict(torch.load(model_path))
                    net = net.cuda()
                else:
                    net.load_state_dict(
                        torch.load(model_path, map_location=torch.device("cpu"))
                    )

        # Training
        net.train_flag = True
        train_loss_list = []

        # Init progbar
        if verbose:
            kbar = pkbar.Kbar(
                target=len(trDataloader),
                epoch=epoch_num,
                num_epochs=args["pretrainEpochs"] + args["trainEpochs"],
            )

        for i, data in enumerate(trDataloader):
            # Unpack data
            smpl_id, hist, nbrs, fut, sc_img = data


            # Shrink hist, nbrs to in_length
            hist = hist[hist.shape[0] - args["in_length"] :, :, :]
            nbrs = nbrs[nbrs.shape[0] - args["in_length"] :, :, :]

            # Shrink fut to out_length
            fut = fut[: args["out_length"], :, :]

            # Optionally initialize them on GPU
            if args["use_cuda"]:
                hist, nbrs, fut, sc_img = cudanize(hist, nbrs, fut, sc_img)

            # Forward pass
            fut_pred1 = net(hist, nbrs, sc_img)

            if epoch_num < args["pretrainEpochs"]:
                loss, _ = MSE(fut_pred1, fut)
                if verbose:
                    kbar.update(i, values=[("MSE", loss)])
            else:
                loss, _ = NLL(fut_pred1, fut)
                if verbose:
                    kbar.update(i, values=[("NLL", loss)])

            # Track train loss
            train_loss_list.append(loss.detach().cpu().numpy())

            # Backprop and update weights
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10)  # Gradient clipping
            optimizer.step()

            if args["vis"]:
                train_vis.update(loss, net, force_save=True)

            if args["debug"]:
                break

        writer.add_scalar("Training_Loss", np.mean(train_loss_list), epoch_num)

        # Add histograms after every training epoch
        # writer = add_histograms(writer, net, global_step=epoch_num)

        # Validation
        net.train_flag = False
        val_loss_list = []

        for i, data in enumerate(valDataloader):
            # Unpack data
            smpl_id, hist, nbrs, fut, sc_img = data

            # Shrink fut to out_length
            fut = fut[: args["out_length"], :, :]

            if args["use_cuda"]:
                hist, nbrs, fut, sc_img = cudanize(hist, nbrs, fut, sc_img)

            # Forward pass
            fut_pred1 = net(hist, nbrs, sc_img)

            if epoch_num < args["pretrainEpochs"]:
                loss, _ = MSE(fut_pred1, fut)
            else:
                loss, _ = NLL(fut_pred1, fut)

            val_loss_list.append(loss.detach().cpu().numpy())

            if args["debug"]:
                break

        val_loss = np.mean(val_loss_list)
        if verbose:
            kbar.add(1, values=[("val_loss", val_loss)])
        writer.add_scalar("Validation_Loss", val_loss, epoch_num)

        # Save model if val_loss_improved
        if args["save_best"]:
            if val_loss < best_val_loss:
                torch.save(net.state_dict(), model_path)
                best_val_loss = val_loss

        if args["debug"]:
            break

    if not args["save_best"]:
        torch.save(net.state_dict(), model_path)

    # Evaluation
    if verbose:
        print("\nEvaluating on test set...")

    # Load best model
    if args["save_best"]:
        if verbose:
            print("Loading best model")
        if args["use_cuda"]:
            net.load_state_dict(torch.load(model_path))
            net = net.cuda()
        else:
            net.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu"))
            )

    # Evaluate on test set
    rmse, nll, probs, img_list = evaluate(tsDataloader, net, args, verbose)

    try:
        # Evaluate on all data
        print("Evaluating on all data:")
        rmse_all, nll_all, probs_all, img_list_all = evaluate(
            allDataloader, net, args, verbose
        )
    except Exception as e:
        print("Not evaluating on all data: {}".format(e))

    # Write to tensorboard
    for i in range(len(rmse)):
        writer.add_scalar("rmse_test", rmse[i], i + 1)
        writer.add_scalar("nll_test", nll[i], i + 1)
        writer.add_scalar("probs_test", probs[i], i + 1)
        try:
            writer.add_scalar("rmse_all", rmse_all[i], i + 1)
            writer.add_scalar("nll_all", nll_all[i], i + 1)
            writer.add_scalar("probs_all", probs_all[i], i + 1)
        except Exception:
            pass

    for i, img in enumerate(img_list):
        writer.add_image("img_" + str(i), img, dataformats="HWC")

    # Write hyperparamters to tensorboard
    args["grid_size"] = str(args["grid_size"])
    writer.add_hparams(args, {"rmse": np.mean(rmse), "nll": np.mean(nll)})

    writer.close()

    return np.mean(nll)







if __name__ == "__main__":

    import random

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=f"{Path.home()}/EDRL/collision_risk/configs/default.json"
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--vis", action="store_true", default=False)
    args = parser.parse_args()


    # Load config
    with open(args.config, "r") as f:
        train_args = json.load(f)

    seed = random.randint(0, 20000)
    
    train_args["model_name"] = args.config.split("/")[6].split(".")[0] + f"_{seed}"

    # Network Arguments
    train_args["use_cuda"] = bool(train_args["gpu"])
    train_args["debug"] = args.debug
    train_args["vis"] = args.vis
    train_args["online_layer"] = 0
    
    main(args = train_args)



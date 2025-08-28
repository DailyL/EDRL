import json
import sys
import os
from pathlib import Path
import torch
import numpy as np
import time
import copy

from env.util.env_utils import get_surrounding_vehicles
from collision_risk.cr_utils.geometry import transform_trajectories, point_in_rectangle, transform_back
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.component.traffic_participants.cyclist import Cyclist
from metadrive.constants import MetaDriveType
from metadrive.component.vehicle.base_vehicle import BaseVehicle

mod_path = (Path(__file__).resolve().parents[2] / "Wale-Net").resolve()
sys.path.append(str(mod_path))
import mod_prediction

# Custom imports
from mod_prediction.utils.model import predictionNet
from mod_prediction.utils.neural_network import MSE, NLL, MSE2, NLL2, add_histograms
from mod_prediction.utils.cuda import cudanize
from mod_prediction.utils.visualization import TrainingsVisualization
from mod_prediction.evaluate import evaluate

#np.set_printoptions(threshold=sys.maxsize)


class Prediction(object):
    def __init__(self, args = None, SEED = None, multiprocessing=False):
        
        self._multiprocessing = multiprocessing
        self.predict_args = args

        if args is None:
            # Load default online args
            with open(f"{Path(__file__).resolve().parents[1]}/collision_risk/configs/default.json") as f:
                self.predict_args = json.load(f)

        mod_prediction_path = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        
        self.model_path = f"{mod_prediction_path}/collision_risk/trained_models/collision_risk_{SEED}.tar"

        # Network Arguments
        self.predict_args["use_cuda"] = bool(self.predict_args["gpu"])
        self.predict_args["online_layer"] = 0
        # Initialize network
        self.net = predictionNet(self.predict_args)

        if self.predict_args["use_cuda"]:
            saved_model_dict = torch.load(self.model_path)
            self.net = self.net.cuda()
        else:
            saved_model_dict = torch.load(
                self.model_path, map_location=torch.device("cpu")
            )
        self.net.load_state_dict(saved_model_dict)
        self.vehicles_config = None

        self.observation_storage = [] # Save buffer without transformation
        self.temp_storage = []        # Save buffer after transformation 
        self.past_points = self.predict_args["in_length"]
        self.time_step = 0
        self.rotation_dict = {}
        self.translation_dict = {}


    def step(self, engine, ego_vehicle):

        """Step function that executes the main function of the prediction.

        Arguments:
            Vehicle -- [Metadrive vehicle object]
            engine  -- [Metadrive engine object]

        Returns:
            prediction_result [dict] -- [dictionary with obstacle ids as keys and x,y position and covariance matrix as values]
        """

        self.prediction_result = []


        # Only predict surrounding vehicles of ego vehicle

        surrounding_vehicles = get_surrounding_vehicles(ego_vehicle, engine)

        temp_observations = self._preprocessing(engine=engine, target_vehicles=surrounding_vehicles)

        
        self.step_multi(temp_observations)
        

        return self.prediction_result



    
    def step_multi(self, observation_list):
        
        for obj in observation_list:
            obstacle_id = obj['id']
        

        """This function makes multiple predictions at the same time based on a list of obstacles.
           This should reduce computational effort.

        Args:
            obstacle_id_list ([list]): [List of obstacle IDs to be predicted]
        """

        # Create tensors
        hist_batch = torch.zeros(
            [self.predict_args["in_length"], len(observation_list), 2], dtype=torch.float32
        )
        no_nbrs_cells = (
            self.predict_args["grid_size"][0]
            * self.predict_args["grid_size"][1]
        )

        nbrs_batch = torch.zeros(
            [
                self.predict_args["in_length"],
                no_nbrs_cells * len(observation_list),
                2,
            ], dtype=torch.float32
        )

        sc_img_batch = torch.zeros([len(observation_list), 1, 256, 256], dtype=torch.float32)

        for obj_num, obj in enumerate(observation_list):
            #hist_batch[:, obj_num, :] = obj['hist'].clone().detach()
            try:
                hist_batch[:, obj_num, :] = torch.as_tensor(obj['hist'], dtype=torch.float32)
            except RuntimeError:
                print("Error for object: ", obj['id'])
            if len(obj['nbrs']) > 0:  # Ensure nbrs data is not empty
                nbrs_data = torch.as_tensor(obj['nbrs'], dtype=torch.float32).clone().detach()
                #nbrs_data =obj['nbrs'].clone().detach()
                nbrs_batch[:, obj_num * no_nbrs_cells : (obj_num + 1) * no_nbrs_cells, :] = nbrs_data
            else:
                print(f"No neighbor data for object {obj['id']}, skipping.")

        
 

        # Neural Network
        self.fut_pred = self._predict(hist_batch, nbrs_batch, sc_img_batch)


        
        

        # Post Processing
        for obj_num, obj in enumerate(observation_list):
            fut_pred = self.fut_pred[:, obj_num, :]

            fut_pos, fut_cov = self._postprocessing(
                torch.unsqueeze(fut_pred, 1), obj['id']
            )

            self.prediction_result.append({
                'id': obj['id'],
                'pos_list': fut_pos,
                'cov_list': fut_cov,
                })

        




    def _predict(self, hist, nbrs, sc_img):
        """[Processing trough the neural network]

        Args:
            hist ([torch.Tensor]): [Past positions of the vehicle being predicted. Shape: [in_length, batch_size, 2]]
            nbrs ([torch.Tensor]): [Neighbor array of the vehicle being predicted. Shape: [in_length, grid_size * batch_size, 2]]
            sc_img ([torch.Tensor]): [Scene image for the prediction. Shape: [batch_size, 1, 256, 256]]

        Returns:
            [torch.Tensor]: [Network output. Shape: [50, batch_size, 5]]
        """

        with torch.inference_mode():
            if self.predict_args["use_cuda"]:
                hist, nbrs, _, sc_img = cudanize(hist, nbrs, None, sc_img)
            if hasattr(self, "obstacle_id"):
                fut_pred = self.net(hist, nbrs, sc_img, self.obstacle_id)
            else:
                fut_pred = self.net(hist, nbrs, sc_img)

        return fut_pred


    




    def init_buffer(self, engine, time_step = 0):

        """Initilize the input for the Prediction

        Returns:
            [list]: A Dict includes ALL vehicles' [hist, nbrs, sc_img]
        """

        if not self.observation_storage:
            for object in engine.get_objects().values():
                if isinstance(object, BaseVehicle):
                    initial_positions = [(0, 0) for _ in range(self.past_points)]
                    self.observation_storage.append({
                        'id': object.id,
                        'hist': initial_positions,
                        "nbrs": [],
                        })
                
            """ for object in engine.get_objects().values():
                if isinstance(object, BaseVehicle):
                    self.temp_storage.append({
                        'id': object.id,
                        'hist': initial_positions,
                        "nbrs": [],
                        }) """
        else:
            pass

    def _hard_init_buffer(self, engine):

        """Initilize the input for RESET function
           Call once when RESET the environment

        Returns:
            [list]: A Dict includes ALL vehicles' [hist, nbrs, sc_img]
        """
        self.observation_storage = []
        self.temp_storage = []

        for object in engine.get_objects().values():
            if isinstance(object, BaseVehicle):
                initial_positions = [(0, 0) for _ in range(self.past_points)]
                self.observation_storage.append({
                    'id': object.id,
                    'hist': initial_positions,
                    "nbrs": [],
                    })
        
        for object in engine.get_objects().values():
            if isinstance(object, BaseVehicle):
                self.temp_storage.append({
                    'id': object.id,
                    'hist': initial_positions,
                    "nbrs": [],
                    })
        
        
    def update_storage(self, storage_buffer, v_id, current_pose, nbrs=None):
        all_vehicle_ids = []

        for vehicle in storage_buffer:
            if vehicle is not None:
                 all_vehicle_ids.append(vehicle['id'])
            else:
                 pass
            
            if v_id in all_vehicle_ids:
                if vehicle['id'] == v_id:
                    vehicle['hist'].append(current_pose)
                    vehicle['nbrs'].append(nbrs)
                    if len(vehicle['hist']) > self.past_points:  # Keep only the last N steps
                        vehicle['hist'].pop(0)
                    if len(vehicle['nbrs']) > self.past_points:  # Keep only the last N steps
                        vehicle['nbrs'].pop(0)
                    break
            else:
                current_positions = [current_pose for _ in range(self.past_points)]
                storage_buffer.append({
                    'id': v_id,
                    'hist': current_positions,
                    'nbrs': [nbrs],
                    })
                break
    
    def _postprocessing(self, fut_pred, obj_id):

        """Transforming the neural network output to a prediction format in world coordinates

        Args:
            fut_pred ([torch.Tensor]): [Network output. Shape: [50, batch_size, 5]]
            obj_id ([int]): [Object ID]

        Returns:
            [tuple]: [Storing fut_pos, fut_cov in real world coordinates]
        """
        # avoid changing fut_pred
        fut_pred_copy = fut_pred.cpu().detach().numpy()
        fut_pred_copy = np.squeeze(
            fut_pred_copy, 1
        )  # use batch size axes for list axes in transform function
        fut_pred_trans = transform_back(
            fut_pred_copy,
            self.translation_dict[obj_id],
            self.rotation_dict[obj_id],
        )

        return fut_pred_trans


    def _preprocessing(self, engine, target_vehicles, time_step=None):

        """Prepare the input for the Prediction

        Returns:
            [list]: [hist, nbrs, sc_img as inputs for the neural network. See _predict for further Information]
        """

        

        if time_step is None:
            time_step = self.time_step

        # Get ids of the target vehicles
        ids = []
        for nbr_vehicle in target_vehicles:
            if nbr_vehicle is not None:
                 ids.append(nbr_vehicle.id)
            else:
                 pass



        
        for object in engine.get_objects().values():
            
            # ALL vehicles inside simulation
            if isinstance(object, BaseVehicle):
                object_nbrs = self.prepare_nbrs(vehicle=object, engine=engine) # Get nbrs in current timestep
                
                # Update the buffer with current information (absolute values)
                # Works for multistep transformations

                if hasattr(object, 'id') and hasattr(object, 'position'):
                    self.update_storage(
                        storage_buffer = self.observation_storage,
                        v_id = object.id,
                        current_pose = object.position,
                        nbrs = object_nbrs,
                    )
                else:
                    print("obj does not have attribute 'id' or 'position' ")

            else:
                pass
        
        all_ids = []
        for nbr_vehicle in self.observation_storage:
            if nbr_vehicle is not None:
                 all_ids.append(nbr_vehicle['id'])
            else:
                 pass

        if not self.observation_storage:
            self._hard_init_buffer(engine)

        # Only use current surrounding vehicles states in temp storage, and use for next step prediction 
        self.temp_storage = []


        
        for object in engine.get_objects().values():
            if object.id in ids:
                # Transform the past trajs according to current pose and rotation
                # And save it in the temp storage for the prediction
                temp_hist, temp_nbrs = self.transform_buffer(
                    storage_buffer = self.observation_storage,
                    current_vehicle = object,
                )
                

                self.temp_storage.append({
                    'id': object.id,
                    'hist': temp_hist,
                    "nbrs": temp_nbrs,
                    })
            else:
                pass
        return self.temp_storage
            


    def transform_buffer(self, storage_buffer, current_vehicle):

        """
        Transform the past trajs according to current pose and rotation
        """

        v_id = current_vehicle.id
        current_pose = current_vehicle.position
        current_rotation = current_vehicle.heading_theta

        self.translation_dict[v_id] = current_pose
        self.rotation_dict[v_id] = current_rotation


        temp_hist = []
        temp_nbrs = []
        for vehicle in storage_buffer:
            if vehicle['id'] == v_id:
                temp_hist = transform_trajectories(
                    trajectories_list = [vehicle['hist']],
                    now_point = current_pose,
                    theta = current_rotation,
                )
                temp_nbrs = transform_trajectories(
                    trajectories_list = vehicle['nbrs'],
                    now_point = current_pose,
                    theta = current_rotation,
                )

                # Generate neighbor array (3, 13, 30, 2)
                temp_nbrs, pir_list, r1, r2 = generate_nbrs_array(nbrs_list=temp_nbrs, pp=self.past_points)

                # Turn (3, 13, 30, 2) into (39, 30, 2)
                temp_nbrs = temp_nbrs.reshape(temp_nbrs.shape[0] * temp_nbrs.shape[1], temp_nbrs.shape[2], temp_nbrs.shape[3])

                temp_nbrs = np.swapaxes(temp_nbrs, 0, 1)



                # Create torch tensors and add batch dimension
                temp_hist = np.array(temp_hist)
                temp_nbrs = np.array(temp_nbrs)
                temp_hist = torch.from_numpy(temp_hist)
                temp_nbrs = torch.from_numpy(temp_nbrs)

                # All NaN to zeros
                torch.nan_to_num_(temp_hist, nan=0.0)
                torch.nan_to_num_(temp_nbrs, nan=0.0)

                break
            else:
                pass  # Do nothing if no match

         
        
        return temp_hist, temp_nbrs


    def prepare_nbrs(self, vehicle, engine):
        nbrs_array = []
    
        surrounding_vehicles = get_surrounding_vehicles(vehicle, engine)

        for nbr_vehicle in surrounding_vehicles:
            if nbr_vehicle is not None:
                 nbrs_array.append(nbr_vehicle.position)
            else:
                 nbrs_array.append([np.nan, np.nan]) 
        
        return nbrs_array
    
    def prepare_bevs(self, vehicle, engine):
        """
        Function to generate the BEV images
        """

    
    ####----- Online Learning -----#####
    # Online Learning for RL ego Vehicle








def generate_nbrs_array(nbrs_list, pp=30, window_size=[18, 78]):
    

    # Define window to identify neihbors
    r1 = [int(-i / 2) for i in window_size]  # [-9, -39]
    r2 = [int(i / 2) for i in window_size]
    nbrs = np.zeros((3, 13, pp, 2))
    pir_list = []

    for nbr in nbrs_list:
        if not np.isnan(nbr.all()):
            pir = point_in_rectangle(r1, r2, nbr)
            if pir:
                nbrs[pir] = nbr
                pir_list.append(pir)
        else:
            pass    
    return nbrs, pir_list, r1, r2





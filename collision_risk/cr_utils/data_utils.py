# Standard imports
import os
import pickle
import numpy as np
# Thrid party imports
import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path
import random

class CRDataset(Dataset):
    def __init__(self, file_path, img_path=None, enc_size=64, grid_size=(13, 3)):
        if isinstance(file_path, list):
            self.D = {"id": [], "hist": [], "fut": [], "nbrs": []}
            for file_p in file_path:
                with open(file_p, "rb") as fp:
                    d = pickle.load(fp)
                    for key in d:
                        self.D[key].extend(d[key])
        else:
            with open(file_path, "rb") as fp:
                self.D = pickle.load(fp)
        self.enc_size = enc_size  # size of encoder LSTM
        self.grid_size = grid_size  # size of social context grid

        self.img_folders = img_path  # List of folders to search for images

    def __len__(self):
        return len(self.D["hist"])
    
    def find_image(self, smpl_id):
        """Search for an image in the provided folders."""
        for folder in self.img_folders:
            img_path = os.path.join(folder, smpl_id + ".png")
            if os.path.exists(img_path):
                return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return None  # Image not found

    def __getitem__(self, idx):

        try: 
            # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
            smpl_id = self.D["id"][idx]
            hist = self.D["hist"][idx]
            fut = self.D["fut"][idx]

            nbrs = self.D["nbrs"][idx]  # shape (3, 13, 31, 2)
            neighbors = nbrs.reshape(
                nbrs.shape[0] * nbrs.shape[1], nbrs.shape[2], nbrs.shape[3]
            )  # shape (39, 31, 2)

            # Find the image across multiple folders
            sc_img = self.find_image(smpl_id)

            # Resize the image to 256x256
            sc_img = cv2.resize(sc_img, (256, 256))

            # Add channel dimension (1, 256, 256)
            sc_img = np.expand_dims(sc_img, axis=0)


            if sc_img is None:
                raise FileNotFoundError(f"Image for {smpl_id} not found in any folder.")

            return smpl_id, hist, fut, neighbors, sc_img
        
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return None  # or handle appropriately

    # Collate function for dataloader
    def collate_fn(self, samples):

        # Filter out None samples
        samples = [s for s in samples if s is not None]

        if len(samples) == 0:
            return None

        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for _, _, _, nbrs, _ in samples:
            nbr_batch_size += sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])
        len_in = len(samples[0][1])  # takes the length of hist of the first sample
        len_out = len(samples[0][2])  # takes the length of hist of the first sample
        nbrs_batch = torch.zeros(len_in, nbr_batch_size, 2)

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(len_in, len(samples), 2)
        fut_batch = torch.zeros(len_out, len(samples), 2)
        sc_img_batch = torch.zeros(len(samples), 1, 256, 256)

        count = 0
        smpl_ids = []
        for sampleId, (smpl_id, hist, fut, nbrs, sc_img) in enumerate(samples):

            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0 : len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
            hist_batch[0 : len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            fut_batch[0 : len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0 : len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            sc_img_batch[sampleId, :, :, :] = torch.from_numpy(sc_img)
            smpl_ids.append(smpl_id)

            # Set up neighbor, neighbor sequence length, and mask batches:
            for nbr in nbrs:
                if len(nbr) != 0:
                    nbrs_batch[0 : len(nbr), count, 0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0 : len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    count += 1

        return smpl_ids, hist_batch, nbrs_batch, fut_batch, sc_img_batch
    

def get_data_folders(directory, specific_file_name, random_sample=False, sample_num=100):
    end_folders = []
    folder_files = []

    for folder in Path(directory).rglob('*'):
        if folder.is_dir() and not any(subfolder.is_dir() for subfolder in folder.iterdir()):
            end_folders.append(str(folder))
            folder_files.append(str(f"{folder}/{specific_file_name}"))
    

    if random_sample:
        if len(end_folders) > sample_num:
            sample_num = sample_num
        else:
            print(f"Dataset size {len(end_folders)} are smaller than {sample_num}!")
            sample_num = len(end_folders)

        selected_indices = random.sample(range(len(end_folders)), sample_num)
        end_folders = [end_folders[i] for i in selected_indices]
        folder_files = [folder_files[i] for i in selected_indices]

    return end_folders, folder_files



def get_trajs_path(folder_paths, specific_file_name):
    folder_files = []
    for folder_path in folder_paths:
        folder = Path(folder_path)
        specific_file = folder / specific_file_name
        if folder.is_dir() and specific_file.is_file():  # Check if the file exists in the folder
            folder_files.append(specific_file)  # Store the full file path
    return folder_files
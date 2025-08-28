import csv
import os
import cv2
import numpy as np


class risk_logger:
    def __init__(self, file_path, fieldnames):
        self.file_path = file_path.rstrip("/")
        self.fieldnames = ["timestamp"] + fieldnames
        self.ts = 0
        self.epoch = 0
        self._has_logged = False
        self._setup_file()
        self.current_row_data = {}

    def _setup_file(self):
        os.makedirs(self.file_path, exist_ok=True)
        self.csv_path = f"{self.file_path}/Epi_{self.epoch}.csv"
        file_exists = os.path.isfile(self.csv_path)

        self.file = open(self.csv_path, mode="a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)

        if not file_exists:
            self.writer.writeheader()

    def log(self, data: dict):
        """
        Start a new row of data with a timestamp.
        """
        self.current_row_data = {"timestamp": self.ts, **data}
        self._has_logged = True

    def add_log(self, data: dict):
        """
        Add data to the current row and flush it to file.
        """
        if not self._has_logged:
            raise RuntimeError("You must call log() before add_log().")

        self.current_row_data.update(data)
        self.writer.writerow(self.current_row_data)
        self.file.flush()

        # Update internal state
        self.ts += 1
        self._has_logged = False
        self.current_row_data = {}

    def get(self, key):
        """
        Get a value from the current row (before it's flushed).
        """
        return self.current_row_data.get(key)

    def close(self):
        if hasattr(self, "file") and not self.file.closed:
            self.file.close()

    def reset(self):
        self.close()
        self.epoch += 1
        self.ts = 0
        self._setup_file()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()




class frame_logger:
    def __init__(self, base_path):
        """
        Initialize a frame logger to save rendered frames.
        
        Args:
            base_path: Base directory where frames will be stored
        """
        self.base_path = base_path.rstrip("/")
        self.ts = 0  # Timestamp/frame counter
        self.epoch = 0  # Episode counter
        self._setup_directory()
        
    def _setup_directory(self):
        """Create the directory structure for the current episode"""
        self.episode_dir = f"{self.base_path}/frames/Epi_{self.epoch}"
        os.makedirs(self.episode_dir, exist_ok=True)
    
    def save_frame(self, frame):
        """
        Save a frame to the current episode directory.
        
        Args:
            frame: Numpy array containing the image data (RGB format)
        """
        if frame is None:
            print(f"Warning: Attempted to save None frame at timestamp {self.ts}")
            return
            
        # Process the frame for saving with OpenCV
        # The input frame is expected to be a numpy array in RGB format
        frame_bgr = frame
        
        # If the frame is in RGB format, convert to BGR for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Check if dimensions need transposing based on PyGame surface conversion
            if frame.shape[0] < frame.shape[1]:  # Typical game frame is wider than tall
                frame_bgr = np.transpose(frame, axes=(1, 0, 2))
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = frame_bgr[:, :, ::-1]
        
        # Create the frame filename
        frame_path = os.path.join(self.episode_dir, f"frame_{self.ts:04d}.png")
        
        # Save the frame
        success = cv2.imwrite(frame_path, frame_bgr)
        if not success:
            print(f"Warning: Failed to save frame at {frame_path}")
            
        # Increment timestamp
        self.ts += 1
    
    def close(self):
        
        """Clean up resources if needed"""
        # Nothing to close for image saving, but keeping the method for consistency
        pass
    
    def reset(self):
        """Reset for a new episode"""
        self.close()
        self.epoch += 1
        self.ts = 0
        self._setup_directory()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
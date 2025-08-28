import os
from pathlib import Path
def get_folders_in_directory(directory):
    return [item for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))]



def get_end_folders(directory):
    end_folders = []
    for folder in Path(directory).rglob('*'):
        if folder.is_dir() and not any(subfolder.is_dir() for subfolder in folder.iterdir()):
            end_folders.append(str(folder))
    return end_folders

def join_folders_with_specific_file(folder_paths, specific_file_name):
    folder_files = []
    for folder_path in folder_paths:
        folder = Path(folder_path)
        specific_file = folder / specific_file_name
        if folder.is_dir() and specific_file.is_file():  # Check if the file exists in the folder
            folder_files.append(specific_file)  # Store the full file path
    return folder_files

# Example usage
directory = f"{Path.home()}/EDRL/collision_risk/data/raw_data/"
folders = get_end_folders(directory)

result = join_folders_with_specific_file(folders, "processed_traj.pkl")
print(result)


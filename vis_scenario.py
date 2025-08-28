import os
import pickle
from pathlib import Path

import tqdm

import metadrive.scenario.utils as sd_utils
from metadrive.engine.asset_loader import AssetLoader
from metadrive.scenario.scenario_description import ScenarioDescription as SD

# === Specify the path to the original and the new dataset ===
waymo_data = AssetLoader.file_path("/home/dianzhaoli/EDRL/filtered_dataset/evaluation_3/", unix_style=False)


dataset_path = waymo_data


# Read the summary
scenario_summaries, scenario_ids, dataset_mapping = sd_utils.read_dataset_summary(dataset_path)


print(scenario_summaries)

print(hi)

# Define a filter function that return True if the scenario is accepted
def filter(scenario):
    # Get the number of traffic light
    num_tl = len(scenario[SD.DYNAMIC_MAP_STATES])
    # You can also get the number from metadata (recommended)
    num_tl_from_metadata = scenario[SD.METADATA][SD.SUMMARY.NUMBER_SUMMARY][SD.SUMMARY.NUM_TRAFFIC_LIGHTS]
    assert num_tl_from_metadata == num_tl
    print(f"We found {num_tl} traffic lights in scenario {scenario[SD.ID]}")
    has_traffic_light = num_tl > 0
    return has_traffic_light

def filter_ped(scenario):
    num_ped = 0
    num_cyc = 0
    try:
        num_ped = scenario[SD.METADATA][SD.SUMMARY.NUMBER_SUMMARY]['num_objects_each_type']['PEDESTRIAN']
        num_cyc = scenario[SD.METADATA][SD.SUMMARY.NUMBER_SUMMARY]['num_objects_each_type']['CYCLIST']
    except KeyError:
        pass

    has_pedestrian_and_cyc = (num_ped > 5 and num_cyc > 5)
    return has_pedestrian_and_cyc

# Iterate over all scenarios
remove_scenario = []
new_mapping = {}
for file_name in tqdm.tqdm(scenario_summaries.keys(), desc="Filter Scenarios"):
    abs_path = Path(dataset_path) / dataset_mapping[file_name] / file_name
    abs_path_to_file_dir = Path(dataset_path) / dataset_mapping[file_name]

    # Call utility function in MD and get the ScenarioDescription object
    scenario = sd_utils.read_scenario_data(abs_path)

    # Filter
    accepted = filter_ped(scenario)
    #print(f"Processing: {file_name}. This scenario is {'accepted' if accepted else 'rejected'}.")
    if not accepted:
        remove_scenario.append(file_name)

    # Translate the relative path in mapping to the new location
    new_mapping[file_name] = os.path.relpath(abs_path_to_file_dir, new_dataset_path)
print("\n")

for file in remove_scenario:
    scenario_summaries.pop(file)
    new_mapping.pop(file)

summary_file_path = new_dataset_path / SD.DATASET.SUMMARY_FILE
with open(summary_file_path, "wb") as file:
    pickle.dump(scenario_summaries, file)
print(f"Summary file is saved at: {summary_file_path}")

mapping_file_path = new_dataset_path / SD.DATASET.MAPPING_FILE
with open(mapping_file_path, "wb") as file:
    pickle.dump(new_mapping, file)
print(f"Mapping file is saved at: {summary_file_path}")

# Verify
_, _, new_mapping = sd_utils.read_dataset_summary(new_dataset_path)
print(f"\nLet's verify if the new dataset is valid."
      f"\n{len(new_mapping)} scenarios are in the new dataset."
      f"\nThe new mapping:\n{new_mapping}")
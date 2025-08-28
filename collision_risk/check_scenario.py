from metadrive.scenario import utils as sd_utils


data_path = "/home/dianzhaoli/mdsn/dataset/scenario-nuplan-boston"

print(f"Reading the summary file from Waymo data at: {data_path}")

waymo_dataset_summary = sd_utils.read_dataset_summary(data_path)
print(f"The dataset summary is a {type(waymo_dataset_summary)}, with lengths {len(waymo_dataset_summary)}.")

waymo_scenario_summary, waymo_scenario_ids, waymo_scenario_files = waymo_dataset_summary

example_scenario_summary = waymo_scenario_summary['sd_nuplan_v1.1_9f09285a553a57f3.pkl']


print(f"The summary of a scenario is a dict with keys: {example_scenario_summary.keys()}")
print(example_scenario_summary["map_version"])
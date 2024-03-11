import json

jsonl_file_path = "data/track_storage.jsonl"

track_id_cache_dict = {}

with open(jsonl_file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        json_data = json.loads(line)
        track_id = json_data["track_id"]
        cache_type = json_data["storage_class"]
        track_id_cache_dict[track_id] = cache_type

with open("predictions_xgboost.json", "r") as file:
    xgboost_dict = json.load(file)

xgboost_dict = dict(sorted(xgboost_dict.items(), key=lambda item: item[1], reverse=True))

with open("predictions_pytorch.json", "r") as file:
    pytorch_dict = json.load(file)
    for key in pytorch_dict:
        pytorch_dict[key] = pytorch_dict[key]["prediction"]

pytorch_dict = dict(sorted(pytorch_dict.items(), key=lambda item: item[1], reverse=True))


xgboost_coherence = {
    "slow": 0,
    "medium": 0,
    "fast": 0
}

pytorch_coherence = {
    "slow": 0,
    "medium": 0,
    "fast": 0
}

base_dict = {
    "slow": 0,
    "medium": 0,
    "fast": 0
}

def divide_into_groups(input_dict):
    total_items = len(input_dict)
    fast_percentile = 22
    medium_percentile = 3346

    fast_dict = dict(sorted(input_dict.items(), key=lambda item: item[1], reverse=True)[:fast_percentile])
    medium_dict = dict(sorted(input_dict.items(), key=lambda item: item[1], reverse=True)[fast_percentile:fast_percentile + medium_percentile])
    slow_dict = dict(sorted(input_dict.items(), key=lambda item: item[1], reverse=True)[fast_percentile + medium_percentile:])

    return fast_dict, medium_dict, slow_dict

xgboost_fast, xgboost_medium, xgboost_slow = divide_into_groups(xgboost_dict)
pytorch_fast, pytorch_medium, pytorch_slow = divide_into_groups(pytorch_dict)


for key in track_id_cache_dict:
    base_dict[track_id_cache_dict[key]] += 1

    if track_id_cache_dict[key] == "fast" and key in xgboost_fast:
        xgboost_coherence["fast"] += 1
    if track_id_cache_dict[key] == "medium" and key in xgboost_medium:
        xgboost_coherence["medium"] += 1
    if track_id_cache_dict[key] == "slow" and key in xgboost_slow:
        xgboost_coherence["slow"] += 1

    if track_id_cache_dict[key] == "fast" and key in pytorch_fast:
        pytorch_coherence["fast"] += 1
    if track_id_cache_dict[key] == "medium" and key in pytorch_medium:
        pytorch_coherence["medium"] += 1
    if track_id_cache_dict[key] == "slow" and key in pytorch_slow:
        pytorch_coherence["slow"] += 1

print("Pytorch:")
print("Fast coherence: ", pytorch_coherence["fast"] / base_dict["fast"])
print("Medium coherence: ", pytorch_coherence["medium"] / base_dict["medium"])
print("Slow coherence: ", pytorch_coherence["slow"] / base_dict["slow"])

print("Xgboost:")
print("Fast coherence: ", xgboost_coherence["fast"] / base_dict["fast"])
print("Medium coherence: ", xgboost_coherence["medium"] / base_dict["medium"])
print("Slow coherence: ", xgboost_coherence["slow"] / base_dict["slow"])

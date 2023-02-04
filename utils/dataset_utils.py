import os

import pandas as pd
from tqdm import tqdm

from models.sign_model import SignModel
from utils.landmark_utils import save_landmarks_from_video, load_array


def load_dataset():
    dataset = [
        file_name.replace(".pickle", "").replace("pose_", "")
        for root, dirs, files in os.walk(os.path.join("data", "dataset"))
        for file_name in files
        if file_name.endswith(".pickle") and file_name.startswith("pose_")
    ]

    return dataset

def create_dataset():
    videos = [
        file_name.replace(".mp4", "")
        for root, dirs, files in os.walk(os.path.join("data", "videos"))
        for file_name in files
        if file_name.endswith(".mp4")
    ]
    dataset = [
        file_name.replace(".pickle", "").replace("pose_", "")
        for root, dirs, files in os.walk(os.path.join("data", "dataset"))
        for file_name in files
        if file_name.endswith(".pickle") and file_name.startswith("pose_")
    ]

    # Create the dataset from the reference videos
    videos_not_in_dataset = list(set(videos).difference(set(dataset)))
    n = len(videos_not_in_dataset)
    if n > 0:
        print(f"\nExtracting landmarks from new videos: {n} videos detected\n")
        video_cant_extract = []
        for idx in tqdm(range(n)):
            try: 
                save_landmarks_from_video(videos_not_in_dataset[idx])
            except:
                video_cant_extract.append(videos_not_in_dataset[idx])
                continue

        # while True:
        #     x = input("\nDelete videos can not be extracted? (y/n)\n").lower()
        #     if x == 'y':
        #         delete_list_of_videos(video_cant_extract)
        #         print(f"\nDeleted {len(video_cant_extract)} videos can not be extracted.\n")
        #         break
        #     elif x == 'n':
        #         break
        #     else: continue


def delete_list_of_videos(list):
    n = len(list)
    for i in range(n):
        os.remove(os.path.join("data", "videos", list[i].split("-")[0], list[i] + ".mp4"))


def load_reference_signs(videos):
    reference_signs = {"name": [], "sign_model": [], "distance": []}
    for video_name in videos:
        try:
            sign_name = video_name.split("-")[0]
            path = os.path.join("data", "dataset", sign_name, video_name)

            left_hand_list = load_array(os.path.join(path, f"lh_{video_name}.pickle"))
            right_hand_list = load_array(os.path.join(path, f"rh_{video_name}.pickle"))

            reference_signs["name"].append(sign_name)
            reference_signs["sign_model"].append(SignModel(left_hand_list, right_hand_list))
            reference_signs["distance"].append(0)
        except:
            continue
    
    reference_signs = pd.DataFrame(reference_signs, dtype=object)
    print(
        f'Dictionary count: {reference_signs[["name", "sign_model"]].groupby(["name"]).count()}'
    )
    return reference_signs

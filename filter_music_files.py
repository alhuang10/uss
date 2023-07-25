import argparse
import librosa
import time
import json
import torch
import os

from glob import glob
from tqdm import tqdm


ROOT_AUDIO_DIRECTORY = "/inworld/tts_dataset/podcast_audio_combined/wavs"

MUSIC_NORMALIZED_ENERGY_THRESHOLD = 0.1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker_name", type=str, required=True)
    args = parser.parse_args()

    speaker_name = args.speaker_name

    filepath_to_energy_levels_json = f"wav_file_and_label_to_energy_level_{speaker_name}.json"

    audio_directory = os.path.join(ROOT_AUDIO_DIRECTORY, speaker_name)

    print("Detecting WAV files...")
    wav_files = glob(f"{audio_directory}/**/*.wav", recursive=True)
    print(f"Number of wav files: {len(wav_files)}")

    # load json
    with open(filepath_to_energy_levels_json, "r") as f:
        wav_file_and_label_to_energy_level = json.load(f)

    wav_file_and_label_to_energy_level_dict = {x[0]:x[1] for x in wav_file_and_label_to_energy_level}

    num_files_not_found_error = 0
    num_human_zero_files = 0
    num_files_music_above_threshold = 0
    files_to_delete = []

    for f in wav_files:
        # if not found, means USS errored out, file is probably messesd up
        if f not in wav_file_and_label_to_energy_level_dict:
            files_to_delete.append(f)
            num_files_not_found_error += 1
        else:
            energy_dict = wav_file_and_label_to_energy_level_dict[f]
            human_sound_energy = energy_dict["Human sounds"]

            if human_sound_energy == 0:
                num_human_zero_files += 1
                files_to_delete.append(f)
            else:
                music_energy = energy_dict["Music"]
                if music_energy > MUSIC_NORMALIZED_ENERGY_THRESHOLD:
                    num_files_music_above_threshold += 1
                    files_to_delete.append(f)

    print(f"Number of files not found: {num_files_not_found_error}")
    print(f"Number of files with human sound energy 0: {num_human_zero_files}")
    print(f"Number of files with music energy above threshold: {num_files_music_above_threshold}")

    print(f"Speaker name: {speaker_name}")
    print(f"Number of files to delete: {len(files_to_delete)}")
    print(f"Number of original files: {len(wav_files)}")
    print(f"Ratio: {len(files_to_delete) / len(wav_files)}")

    print(f"Deleting files - Press enter to continue...")
    input()
    for f in tqdm(files_to_delete):
        os.remove(f)

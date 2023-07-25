import argparse
import librosa
import time
import json
import torch
import os

from glob import glob
from tqdm import tqdm

from uss.config import (ID_TO_IX, IX_TO_LB, LB_TO_IX, csv_paths_dict,
                        panns_paths_dict)
from uss.inference import separate, load_ss_model, load_pretrained_panns
from uss.utils import (get_audioset632_id_to_lb, get_path,
                       load_pretrained_panns, parse_yaml, remove_silence,
                       repeat_to_length)


AUDIO_NO_BACKGROUND_MUSIC = "/inworld/tts_dataset/podcast_host_audio_combined_filtered/alec_baldwin/wavs/amy_schumer_grew_up_in_a_nude_house/256.wav"

# ROOT_AUDIO_DIRECTORY/speaker_name
ROOT_AUDIO_DIRECTORY = "/inworld/tts_dataset/podcast_audio_combined/wavs"


_CONFIG_YAML = "./scripts/train/ss_model=resunet30,querynet=at_soft,data=full.yaml"
_CHECKPOINT_PATH = "pretrained.ckpt"


def get_normalized_energy(filepath):
    data, sample_rate = librosa.load(filepath)


    energy = sum(abs(data**2))

    duration = librosa.get_duration(y=data, sr=sample_rate)
    normalized_energy = energy / duration
    print("Normalized energy of the audio file:", normalized_energy)


class Args:
    def __init__(self, audio_path, output_dir):
        self.audio_path = audio_path
        self.condition_type = "at_soft"
        self.levels = [1]  # 1 is most coarse level
        self.class_ids = []
        self.queries_dir = ""
        self.query_emb_path = ""
        self.output_dir = output_dir

        # config, checkpoint TODO
        self.config_yaml = _CONFIG_YAML
        self.checkpoint_path = _CHECKPOINT_PATH



def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker_name", type=str, required=True)
    args = parser.parse_args()

    speaker_name = args.speaker_name

    audio_directory = os.path.join(ROOT_AUDIO_DIRECTORY, speaker_name)
    wav_files = glob(f"{audio_directory}/**/*.wav", recursive=True)

    print(f"Using audio directory: {audio_directory}")
    print(f"Number of wav files: {len(wav_files)}")

    wav_file_and_label_to_energy_level = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained universal source separation model
    configs = parse_yaml(_CONFIG_YAML)
    pl_model = load_ss_model(
        configs=configs,
        checkpoint_path=_CHECKPOINT_PATH,
    ).to(device)

    # Load pretrained audio tagging model
    at_model_type = "Cnn14"
    at_model = load_pretrained_panns(
        model_type=at_model_type,
        checkpoint_path=get_path(panns_paths_dict[at_model_type]),
        freeze=True,
    ).to(device)

    print(f"Working on {len(wav_files)} wav files")

    music_0_count = 0
    music_non_zero_count = 0
    error_count = 0

    for i, wav_file in enumerate(tqdm(wav_files)):
        input_args = Args(wav_file, "dummy_output_dir_not_used")
        try:
            label_to_energy_level = separate(input_args, pl_model, at_model)
        except Exception as e:
            error_count += 1
            continue

        wav_file_and_label_to_energy_level.append((wav_file, label_to_energy_level))
        
        # Track ratio to see if we should stop a run
        if label_to_energy_level["Music"] > 0:
            music_non_zero_count += 1
        else:
            music_0_count += 1

        if i % 100 == 0:
            print(f"Music 0 count: {music_0_count}")
            print(f"Music non-zero count: {music_non_zero_count}")
            print(f"Ratio: {music_non_zero_count / (music_0_count + music_non_zero_count)}")
            print(f"Error count: {error_count}")


            # dump to json
            with open(f"wav_file_and_label_to_energy_level_{speaker_name}.json", "w") as f:
                json.dump(wav_file_and_label_to_energy_level, f)

    # TODO: some sort of recovery?
    # dump to json
    with open(f"wav_file_and_label_to_energy_level_{speaker_name}.json", "w") as f:
        json.dump(wav_file_and_label_to_energy_level, f)


if __name__ == "__main__":
    main()
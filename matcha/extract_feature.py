import argparse
import csv
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict
# import get_f0
import numpy as np
from threadpoolctl import threadpool_limits
from tqdm import tqdm
import re

from torchtts.data.core.audio import load_wav
from torchtts.data.core.audio import mel_spectrogram
from utils import load_metadata
from utils import stringify_metadata


def get_args():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ["true", "false"]:
            raise ValueError("Argument needs to be a " "boolean, got {}".format(s))
        return {"true": True, "false": False}[s.lower()]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Metadata file extracted from previous step",
    )
    parser.add_argument(
        "--mel_output",
        type=Path,
        default="mel",
        help="Output folder for Mel spectrogram",
    )
    parser.add_argument(
        "--sf_output",
        type=Path,
        default="sf",
        help="Output folder for interpolated f0",
    )
    parser.add_argument(
        "--uv_output",
        type=Path,
        default="uv",
        help="Output folder for uv",
    )
    parser.add_argument(
        "--metadata_output",
        type=Path,
        default="metadata.csv",
        help="Updated metadata output file",
    )
    parser.add_argument(
        "--pre_emphasis_coeff",
        type=float,
        default=0.97,
        help="Pre-emphasis coefficient",
    )
    parser.add_argument(
        "--num_freq",
        type=int,
        default=1025,
        help="Number of frequency, which equals fft_size / 2 + 1",
    )
    parser.add_argument(
        "--frame_shift_in_ms",
        type=float,
        default=12.5,
        help="Frame shift in ms for Mel extraction",
    )
    parser.add_argument(
        "--frame_length_in_ms",
        type=float,
        default=50,
        help="Frame length in ms for Mel extraction",
    )
    parser.add_argument(
        "--mel_sample_rate",
        type=int,
        default=16000,
        help="Sample rate of waveform which is used for Mel extraction",
    )
    parser.add_argument(
        "--num_mels",
        type=int,
        default=80,
        help="Number of Mel bands to use",
    )
    parser.add_argument(
        "--min_mel_freq",
        type=int,
        default=0,
        help="The minimum Mel filter frequency",
    )
    parser.add_argument(
        "--max_mel_freq",
        type=int,
        default=8000,
        help="The maximum Mel filter frequency",
    )
    parser.add_argument(
        "--ref_level_db",
        type=float,
        default=0,
        help="The reference db level for Mel spectrogram normalization",
    )
    parser.add_argument(
        "--min_level_db",
        type=float,
        default=-100,
        help="The minimum db level for Mel spectrogram normalization",
    )
    parser.add_argument(
        "--symmetric_specs",
        type=_str_to_bool,
        default=True,
        help="Whether to use normalize Mel spectrogram symmetrically",
    )
    parser.add_argument(
        "--max_abs_value",
        type=float,
        default=4,
        help="Maximum absolute value of normalized Mel spectrogram",
    )
    parser.add_argument(
        "--clip_norm",
        type=_str_to_bool,
        default=True,
        help="Whether to clip normalize Mel spectrogram",
    )
    parser.add_argument(
        "--min_pitch",
        type=float,
        default=50,
        help="The lower bound of extracted pitch",
    )
    parser.add_argument(
        "--max_pitch",
        type=float,
        default=800,
        help="The upper bound of extracted pitch",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Max workers to use for feature extraction",
    )
    parser.add_argument(
        "--restrict_pause",
        type=_str_to_bool,
        default=True,
        help="whether to restrict pause (br, /, punc) during label generation.",
    )
    parser.add_argument(
        "--use_br",
        type=_str_to_bool,
        default=True,
        help="whether to use br phones, if not, just use word boundary /",
    )
    return parser.parse_args()


def main(args):
    os.makedirs(args.mel_output, exist_ok=True)
    os.makedirs(args.sf_output, exist_ok=True)
    os.makedirs(args.uv_output, exist_ok=True)

    futures = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit parallel feature extraction tasks
        for metadata_item in load_metadata(args.metadata):
            futures.append(executor.submit(extract_features, args, metadata_item))

        # Collect parallel execution results
        with open(args.metadata_output, "w", newline="", encoding="utf-8") as f:
            metadata_writer = csv.writer(f, delimiter="|")
            header_written = False
            # Collect updated metadata items
            for future in as_completed(futures):
                updated_metadata_item = future.result()
                # Stringify items in metadata (for example, list)
                metadata_item = stringify_metadata(updated_metadata_item)
                # Write csv header first if not written before
                if not header_written:
                    metadata_writer.writerow(metadata_item.keys())
                    header_written = True
                # Write real data
                metadata_writer.writerow(metadata_item.values())


def extract_features(args: argparse.Namespace, metadata: Dict):
    # We need this to limit the threads of common native libraries used by numpy for scientific
    # computing and data science (e.g. BLAS and OpenMP). Otherwise, each worker may have risk
    # of draining system resources.
    with threadpool_limits(limits=1, user_api="blas"):
        basename = metadata["speech_path"].stem
        speech, _ = load_wav(metadata["speech_path"], args.mel_sample_rate)

        mel_spec = mel_spectrogram(
            audio=speech,
            pre_emphasis_coeff=args.pre_emphasis_coeff,
            num_freq=args.num_freq,
            frame_shift_ms=args.frame_shift_in_ms,
            frame_length_ms=args.frame_length_in_ms,
            sample_rate=args.mel_sample_rate,
            num_mels=args.num_mels,
            min_mel_freq=args.min_mel_freq,
            max_mel_freq=args.max_mel_freq,
            ref_level_db=args.ref_level_db,
            symmetric_specs=args.symmetric_specs,
            max_abs_value=args.max_abs_value,
            min_level_db=args.min_level_db,
            clip_norm=args.clip_norm,
        )

        metadata["mel_path"] = args.mel_output / f"{basename}.npy"
        np.save(metadata["mel_path"], mel_spec)

        f0 = get_f0.extract_pitch(
            speech,
            args.mel_sample_rate,
            min_pitch=args.min_pitch,
            max_pitch=args.max_pitch,
            frame_duration=args.frame_shift_in_ms / 1000,
        )
        uv = get_vuv(f0)
        sf = interpolate_f0(f0)

        metadata["sf_path"] = args.sf_output / f"{basename}.npy"
        np.save(metadata["sf_path"], sf)

        metadata["uv_path"] = args.uv_output / f"{basename}.npy"
        np.save(metadata["uv_path"], uv)

        # update metadata: adjust pause, breaks, etc
        if not args.restrict_pause:
            metadata["phones"], metadata["durations"] = relax_pause(metadata["phones"], metadata["durations"])

        if not args.use_br:
            metadata["phones"], metadata["durations"] = remove_break(metadata["phones"], metadata["durations"])

        return metadata


def relax_pause(in_phone_list, in_dur_list):
    out_phone_str, out_dur_list = "", []
    split_phone_list = re.split(r"br\d sil br\d", " ".join(in_phone_list))
    pre_phone_num = 0
    for idx, phone in enumerate(split_phone_list):
        if idx == len(split_phone_list) - 1:
            out_phone_str += phone.strip()
            out_dur_list += in_dur_list[pre_phone_num:]
            break

        phone_list = phone.strip().split(' ')
        # sum duration of br\d sil br\d
        sil_dur_sum = sum(in_dur_list[pre_phone_num + len(phone_list):pre_phone_num + len(phone_list) + 3])

        if phone_list[-1].find('punc') >= 0:
            # if punc* br\d sil br\d, then phones: punc* br0, durations: dur_punc* + dur_sum 0 (align with runtime)
            out_phone_str += phone.strip() + " br0 "
            punc_dur = in_dur_list[pre_phone_num + len(phone_list) - 1] + sil_dur_sum
            out_dur_list += in_dur_list[pre_phone_num:pre_phone_num + len(phone_list) - 1] + [punc_dur, 0]
        else:
            # else phones: br3, durations: dur_sum
            out_phone_str += phone.strip() + " br3 "
            out_dur_list += in_dur_list[pre_phone_num:pre_phone_num + len(phone_list)] + [sil_dur_sum]

        pre_phone_num += len(phone_list) + 3

    assert len(out_phone_str.split(' ')) == len(out_dur_list)
    return out_phone_str.split(' '), out_dur_list


def remove_break(in_phone_list, in_dur_list):
    if re.match("^br\d$", in_phone_list[1]): # remove first br phone between -bos- and first phone
        if in_phone_list[1] != "br0":
            logging.warning(f"Second phone is not br0, but {in_phone_list[1]} with duration = {in_dur_list[1]}")
        in_phone_list = [in_phone_list[0]] + in_phone_list[2:]
        in_dur_list = [in_dur_list[0] + in_dur_list[1]] + in_dur_list[2:]
    if re.match("^br\d$", in_phone_list[-2]): # remove last br phone between last punc and -eos-
        if in_phone_list[-2] != "br0":
            logging.warning(f"Penultimate phone is not br0, but {in_phone_list[-2]} with duration = {in_dur_list[-2]}")
        in_phone_list = in_phone_list[:-2] + [in_phone_list[-1]]
        in_dur_list = in_dur_list[:-2] + [in_dur_list[-2] + in_dur_list[-1]]
    out_phone_list = re.sub(r'br\d', '/', " ".join(in_phone_list)).split(' ')
    return out_phone_list, in_dur_list


def get_vuv(f0):
    vuv = f0.copy()
    vuv[vuv > 0.0] = 1.0
    vuv[vuv <= 0.0] = 0.0
    return vuv


def interpolate_f0(f0):
    inter_f0 = f0
    frame_number = f0.size
    last_value = 0.0
    for i in range(frame_number):
        if f0[i] <= 0.0:
            j = i + 1
            for j in range(i + 1, frame_number):
                if f0[j] > 0.0:
                    break
            if j < frame_number - 1:
                if last_value > 0.0:
                    step = (f0[j] - f0[i - 1]) / float(j - i + 1)
                    for k in range(i, j):
                        inter_f0[k] = f0[i - 1] + step * (k - i + 1)
                else:
                    for k in range(i, j):
                        inter_f0[k] = f0[j]
            else:
                for k in range(i, frame_number):
                    inter_f0[k] = last_value
        else:
            inter_f0[i] = f0[i]
            last_value = f0[i]
    return inter_f0


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main(get_args())

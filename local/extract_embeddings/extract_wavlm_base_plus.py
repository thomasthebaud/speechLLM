from pathlib import Path
import math
import sys
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoModel, AutoFeatureExtractor
import os

#!/usr/bin/env python3
"""
extract_wavlm_base_plus.py

Usage:
    python extract_wavlm_base_plus.py <csv1> <csv2> ...

Given one or more CSV files, load each CSV with pandas. For each row,
load the audio at column 'audio_path', split into non-overlapping chunks
of 30 seconds, extract representations using pretrained
"microsoft/wavlm-base-plus" and save chunk representations under:

    /export/fs05/tthebau1/EDART/wavlm_base_plus/<dataset_name>/

Where dataset_name is the stem of the CSV filename.
"""



CHUNK_SECONDS = 30
TARGET_SR = 16000
OUTPUT_ROOT = Path("/export/fs05/tthebau1/EDART/wavlm_base_plus")

def load_audio(path: Path, target_sr: int = TARGET_SR):
        waveform, sr = torchaudio.load(str(path))
        # convert to mono
        if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # shape (num_samples,)
        if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)
                sr = target_sr
        # ensure float32
        if waveform.dtype != torch.float32:
                waveform = waveform.to(torch.float32)
        return waveform, sr

def process_csv(csv_path: Path, model, device):
        dataset_name = csv_path.stem
        out_dir = OUTPUT_ROOT / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(csv_path)
        if "audio_path" not in df.columns:
                print(f"Skipping {csv_path}: no 'audio_path' column", file=sys.stderr)
                return

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_name}"):
                
                
                audio_path = Path(row["audio_path"])
                out_name = f"{audio_path.stem}.npy"
                out_path = out_dir / out_name
                if os.path.exists(out_path):continue  # skip if already processed
                if not audio_path.exists():
                        print(f"Missing audio: {audio_path} (row {idx})", file=sys.stderr)
                        continue

                try:
                        waveform, sr = load_audio(audio_path, target_sr=TARGET_SR)
                except Exception as e:
                        print(f"Error loading {audio_path}: {e}", file=sys.stderr)
                        continue

                num_samples = waveform.shape[0]
                chunk_size = CHUNK_SECONDS * sr
                n_chunks = math.ceil(num_samples / chunk_size)

                
                chunk_reps = []
                for ci in range(n_chunks):
                        start = ci * chunk_size
                        end = min((ci + 1) * chunk_size, num_samples)
                        chunk = waveform[start:end]
                        # pad last chunk to chunk_size with zeros (optional, keeps representation size stable)
                        if chunk.shape[0] < chunk_size:
                                pad = torch.zeros(chunk_size - chunk.shape[0], dtype=chunk.dtype)
                                chunk = torch.cat([chunk, pad], dim=0)

                        # prepare tensor for model: (batch=1, time)
                        inputs = chunk.unsqueeze(0).to(device)

                        with torch.no_grad():
                                outputs = model(inputs).last_hidden_state  # (1, seq_len, hidden)
                        reps = outputs.squeeze(0).cpu().numpy()  # (seq_len, hidden)

                        chunk_reps.append(reps)

                # concatenate along time axis and save a single file per audio
                if len(chunk_reps) > 0:
                        concatenated = np.concatenate(chunk_reps, axis=0)
                        np.save(out_path, concatenated)
                        # free memory
                        del chunk_reps
        return

def main():
        csv_list = ['AMI_test.csv', 'AMI_train.csv', 'AMI_val.csv']
        csv_list = ['ICSI_test.csv', 'ICSI_train.csv', 'ICSI_val.csv']
        # csv_list = ['switchboard_test.csv', 'switchboard_train.csv', 'switchboard_val.csv']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "microsoft/wavlm-base-plus"

        # Load model (AutoModel returns appropriate WavLM model)
        print("Loading model...", file=sys.stderr)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        model.to(device)

        # feature extractor isn't strictly necessary for basic raw input feeding,
        # but keeping here in case of future normalization use.
        try:
                _ = AutoFeatureExtractor.from_pretrained(model_name)
        except Exception:
                pass

        for csv in csv_list:
                csv_path = Path('/home/tthebau1/EDART/SpeechLLM/data/'+csv)
                if not csv_path.exists():
                        print(f"CSV not found: {csv}", file=sys.stderr)
                        continue
                process_csv(csv_path, model, device)

if __name__ == "__main__":
        main()
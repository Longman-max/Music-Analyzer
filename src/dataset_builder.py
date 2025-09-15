import os
import pandas as pd
import numpy as np
from pathlib import Path
from feature_extraction import extract_features


def build_dataset_from_folder(root_dir="data/raw",
                              meta_csv=None,
                              out_csv="data/processed/features.csv",
                              sr=22050,
                              save_melspec_dir="data/processed/melspecs"):
    """
    Build dataset of features from audio files and save mel-spectrograms
    in genre-labeled subfolders for CNN training.

    - If `meta_csv` provided, expects columns: filename,label
    - Otherwise, assumes `root_dir` has subfolders, one per label (genre).
    - Features are extracted via `extract_features` and expanded into columns.
    - Mel-spectrogram images are saved in subfolders under `save_melspec_dir`.
    """

    records = []
    root = Path(root_dir)

    if meta_csv:
        meta = pd.read_csv(meta_csv)
        for _, row in meta.iterrows():
            fp = root / row["filename"]
            label = row["label"]
            try:
                img_path = None
                if save_melspec_dir:
                    img_path = Path(save_melspec_dir) / label / (fp.stem + ".png")
                    img_path.parent.mkdir(parents=True, exist_ok=True)

                feat = extract_features(
                    str(fp),
                    sr=sr,
                    save_melspec_path=str(img_path) if img_path else None
                )

                records.append(dict(filename=str(fp), label=label, features=feat))
            except Exception as e:
                print("FAILED", fp, e)
    else:
        # Each subfolder = label (genre)
        for label_dir in root.iterdir():
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for audio in label_dir.glob("*"):
                try:
                    img_path = None
                    if save_melspec_dir:
                        img_path = Path(save_melspec_dir) / label / (audio.stem + ".png")
                        img_path.parent.mkdir(parents=True, exist_ok=True)

                    feat = extract_features(
                        str(audio),
                        sr=sr,
                        save_melspec_path=str(img_path) if img_path else None
                    )

                    records.append(dict(filename=str(audio), label=label, features=feat))
                except Exception as e:
                    print("FAILED", audio, e)

    # Convert to DataFrame
    df = pd.DataFrame(records)

    if df.empty:
        print("No features extracted! Check your dataset path.")
        return df

    # Expand feature arrays into separate columns
    feat_arr = np.stack(df["features"].values)
    feat_cols = [f"f{i}" for i in range(feat_arr.shape[1])]
    feat_df = pd.DataFrame(feat_arr, columns=feat_cols)

    df = pd.concat(
        [df[["filename", "label"]].reset_index(drop=True),
         feat_df.reset_index(drop=True)],
        axis=1
    )

    # Save dataset CSV
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Saved features to", out_csv)

    return df


if __name__ == "__main__":
    build_dataset_from_folder()

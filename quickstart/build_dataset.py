import argparse
import os
import joblib

from data import ExcelDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an ExcelDataset object from MassLinker tokenized Excel outputs."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help=(
            "Directory containing MassLinker tokenized outputs. "
            "Each sample is expected to be stored as a subdirectory containing an xlsx file."
        ),
    )

    parser.add_argument(
        "--target_path",
        type=str,
        default="target.xlsx",
        help=(
            "Path to the target Excel file. "
            "The file should contain at least filename, label, and is_positive columns."
        ),
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/processed_dataset.joblib",
        help="Output path for the serialized ExcelDataset object.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="origin",
        choices=["origin", "enhance"],
        help=(
            "Dataset loading mode. Use 'origin' for original samples and 'enhance' "
            "for augmented samples whose filenames contain augmentation prefixes."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("Building ExcelDataset...")
    print(f"Data path:   {args.data_path}")
    print(f"Target path: {args.target_path}")
    print(f"Mode:        {args.mode}")

    dataset = ExcelDataset(
        data_path=args.data_path,
        target_path=args.target_path,
        mode=args.mode,
    )

    dataset.gen_classes()

    joblib.dump(dataset, args.output_path)

    print(f"Dataset saved to: {args.output_path}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Sample tensor shape: {dataset.samples.shape}")


if __name__ == "__main__":
    main()

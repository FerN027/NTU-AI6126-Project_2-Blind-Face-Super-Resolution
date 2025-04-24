import argparse

# initialize a parser
def get_parser():
    parser = argparse.ArgumentParser(

    )

    # Expected directory structure:
    # input_dir/
    # ├── train/
    # │   ├── GT/           # High-quality ground truth images
    # │   └── meta.txt      # text file containing the names
    # └── val/
    #     ├── GT/           # Validation ground truth
    #     └── LQ/           # Validation low-quality
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        required=True,
        default="./data",
        help="Directory containing the dataset. It should contain train and val folders.",
    )

    # where to save the model weights (.pth)
    parser.add_argument(
        "--save_model", "-w",
        type=str,
        required=True,
        default="./model.pth",
        help=".../.../.../model.pth",
    )

    # device string: choose from "cuda" or "cpu"
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        help="Device to use for training. Choose from 'cuda' or 'cpu'.",
    )

    # number of epochs to train the model for
    parser.add_argument(
        "--num_epochs", "-ep",
        type=int,
        default=100,
        help="Number of epochs to train the model.",
    )

    # batch size for training
    parser.add_argument(
        "--batch_size", "-bs",
        type=int,
        default=8,
        help="Batch size for training.",
    )

    # learning rate for the optimizer
    parser.add_argument(
        "--learning_rate", "-lr",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer.",
    )

    # number of workers for the data loader
    parser.add_argument(
        "--num_workers", "-nw",
        type=int,
        default=4,
        help="Number of workers for the data loader.",
    )

    return parser
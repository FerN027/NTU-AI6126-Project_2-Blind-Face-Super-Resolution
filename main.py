import torch

from utils import opt, get_parser, modify_opt, degradation
from ffhqsub_dataset import FFHQsubDataset
from data_loader import get_data_loader
from val_400_dataset import ValidationDataset
from test_400_dataset import TestDataset
from model import *
from train import train
from test import inference

if __name__ == "__main__":
    # Adjust the options based on the command line arguments
    parser = get_parser()
    args = parser.parse_args()
    opt = modify_opt(args, opt)

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    )
    print("Using device:", device)

    # ----------------------------------------------------------------------
    # Only define the model once here for good!
    model = RCAN()
    # ----------------------------------------------------------------------

    # # test case
    # model = model.to(device)

    # lq_train = degradation(
    #     next(iter(train_loader)), 
    #     opt, 
    #     device
    # )['lq']

    # sr_train = model(lq_train)
    # print(sr_train.shape)
    # print(sr_train[0].max(), sr_train[0].min())


    if not args.inferring:  # train and save weights
        # Initialize train and validation data loaders
        train_loader = get_data_loader(
            'train',
            FFHQsubDataset(opt),
            args
        )
        val_loader = get_data_loader(
            'val',
            ValidationDataset(opt),
            args
        )

        print("Training...")
        train(model, args, opt, device, train_loader, val_loader)
        print("Done training!")

    else:  # load model werights and do the inferring
        test_loader = get_data_loader(
            'test',
            TestDataset(opt, args),
            args
        )

        print("Inferring...")
        inference(model, args, device, test_loader)

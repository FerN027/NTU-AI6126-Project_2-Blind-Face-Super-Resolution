import torch

from utils import opt, get_parser, modify_opt, degradation
from ffhqsub_dataset import FFHQsubDataset
from data_loader import get_data_loader
from val_400_dataset import ValidationDataset
from model import *
from train import train

if __name__ == "__main__":
    # Adjust the options based on the command line arguments
    parser = get_parser()
    args = parser.parse_args()
    opt = modify_opt(args, opt)

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    )
    print("Using device:", device)

    # Initialize train & validation data loaders
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


    # # test case
    # model = BlindSR()

    # model = model.to(device)

    # lq_train = degradation(
    #     next(iter(train_loader)), 
    #     opt, 
    #     device
    # )['lq']

    # sr_train = model(lq_train)
    # print(sr_train.shape)
    # print(sr_train[0].max(), sr_train[0].min())

    
    train(args, opt, device, train_loader, val_loader)
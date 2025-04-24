from torch.utils.data import DataLoader

def get_data_loader(which_loader: str, dataset, args):

    if which_loader == 'train':
        return DataLoader(
            dataset,
            shuffle=True,
            # pin_memory=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    elif which_loader == 'val':
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    else:
        raise ValueError(f"Unknown data loader type: {which_loader}. Expected 'train' or 'val'.")
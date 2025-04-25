import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import degradation, compute_psnr


def train(model, args, opt, device, train_loader, val_loader):
    """
    0. Initialize the model and move it to the specified device
    """
    model = model.to(device)


    """
    1. Set hyperparameters
    """
    epochs = args.num_epochs
    lr = args.learning_rate

    criterion = nn.L1Loss()

    optimizer = optim.Adam(
        model.parameters(), 
        lr=lr,
        weight_decay=0,
        betas=(0.9, 0.99)
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,  # Use full number of epochs as period
        eta_min=1e-7   # Minimum learning rate from config
    )


    """
    2. Training loop
    """
    best_val_psnr = 0.0
    num_val_images = len(val_loader.dataset)

    for ith_epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for idx, batch_dict in enumerate(train_loader):
            # Apply degradation to create LQ-GT pairs
            processed_batch = degradation(batch_dict, opt, device)
            
            # Get inputs and targets
            inputs = processed_batch['lq']
            targets = processed_batch['gt']
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate batch loss
            epoch_loss += loss.item()

        # One training epoch completed
        scheduler.step()
        avg_epoch_loss = epoch_loss / len(train_loader)


        """
        3. Validation step
        """
        all_psnrs = []

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                # Get LQ and GT images
                lq = batch['lq'].to(device)
                gt = batch['gt'].to(device)
                
                # Forward pass
                sr = model(lq)
                
                # Calculate PSNR
                batch_psnr_list = compute_psnr(sr, gt)
                all_psnrs.extend(batch_psnr_list)

        # Calculate average PSNR for the validation set
        assert len(all_psnrs) == num_val_images, "Mismatch in number of PSNR values"
        this_psnr = np.mean(all_psnrs)

        # Decide whether to save the current model
        SAVED = False

        if this_psnr > best_val_psnr:
            best_val_psnr = this_psnr
            torch.save(model.state_dict(), args.save_model)

            SAVED = True


        """
        4. Print training and validation results of this epoch
        """
        print(f'Epoch [{ith_epoch}/{epochs}], Train Loss: {avg_epoch_loss:.6f}, Val_PSNR: {this_psnr:.4f}, ~{time.time()-start_time:.2f}s, Saved: {SAVED}')
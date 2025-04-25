import os
import numpy as np
import torch
from PIL import Image

def inference(model, args, device, test_loader):
    save_dir = args.test_output_dir    

    # load model weights
    weights_path = args.save_model
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    print("Model weights loaded successfully.")

    # make predictions
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # Get LQ images and filenames
            lq = batch['lq'].to(device)
            filenames = batch['filename']
            
            # Forward pass
            sr = model(lq)
            
            # Process and save each image in the batch
            for i, filename in enumerate(filenames):
                # Convert tensor to numpy image (0-255)
                sr_img = sr[i].detach().cpu().clamp(0, 1).numpy()
                sr_img = sr_img.transpose(1, 2, 0) * 255.0  # CHW to HWC, scale to [0, 255]
                sr_img = sr_img.astype(np.uint8)
                
                # Save using PIL (same filename as input)
                sr_pil = Image.fromarray(sr_img)
                sr_pil.save(os.path.join(save_dir, filename))
    
    print(f"Inference completed. Results saved to {save_dir}")
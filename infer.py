import cv2
import glob
import numpy as np
import os
import torch

from basicsr.archs.srresnet_arch import MSRResNet

# configuration
####### Modify to your paths
model_path = 'experiments/train_SRResNet_x4_FFHQ_300k/models/net_g_latest.pth'
folder = './data/FFHQ/test/LQ'
output_path = 'results/train_SRResNet_x4_FFHQ_300k'
############################

device = 'cuda'
device = torch.device(device)

# set up model
model = MSRResNet(
    num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4)
print(f'Number of Params: {sum(p.numel() for p in model.parameters())}')
model.load_state_dict(torch.load(model_path)['params'], strict=True)
model.eval()
model = model.to(device)

os.makedirs(output_path, exist_ok=True)
for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
    imgname = os.path.splitext(os.path.basename(path))[0]
    print(idx, imgname)
    # read image
    img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]],
                                        (2, 0, 1))).float()
    img = img.unsqueeze(0).to(device)
    # inference
    with torch.no_grad():
        output = model(img)
    # save image
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    cv2.imwrite(f'{output_path}/{imgname}.png', output)
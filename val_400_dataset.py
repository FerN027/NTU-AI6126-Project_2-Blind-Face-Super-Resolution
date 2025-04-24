import os
import torch
from basicsr.utils import FileClient, imfrombytes, img2tensor
from torch.utils.data import Dataset

class ValidationDataset(Dataset):
    """Validation dataset for paired LQ-HQ images.
    
    Processes images exactly the same way as the FFHQsubDataset.
    Includes checks to ensure LQ and HQ images are properly paired.
    """
    
    def __init__(self, opt):
        super(ValidationDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.lq_folder = opt['val_lq']
        self.gt_folder = opt['val_gt']
        
        # Check that the folders exist
        if not os.path.exists(self.lq_folder):
            raise ValueError(f"LQ folder does not exist: {self.lq_folder}")
        if not os.path.exists(self.gt_folder):
            raise ValueError(f"GT folder does not exist: {self.gt_folder}")
        
        # Get image list - only include files that exist in both folders
        all_lq_files = [f for f in os.listdir(self.lq_folder) if f.endswith('.png')]
        
        self.image_files = []
        for lq_file in all_lq_files:
            gt_file = lq_file  # Same filename expected in GT folder
            if os.path.exists(os.path.join(self.gt_folder, gt_file)):
                self.image_files.append(lq_file)
        
        # Sort for consistent ordering
        self.image_files.sort()
        
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type', 'disk'), **self.io_backend_opt)
        
        # Get image filename
        img_name = self.image_files[index]
        
        # Get paths
        lq_path = os.path.join(self.lq_folder, img_name)
        gt_path = os.path.join(self.gt_folder, img_name)
        
        # Read LQ image - same as FFHQsubDataset
        img_bytes = self.file_client.get(lq_path)
        img_lq = imfrombytes(img_bytes, float32=True)
        
        # Read GT image - same as FFHQsubDataset
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)
        
        # BGR to RGB, HWC to CHW, numpy to tensor - EXACTLY as in FFHQsubDataset
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]
        
        # Return just the LQ and GT tensors - minimal return
        return {
            'lq': img_lq,
            'gt': img_gt
        }

    def __len__(self):
        return len(self.image_files)
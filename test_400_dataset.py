import os
import torch
from basicsr.utils import FileClient, imfrombytes, img2tensor
from torch.utils.data import Dataset


class TestDataset(Dataset):
    """Test dataset for blind face super-resolution.
    
    Only requires LQ images folder.
    Processes images exactly the same way as training and validation.
    Records filenames for saving outputs with matching names.
    """
    
    def __init__(self, opt, args):
        super(TestDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.lq_folder = args.test_input_dir
        
        # Check that the folder exists
        if not os.path.exists(self.lq_folder):
            raise ValueError(f"Test LQ folder does not exist: {self.lq_folder}")
        
        # Get image list - all PNG images
        self.image_files = sorted([f for f in os.listdir(self.lq_folder) if f.endswith('.png')])
        
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type', 'disk'), **self.io_backend_opt)
        
        # Get image filename
        img_name = self.image_files[index]
        
        # Get path
        lq_path = os.path.join(self.lq_folder, img_name)
        
        # Read LQ image - same as in validation/training
        img_bytes = self.file_client.get(lq_path)
        img_lq = imfrombytes(img_bytes, float32=True)
        
        # BGR to RGB, HWC to CHW, numpy to tensor - EXACTLY as in other datasets
        img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]
        
        # Return LQ tensor and filename (for saving outputs)
        return {
            'lq': img_lq,
            'filename': img_name  # Store filename for saving outputs later
        }

    def __len__(self):
        return len(self.image_files)
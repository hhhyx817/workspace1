import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from time import time
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import os

class CelebADataset(Dataset):
    def __init__(self, root, img_shape=(64,64)) -> None:
        super().__init__()
        self.root = root
        self.img_shape = img_shape
        self.filesnames = sorted(os.listdir(root))
    
    def __len__(self) -> int:
        return len(self.filesnames)
    
    def __getitem__(self, index: int):
        path = os.path.join(self.root, self.filesnames[index])
        img = Image.open(path).convert("RGB")
        pipeline = transforms.Compose([
            transforms.CenterCrop(168),
            transforms.Resize(self.img_shape),
            transforms.ToTensor()
        ])       
        return pipeline(img)
        
def get_dataloader(root="/data/home/huangyx/workspace/diffusion_learn/VAE/img_align_celeba/", **kwargs):
    dataset = CelebADataset(root, **kwargs)
    return DataLoader(dataset, 16, shuffle=True)


import logging
from torch.utils.data import Dataset
from pathlib import Path
from os import listdir
from os.path import splitext
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class CarvanaDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mask_suffix = "_mask"
        self.scale = scale
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        logging.info(f"Creating dataset with {len(self.ids)} examples")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + ".*"))
        img_file = list(self.images_dir.glob(name + ".*"))

        assert len(mask_file) == 1
        assert len(img_file) == 1

        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size
        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            "image": torch.as_tensor(img.copy()).float().contiguous(),
            "mask": torch.as_tensor(mask.copy()).long().contiguous()
        }
    
    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in [".npz", ".npy"]:
            return Image.fromarray(np.load(filename))
        elif ext in [".pt", ".pth"]:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)
    
    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale*w), int(scale * h)
        assert newW > 0 and newH > 0
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST  if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)
        
        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1)) #(w, h, rgb) to (rgb, w, h)
            img_ndarray = img_ndarray / 255 # (0, 255) to (0, 1)
        return img_ndarray

if __name__ == "__main__":
    images_dir = "data/train_hq/"
    masks_dir = "data/train_masks/"
    data = CarvanaDataset(images_dir, masks_dir, scale=1.0)
    
    #sample
    data_dict = data[0]
    plt.imshow(data_dict["image"])
    plt.imshow(data_dict["mask"])

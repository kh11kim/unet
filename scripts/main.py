from unittest import loader
import torch
from unet.data_loading import CarvanaDataset
from torch.utils.data import DataLoader, random_split
import wandb
import logging
#torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    images_dir = "data/train_hq/"
    masks_dir = "data/train_masks/"
    dataset = CarvanaDataset(images_dir, masks_dir, 1.0)
    
    # split train/validation dataset
    val_ratio = 0.1
    batch_size = 1
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, 
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    loader_args = dict(batch_size=1, num_workers=4, pin_memory=True) #??
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(train_set, shuffle=False, drop_last=True, **loader_args)
    
    # (Initialize logging)
    epochs = 5
    learning_rate = 1e-5
    val_percent = val_ratio
    save_checkpoint = True
    img_scale = 1.
    amp = False
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')
    
    # set up the optimizer, loss, lr, loss scaling
    net = Unet()
    optimizer = optim.RMSprop()
    #training
    for epoch in range(epochs):
        pass
    print("aa")
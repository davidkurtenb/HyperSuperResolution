import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import argparse
import sys


class SRDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, scale_factor=2, patch_size=96, augment=True):

        self.root_dir = root_dir
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.augment = augment
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        #extract random patch 
        w, h = image.size
        if w > self.patch_size and h > self.patch_size:
            left = np.random.randint(0, w - self.patch_size)
            top = np.random.randint(0, h - self.patch_size)
            image = image.crop((left, top, left + self.patch_size, top + self.patch_size))
        else:
            image = image.resize((self.patch_size, self.patch_size), Image.BICUBIC)
            
        #Data augmentation
        if self.augment:
            #rotation
            if np.random.rand() > 0.5:
                angle = np.random.choice([90, 180, 270])
                image = image.rotate(angle)
                
            #horizontal flip
            if np.random.rand() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                
            #vertical flip
            if np.random.rand() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        #Create HR image
        hr_image = image
        
        #Create LR image
        lr_size = (hr_image.width // self.scale_factor, hr_image.height // self.scale_factor)
        lr_image = hr_image.resize(lr_size, Image.BICUBIC)
        lr_upscaled = lr_image.resize((hr_image.width, hr_image.height), Image.BICUBIC)
        
        to_tensor = transforms.ToTensor()
        hr_tensor = to_tensor(hr_image)
        lr_tensor = to_tensor(lr_upscaled)
        
        return lr_tensor, hr_tensor

#SRCNN model
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)  
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def save_checkpoint(state, is_best, save_dir, scale_factor):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f'checkpoint_latest_sf{scale_factor}.pth')
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(save_dir, f'model_best_sf{scale_factor}.pth')
        torch.save(state, best_path)
        print(f"Saved best model to {best_path}")
        
    metadata = {
        'epoch': state['epoch'],
        'val_loss': state['val_loss'],
        'val_psnr': state.get('val_psnr', 0),
        'scale_factor': scale_factor,
        'timestamp': str(datetime.datetime.now())
    }
    metadata_path = os.path.join(save_dir, f'training_metadata_sf{scale_factor}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def plot_and_save_metrics(train_losses, val_losses, train_psnrs, val_psnrs, save_path):
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('SRCNN Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_psnrs, 'g-', label='Training PSNR')
    plt.plot(epochs, val_psnrs, 'm-', label='Validation PSNR')  # 'm' for magenta
    plt.title('SRCNN Training and Validation PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics plot saved to {save_path}")

def train_model(model, train_loader, val_loader, num_epochs, device, save_dir, plot_dir, scale_factor, lr=0.0001):
    criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    time_log_path = os.path.join(save_dir, f'training_time_log_sf{scale_factor}.txt')
    with open(time_log_path, 'w') as f:
        f.write("Epoch,Elapsed_Time,Epoch_Duration\n")
    
    training_start_time = datetime.datetime.now()
    epoch_start_time = training_start_time

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    train_psnrs = []
    val_psnrs = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_psnr = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for low_res, high_res in pbar:
                low_res, high_res = low_res.to(device), high_res.to(device)
                
                optimizer.zero_grad()
                output = model(low_res)
                loss = criterion(output, high_res)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                with torch.no_grad():
                    batch_psnr = 0
                    for i in range(output.size(0)):
                        img1 = output[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                        img2 = high_res[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                        batch_psnr += calculate_psnr(img1, img2)
                    batch_psnr /= output.size(0)
                    train_psnr += batch_psnr
                
                train_loss += loss.item()
                pbar.set_postfix({'train_loss': train_loss/len(train_loader), 'psnr': batch_psnr})
        
        epoch_train_loss = train_loss/len(train_loader)
        epoch_train_psnr = train_psnr/len(train_loader)
        train_losses.append(epoch_train_loss)
        train_psnrs.append(epoch_train_psnr)

        model.eval()
        val_loss = 0
        val_psnr = 0
        with torch.no_grad():
            for low_res, high_res in val_loader:
                low_res, high_res = low_res.to(device), high_res.to(device)
                output = model(low_res)
                loss = criterion(output, high_res)
                val_loss += loss.item()
                
                for i in range(output.size(0)):
                    img1 = output[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                    img2 = high_res[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                    val_psnr += calculate_psnr(img1, img2)
        
        epoch_val_loss = val_loss/len(val_loader)
        epoch_val_psnr = val_psnr/len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        val_psnrs.append(epoch_val_psnr)

        scheduler.step(epoch_val_loss)

        current_time = datetime.datetime.now()
        time_since_start = current_time - training_start_time
        epoch_duration = current_time - epoch_start_time

        with open(time_log_path, 'a') as f:
            f.write(f"{epoch+1},{time_since_start},{epoch_duration}\n")

        epoch_start_time = current_time
        
        # Print time information
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        print(f'Train PSNR: {epoch_train_psnr:.2f} dB, Val PSNR: {epoch_val_psnr:.2f} dB')
        print(f'Time elapsed: {time_since_start}, Epoch duration: {epoch_duration}')

        #print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        #print(f'Train PSNR: {epoch_train_psnr:.2f} dB, Val PSNR: {epoch_val_psnr:.2f} dB')
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'train_psnr': epoch_train_psnr,
            'val_psnr': epoch_val_psnr,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_psnrs': train_psnrs,
            'val_psnrs': val_psnrs,
            'scale_factor': scale_factor
        }
        
        is_best = epoch_val_loss < best_val_loss
        save_checkpoint(checkpoint, is_best, save_dir, scale_factor)
        
        if is_best:
            best_val_loss = epoch_val_loss
        
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            plot_and_save_metrics(train_losses, val_losses, train_psnrs, val_psnrs, 
                                 os.path.join(plot_dir, f'metrics_epoch_{epoch+1}_sf{scale_factor}.png'))
    
    total_training_time = datetime.datetime.now() - training_start_time
    with open(time_log_path, 'a') as f:
        f.write(f"Total,{total_training_time},--\n")
    
    print(f"Total training time: {total_training_time}")

    plot_and_save_metrics(train_losses, val_losses, train_psnrs, val_psnrs, 
                         os.path.join(plot_dir, f'metrics_final_sf{scale_factor}.png'))
    
    metrics_data = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_psnrs': train_psnrs,
        'val_psnrs': val_psnrs,
        'epochs': list(range(1, num_epochs + 1))
    }
    
    with open(os.path.join(plot_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics_data, f, indent=4)
    
    return train_losses, val_losses, train_psnrs, val_psnrs

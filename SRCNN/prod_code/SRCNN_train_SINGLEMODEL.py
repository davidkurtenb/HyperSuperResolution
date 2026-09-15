import sys
import os

sys.path.append('/homes/dkurtenb/projects/hypersuperresolution/prod_code')

from SRCNN_utils import *

def main(save_dir, plot_dir, train_dir, val_dir, batch_size=16, epochs=100, 
         patch_size=96, scale_factor=2, lr=0.0001, resume='', use_cuda=True):
    
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    model = SRCNN().to(device)
    
    start_epoch = 0
    if resume:
        if os.path.isfile(resume):
            print(f"Loading checkpoint '{resume}'")
            checkpoint = torch.load(resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Loaded checkpoint '{resume}' (epoch {start_epoch})")
        else:
            print(f"No checkpoint found at '{resume}'")
    
    train_dataset = SRDataset(
        root_dir=train_dir,
        scale_factor=scale_factor,
        patch_size=patch_size,
        augment=True
    )
    
    val_dataset = SRDataset(
        root_dir=val_dir,
        scale_factor=scale_factor,
        patch_size=patch_size,
        augment=False
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=use_cuda
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=use_cuda
    )
       
    train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=epochs, 
        device=device, 
        save_dir=save_dir, 
        plot_dir=plot_dir,
        scale_factor=scale_factor,
        lr=lr
    )


if __name__ == '__main__':
    main(
        save_dir = '/homes/dkurtenb/projects/hypersuperresolution/outputs/checkpoints',
        plot_dir = '/homes/dkurtenb/projects/hypersuperresolution/outputs/plots',
        train_dir = '/homes/dkurtenb/projects/hypersuperresolution/data/beemachine/images/train',
        val_dir = '/homes/dkurtenb/projects/hypersuperresolution/data/beemachine/images/train/val',
        batch_size = 16,
        scale_factor = scale_factor, 
        epochs = 250,
        lr = 0.0001
    )
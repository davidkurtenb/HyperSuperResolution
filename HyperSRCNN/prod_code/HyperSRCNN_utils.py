import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import datetime 

sys.path.append(r'C:\Users\dk412\Desktop\David\Python Projects\HyperSuperResolution\ISR\SRCNN\prod_code')

from SRCNN_utils import *


############################################################################################
#                                Training Utils
############################################################################################

#Setup the shell target network (tnet) that ingests weights from the hnet
class HyperSRCNN(nn.Module):
    def __init__(self, num_models=4):
        super(HyperSRCNN, self).__init__()
        
        template_srcnn = SRCNN()
        self.param_shapes = {name: param.shape for name, param in template_srcnn.named_parameters()}
        
        #conditional embedding for different scale factors (2, 4, 8, 16)
        self.scale_embeddings = nn.Parameter(torch.randn(num_models, 32))
        
        #hnet layers
        self.hidden1 = nn.Linear(32, 128)
        self.hidden2 = nn.Linear(128, 256)
        
        #output layers to tnet
        self.weight_generators = nn.ModuleDict()
        for name, shape in self.param_shapes.items():
            output_size = torch.prod(torch.tensor(shape)).item()
            safe_name = name.replace('.', '_')
            self.weight_generators[safe_name] = nn.Linear(256, output_size)
        
    def forward(self, cond_id):
        embedding = self.scale_embeddings[cond_id]
        
        #forward hnet
        x = torch.relu(self.hidden1(embedding))
        x = torch.relu(self.hidden2(x))
        
        weights = {}
        for name, shape in self.param_shapes.items():
            safe_name = name.replace('.', '_')
            flat_weight = self.weight_generators[safe_name](x)
            weights[name] = flat_weight.reshape(shape)
        
        return weights

#train hnet   
def train_hyper_srcnn(hnet, mnet, train_datasets, val_datasets, scale_factors, results_dir, 
                     batchsize=16, num_epochs=100, device='cuda', lr=0.0001):
    
    epoch_start_time = datetime.datetime.now()
    training_start_time = epoch_start_time
    


    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    
    #time log file
    time_log_path = os.path.join(results_dir, 'epoch_time_log.txt')
    with open(time_log_path, 'w') as f:
        f.write("Epoch,Elapsed_Time\n")   

    optimizer = torch.optim.Adam(hnet.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')

    #training data
    train_dataloaders = {}
    for i, scale in enumerate(scale_factors):
        train_dataloaders[i] = DataLoader(
            train_datasets[scale], 
            batch_size=batchsize, 
            shuffle=True, 
            num_workers=4
        )
    
    #val data
    val_dataloaders = {}
    for i, scale in enumerate(scale_factors):
        if scale in val_datasets:
            val_dataloaders[i] = DataLoader(
                val_datasets[scale],
                batch_size=batchsize,
                shuffle=False,
                num_workers=4
            )

    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for scale_idx in range(len(scale_factors)):
        history[f'scale_{scale_factors[scale_idx]}_train_loss'] = []
        history[f'scale_{scale_factors[scale_idx]}_val_loss'] = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        hnet.train()
        mnet.train()
        
        min_batches = min([len(dataloader) for dataloader in train_dataloaders.values()])
        iterators = {i: iter(dataloader) for i, dataloader in train_dataloaders.items()}
        
        total_loss = 0
        scale_losses = {scale_idx: 0 for scale_idx in range(len(scale_factors))}
        
        for batch in range(min_batches):
            optimizer.zero_grad()
            combined_loss = 0
            
            for cond_id, scale in enumerate(scale_factors):
                lr_imgs, hr_imgs = next(iterators[cond_id])
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                
                #gen wghts
                weights = hnet(cond_id)
                
                #forward pass
                outputs = mnet(lr_imgs, weights=weights)
                
                # Compute loss
                loss = criterion(outputs, hr_imgs)
                scale_losses[cond_id] += loss.item()
                combined_loss += loss
                
                if batch % 100 == 0:
                    with torch.no_grad():
                        psnr = 0
                        for i in range(outputs.size(0)):
                            img1 = outputs[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                            img2 = hr_imgs[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                            psnr += calculate_psnr(img1, img2)
                        psnr /= outputs.size(0)
                    print(f'Scale {scale}x - Batch {batch+1}/{min_batches} - Loss: {loss.item():.4f}, PSNR: {psnr:.2f} dB')
            
            #backprop
            combined_loss.backward()
            optimizer.step()
            
            total_loss += combined_loss.item()
            
            if batch % 100 == 0:
                print(f'Batch {batch+1}/{min_batches} - Combined Loss: {combined_loss.item():.4f}')
        
        #avg training loss
        avg_loss = total_loss / min_batches
        history['train_loss'].append(avg_loss)
        
        #avg loss by scale
        for scale_idx, scale in enumerate(scale_factors):
            avg_scale_loss = scale_losses[scale_idx] / min_batches
            history[f'scale_{scale}_train_loss'].append(avg_scale_loss)
        
        print(f'Epoch {epoch+1} - Average Training Loss: {avg_loss:.4f}')

        has_val_data = False
        avg_val_loss = float('inf')
        
        if val_dataloaders:
            hnet.eval()
            mnet.eval()
            
            val_total_loss = 0
            val_scale_losses = {scale_idx: 0 for scale_idx in range(len(scale_factors))}
            val_batches = 0
            
            with torch.no_grad():
                for cond_id, scale in enumerate(scale_factors):
                    if cond_id in val_dataloaders:
                        scale_val_loss = 0
                        scale_val_batches = 0
                        
                        for lr_imgs, hr_imgs in val_dataloaders[cond_id]:
                            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                            
                            #gen wghts
                            weights = hnet(cond_id)
                            
                            #forward pass
                            outputs = mnet(lr_imgs, weights=weights)
                            
                            #loss
                            loss = criterion(outputs, hr_imgs)
                            scale_val_loss += loss.item()
                            val_scale_losses[cond_id] += loss.item()
                            
                            scale_val_batches += 1
                        
                        avg_scale_val_loss = scale_val_loss / scale_val_batches
                        history[f'scale_{scale}_val_loss'].append(avg_scale_val_loss)
                        print(f'Scale {scale}x - Validation Loss: {avg_scale_val_loss:.4f}')
                        
                        val_total_loss += scale_val_loss
                        val_batches += scale_val_batches
            
            if val_batches > 0:
                has_val_data = True
                avg_val_loss = val_total_loss / val_batches
                history['val_loss'].append(avg_val_loss)
                print(f'Epoch {epoch+1} - Average Validation Loss: {avg_val_loss:.4f}')
        
        if has_val_data and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'hnet_state_dict': hnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'history': history,
            }, os.path.join(results_dir, 'checkpoints', 'hypernetwork_checkpoint_best.pth'))
            print(f"Saved new best model with validation loss: {avg_val_loss:.4f}")
        
        if epoch == num_epochs - 1:
            torch.save({
                'epoch': epoch,
                'hnet_state_dict': hnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss if not has_val_data else avg_val_loss,
                'history': history,
            }, os.path.join(results_dir, 'checkpoints', 'hypernetwork_checkpoint_last.pth'))
            print("Saved final model checkpoint")
        
        #log time
        if (epoch + 1) % 1 == 0 or epoch == num_epochs - 1:
            current_time = datetime.datetime.now()
            time_since_start = current_time - training_start_time
            time_since_last_check = current_time - epoch_start_time
            
            print(f"\n{'='*50}")
            print(f"Time after {epoch+1} epochs: {time_since_start}")
            print(f"Time for last {(epoch+1) % 10 if epoch == num_epochs - 1 else 10} epochs: {time_since_last_check}")
            print(f"{'='*50}\n")
            
            with open(time_log_path, 'a') as f:
                f.write(f"{epoch+1},{time_since_start}\n")
            
            #training_end_time = datetime.datetime.now()
            #total_training_time = training_end_time - training_start_time

            #with open(time_log_path, 'a') as f:
            #    f.write(f"Total,{total_training_time}\n")
                
            epoch_start_time = current_time
    
    plot_training_progress(history, scale_factors, os.path.join(results_dir, 'plots'), num_epochs-1, final=True)
    
    return history

def plot_training_progress(history, scale_factors, plot_dir, epoch, final=False):
    os.makedirs(plot_dir, exist_ok=True)
    
    #overall loss
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    
    plt.title('Hypernetwork Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'overall_loss_final.png'), dpi=300)
    plt.close()
    
 
    #loss by scale
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, scale in enumerate(scale_factors):
        train_key = f'scale_{scale}_train_loss'
        val_key = f'scale_{scale}_val_loss'
        
        ax = axes[i]
        
        if train_key in history:
            ax.plot(epochs, history[train_key], 'b-', label='Train')
        
        if val_key in history and history[val_key]:
            ax.plot(epochs, history[val_key], 'r-', label='Validation')
        
        ax.set_title(f'Scale {scale}x Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss (MSE)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'detailed_loss_per_scale.png'), dpi=300)
    plt.close()

############################################################################################
#                                       Inference Utils
############################################################################################

#SRCNN model with no_weights flag
class SRCNNNoWeights(SRCNN):
    def forward(self, x, weights=None):
        if weights is None:
            return super().forward(x)
        
        #apply weights generated from the hnet
        x = F.relu(F.conv2d(x, weights['conv1.weight'], weights['conv1.bias'], padding=4))
        x = F.relu(F.conv2d(x, weights['conv2.weight'], weights['conv2.bias'], padding=2))
        x = F.conv2d(x, weights['conv3.weight'], weights['conv3.bias'], padding=2)
        return x

def calculate_ssim(img1, img2):
    return ssim(img1, img2, channel_axis=2, data_range=255)

def comprehensive_inference(hyper_checkpoint_path, srcnn_model_paths, image_path, scale_factor_index, output_dir='results'):

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #load hnet
    mnet = SRCNNNoWeights().to(device)
    hnet = HyperSRCNN(num_models=4).to(device)
    
    hyper_checkpoint = torch.load(hyper_checkpoint_path, map_location=device)
    hnet.load_state_dict(hyper_checkpoint['hnet_state_dict'])
    
    hnet.eval()
    mnet.eval()
    
    #map sf to index
    scale_factors = [2, 4, 8, 16]
    scale_factor = scale_factors[scale_factor_index]
    print(f"Using scale factor: {scale_factor}x")
    
    #load SRCNN model for the speicifc sf 
    srcnn_model = SRCNN().to(device)
    srcnn_model_path = srcnn_model_paths.get(scale_factor)
    
    if srcnn_model_path and os.path.exists(srcnn_model_path):
        srcnn_checkpoint = torch.load(srcnn_model_path, map_location=device)
        srcnn_model.load_state_dict(srcnn_checkpoint['model_state_dict'])
        srcnn_model.eval()
        #print(f"Loaded SRCNN model for {scale_factor}x from {srcnn_model_path}")
    else:
        srcnn_model = None
        #print(f"No SRCNN model found for {scale_factor}x")
    
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    
    low_res = img.resize((w//scale_factor, h//scale_factor), Image.BICUBIC)
    bicubic_upscaled = low_res.resize((w, h), Image.BICUBIC)
    
    to_tensor = transforms.ToTensor()
    input_tensor = to_tensor(bicubic_upscaled).unsqueeze(0).to(device)
    
    #HyperSRCNN inf
    with torch.no_grad():
        #HyperSRCNN
        weights = hnet(scale_factor_index)
        hyper_output = mnet(input_tensor, weights=weights)
        
        #SRCNN
        if srcnn_model is not None:
            srcnn_output = srcnn_model(input_tensor)
    
    hyper_output_img = hyper_output.squeeze().cpu().clamp(0, 1)
    hyper_output_img = transforms.ToPILImage()(hyper_output_img)
    
    if srcnn_model is not None:
        srcnn_output_img = srcnn_output.squeeze().cpu().clamp(0, 1)
        srcnn_output_img = transforms.ToPILImage()(srcnn_output_img)
    
    #PSNR
    img_np = np.array(img)
    bicubic_np = np.array(bicubic_upscaled)
    hyper_sr_np = np.array(hyper_output_img)
    
    bicubic_psnr = calculate_psnr(img_np, bicubic_np)
    hyper_sr_psnr = calculate_psnr(img_np, hyper_sr_np)
        
    if srcnn_model is not None:
        srcnn_np = np.array(srcnn_output_img)
        srcnn_psnr = calculate_psnr(img_np, srcnn_np)
        print("SRCNN PSNR", srcnn_psnr)

    #ssim
    bicubic_ssim = calculate_ssim(img_np, bicubic_np)
    hyper_sr_ssim = calculate_ssim(img_np, hyper_sr_np)

    if srcnn_model is not None:
        srcnn_ssim = calculate_ssim(img_np, srcnn_np)
        print("SRCNN SSIM", srcnn_ssim)

    filename = os.path.splitext(os.path.basename(image_path))[0]

    #comp visual
    num_cols = 5
    fig, axes = plt.subplots(1, num_cols, figsize=(num_cols*4, 8))
    axes = axes.flat
    
    axes[0].imshow(np.array(img))
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    axes[1].imshow(np.array(low_res))
    axes[1].set_title(f"Low-res ({w//scale_factor}x{h//scale_factor})")
    axes[1].axis("off")
    
    axes[2].imshow(np.array(bicubic_upscaled))
    axes[2].set_title(f"Bicubic Upscaling\nPSNR: {bicubic_psnr:.2f} dB")
    axes[2].axis("off")

    if srcnn_model is not None:
        axes[3].imshow(np.array(srcnn_output_img))
        axes[3].set_title(f"SRCNN ({scale_factor}x)\nPSNR: {srcnn_psnr:.2f} dB")
        axes[3].axis("off")
    
    axes[4].imshow(np.array(hyper_output_img))
    axes[4].set_title(f"Hyper-SRCNN ({scale_factor}x)\nPSNR: {hyper_sr_psnr:.2f} dB")
    axes[4].axis("off")
    
    comparison_path = os.path.join(output_dir, f"{filename}_full_comparison_{scale_factor}x.png")
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=300)
    plt.close()
    
    #zoomed comp
    zoomed_fig = plt.figure(figsize=(num_cols*6, 6))
    
    crop_w, crop_h = w//4, h//4
    crop_x, crop_y = w//3, h//3 
    
    original_crop = np.array(img)[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    bicubic_crop = np.array(bicubic_upscaled)[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    hyper_sr_crop = np.array(hyper_output_img)[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    
    num_zoomed = 3
    if srcnn_model is not None:
        srcnn_crop = np.array(srcnn_output_img)[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        num_zoomed = 4
    
    for i in range(num_zoomed):
        ax = zoomed_fig.add_subplot(1, num_zoomed, i+1)
        
        if i == 0:
            ax.imshow(original_crop)
            ax.set_title("Original (zoomed)")
        elif i == 1:
            ax.imshow(bicubic_crop)
            ax.set_title(f"Bicubic (zoomed)\nPSNR: {bicubic_psnr:.2f} dB")
        elif i == 3:
            ax.imshow(hyper_sr_crop)
            ax.set_title(f"Hyper-SRCNN (zoomed)\nPSNR: {hyper_sr_psnr:.2f} dB")
        elif i == 2 and srcnn_model is not None:
            ax.imshow(srcnn_crop)
            ax.set_title(f"SRCNN (zoomed)\nPSNR: {srcnn_psnr:.2f} dB")
        
        ax.axis("off")
    
    zoomed_comparison_path = os.path.join(output_dir, f"{filename}_zoomed_comparison_{scale_factor}x.png")
    plt.tight_layout()
    plt.savefig(zoomed_comparison_path, dpi=300)
    plt.close()
    
    results = {
        "comparison": comparison_path,
        "zoomed_comparison": zoomed_comparison_path,
        "bicubic_psnr": bicubic_psnr,
        "bicubic_ssim": bicubic_ssim,
        "srcnn_psnr": srcnn_psnr,
        "srcnn_ssim":srcnn_ssim,
        "hyper_sr_psnr": hyper_sr_psnr,
        "hyper_sr_ssim": hyper_sr_ssim
    }
        
    return results


def batch_comprehensive_inference(hyper_checkpoint_path, srcnn_model_paths, image_dir, scale_factor_index, output_dir='results'):
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    all_results = []
    scale_factors = [2, 4, 8, 16]
    scale_factor = scale_factors[scale_factor_index]
    
    for image_file in tqdm(image_files, desc=f"Processing images at {scale_factor}x scale"):
        image_path = os.path.join(image_dir, image_file)
        print(f"Processing: {image_path}")
        
        result = comprehensive_inference(
            hyper_checkpoint_path=hyper_checkpoint_path,
            srcnn_model_paths=srcnn_model_paths,
            image_path=image_path,
            scale_factor_index=scale_factor_index,
            output_dir=output_dir
        )
        
        result["image_name"] = image_file
        all_results.append(result)
    
    avg_bicubic_psnr = sum(r["bicubic_psnr"] for r in all_results) / len(all_results)
    avg_hyper_sr_psnr = sum(r["hyper_sr_psnr"] for r in all_results) / len(all_results)
    
    has_srcnn = "srcnn_psnr" in all_results[0]
    if has_srcnn:
        avg_srcnn_psnr = sum(r["srcnn_psnr"] for r in all_results) / len(all_results)

    avg_bicubic_ssim = sum(r["bicubic_ssim"] for r in all_results) / len(all_results)
    avg_hyper_sr_ssim = sum(r["hyper_sr_ssim"] for r in all_results) / len(all_results)
    avg_srcnn_ssim = sum(r["srcnn_ssim"] for r in all_results) / len(all_results)


    """
    #summary
    print("\n===== BATCH INFERENCE RESULTS =====")
    print(f"Number of test images: {len(image_files)}")
    print(f"Scale factor: {scale_factor}x")
    print(f"Average Bicubic PSNR: {avg_bicubic_psnr:.2f} dB")
    print(f"Average Hyper-SRCNN PSNR: {avg_hyper_sr_psnr:.2f} dB")
    print(f"Average Hyper-SRCNN improvement over Bicubic: {avg_hyper_improvement:.2f} dB")
    
    if has_srcnn:
        print(f"Average SRCNN PSNR: {avg_srcnn_psnr:.2f} dB")
        print(f"Average SRCNN improvement over Bicubic: {avg_srcnn_improvement:.2f} dB")
        print(f"Average Hyper-SRCNN vs SRCNN difference: {avg_hyper_vs_srcnn:.2f} dB")
    """
    #summary
    results_df = pd.DataFrame([{
        'image_name': r['image_name'],
        'bicubic_psnr': r['bicubic_psnr'],
        'bicubic_ssim': r['bicubic_ssim'],
        'hyper_srcnn_psnr': r['hyper_sr_psnr'],
        'hyper_srcnn_ssim': r['hyper_sr_ssim'],
        **(
            {
                'srcnn_psnr': r['srcnn_psnr'],
                'srcnn_ssim': r['srcnn_ssim'],
            } if has_srcnn else {}
        )
    } for r in all_results])

    summary_df = pd.DataFrame({
        'metric': [
            'Scale Factor',
            'Number of Test Images',
            'Average Bicubic PSNR (dB)',
            'Average Bicubic SSIM',
            'Average Hyper-SRCNN PSNR (dB)',
            'Average Hyper-SRCNN SSIM',
            *([
                'Average SRCNN PSNR (dB)',
                'Average SRCNN SSIM',
            ] if has_srcnn else [])
        ],
        'value': [
            f'{scale_factor}x',
            len(image_files),
            f'{avg_bicubic_psnr:.4f}',
            f'{avg_bicubic_ssim:.4f}',
            f'{avg_hyper_sr_psnr:.4f}',
            f'{avg_hyper_sr_ssim:.4f}',
            *([
                f'{avg_srcnn_psnr:.4f}',
                f'{avg_srcnn_ssim:.4f}',
            ] if has_srcnn else [])
        ]
    })
    
    excel_path = os.path.join(output_dir, f"comparison_results_{scale_factor}x.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        results_df.to_excel(writer, sheet_name='Detailed Results', index=False)
    
    print(f"Results saved to: {excel_path}")
    
    #summary boxplot
    plt.figure(figsize=(10, 6))
    
    data_to_plot = [
        results_df['bicubic_psnr'].values,
        results_df['srcnn_psnr'].values,
        results_df['hyper_srcnn_psnr'].values
    ]
    
    labels = ['Bicubic', 'SRCNN','Hyper-SRCNN']
    
    plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                boxprops=dict(facecolor='lightblue'))
    plt.title(f'PSNR Distribution at {scale_factor}x Scale')
    plt.ylabel('PSNR (dB)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    boxplot_path = os.path.join(output_dir, f"psnr_boxplot_{scale_factor}x.png")
    plt.tight_layout()
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    
    return all_results, results_df
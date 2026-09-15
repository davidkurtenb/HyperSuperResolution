import sys
import os
import datetime

sys.path.append('/homes/dkurtenb/projects/hypersuperresolution/prod_code')

from SRCNN_train_SINGLEMODEL import main

def train_multiple_scales():
    #scale_factor_lst = [2, 4, 8, 16]
    scale_factor_lst = [32, 64]
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = '/homes/dkurtenb/projects/hypersuperresolution/SRCNN/outputs'

    #################    DATA SELECT ##################################

    # BeeMachine   ####################################################### 
    train_dir='/homes/dkurtenb/projects/hypersuperresolution/data/beemachine/images/train'
    val_dir='/homes/dkurtenb/projects/hypersuperresolution/data/beemachine/images/val'

    # DF2K   #######################################################
    #train_dir='/homes/dkurtenb/projects/hypersuperresolution/data/DF2K/DF2K_train_HR'
    #val_dir='/homes/dkurtenb/projects/hypersuperresolution/data/DF2K/DF2K_valid_HR'
            
    epochs = 250
    compute = '4c2gpu'

    if 'bee' in train_dir:
        save_dir_prefix = os.path.join(base_save_dir,f'training_run_beemachine_{epochs}epochs_{compute}_{current_time}')
        model_dir = os.path.join(save_dir_prefix,'checkpoints')
        plot_save_dir = os.path.join(save_dir_prefix,'plots')
    elif 'DF2K' in train_dir:
        save_dir_prefix = os.path.join(base_save_dir,f'training_run_df2k_{epochs}epochs_{compute}_{current_time}')
        model_dir = os.path.join(save_dir_prefix,'checkpoints')
        plot_save_dir = os.path.join(save_dir_prefix,'plots')
    else:
        save_dir_prefix = os.path.join(base_save_dir,f'training_run_{epochs}epochs_{compute}_{current_time}')
        model_dir = os.path.join(save_dir_prefix,'checkpoints')
        plot_save_dir = os.path.join(save_dir_prefix,'plots')

    scale_times = {}

    for scale_factor in scale_factor_lst:
        print(f"\n{'='*50}")
        print(f"Training model with scale factor: {scale_factor}")
        print(f"{'='*50}\n")
        
        # Create scale-specific directories
        save_dir = os.path.join(model_dir, f'sf{scale_factor}')
        plot_dir = os.path.join(plot_save_dir, f'sf{scale_factor}')
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        
        scale_start_time = datetime.datetime.now()

        main(
            save_dir=save_dir,
            plot_dir=plot_dir,
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=16,
            epochs=epochs,
            scale_factor=scale_factor,
            lr=0.0001
        )

        scale_end_time = datetime.datetime.now()
        scale_time = scale_end_time - scale_start_time
        scale_times[scale_factor] = scale_time
        print(f"Training time for scale factor {scale_factor}x: {scale_time}")
    
    # Save the training times to a file
    with open(os.path.join(save_dir_prefix, 'training_times.txt'), 'w') as f:
        f.write("SRCNN Training Times:\n")
        for scale_factor, time in scale_times.items():
            f.write(f"Scale factor {scale_factor}x: {time}\n")
    
    return scale_times, current_time, save_dir_prefix

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    scale_times,current_time,save_dir_prefix = train_multiple_scales()
    end_time = datetime.datetime.now()
    total_training_time = end_time - start_time
    
    print(f"\n{'='*50}")
    print(f"SRCNN Training Summary:")
    print(f"{'='*50}")
    for scale_factor, time in scale_times.items():
        print(f"Scale factor {scale_factor}x: {time}")
    print(f"Total Training Time for all SRCNN models: {total_training_time}")
    
    # Save the total time to the file
    with open(os.path.join(save_dir_prefix, 'training_times.txt'), 'a') as f:
        f.write(f"Total Training Time for all SRCNN models: {total_training_time}\n")
import os
import argparse
import zarr
from PIL import Image
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--root_dir',type=str) 
parser.add_argument('--save_dir',type=str, default='../data/npy')
parser.add_argument('--file_type', type=str, default='zarr') 
parser.add_argument('--slide_name',type=str)

args = parser.parse_args()
#root_dir = '../data/SQ1631_img_cropped'
#save_dir = '../data/npy'
#slide_name = 'SQ1631'
root_dir = args.root_dir
save_dir = args.save_dir
file_type = args.file_type
slide_name = args.slide_name

fn_ls = [img for img in os.listdir(root_dir) if img.endswith('.png')]
fn_ls.sort()
for i in range(4):
    arr = []
    channel_names = []
    for fn in fn_ls:
        parts = fn.split(' ')
        s = parts[8].split('_')[-2]
        if s.endswith(str(i+1)):
            r = parts[8].split('_')[-1][0]
            channel = np.asarray(Image.open(os.path.join(root_dir, fn)).convert('L'))    # read image as single channel and convert to np array
            arr.append(channel)
            name = '-'.join(parts[:8] + [parts[8].split('_')[0]])
            channel_names.append(name)
    arr = np.array(arr)
    print(f'Shape of slide {i+1} in {root_dir}: {arr.shape}.')
    df = pd.DataFrame(channel_names, columns=['channel_name'])
    if file_type == 'npy':
        out = os.path.join(save_dir, f'{slide_name}_s{i+1}_{r}.npy')
        np.save(out, arr)
        print(f'Data saved to {out}')
        channel_fn = os.path.join(save_dir, f'{slide_name}_s{i+1}_{r}_channel_names.csv')
        df.to_csv(channel_fn, header=False)
        print(f'Channel name list saved to {channel_fn}')

    elif file_type == 'zarr':
        out = os.path.join(save_dir, f'{slide_name}_s{i+1}_{r}')
        os.makedirs(out, exist_ok=True)
        zarr.save(f'{out}/slide.zarr', arr)
        print(f'Data saved to {out}/slide.zarr')
        channel_fn = os.path.join(out, f'{slide_name}_s{i+1}_{r}_channel_names.csv')
        df.to_csv(channel_fn, header=False)
        print(f'Channel name list saved to {channel_fn}')

    else:
        print(f'wrong file type: {file_type}!')
    
    
    
    
    
    
    

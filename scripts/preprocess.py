import os
import argparse
import zarr
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument('--root_dir',type=str) 
parser.add_argument('--save_dir',type=str, default='../data/npy')
parser.add_argument('--file_type', type=str, default='zarr')
parser.add_argument('--sample_size', type=int, default=None)

args = parser.parse_args()
root_dir = args.root_dir
save_dir = args.save_dir
file_type = args.file_type

fn_ls = [str(img) for img in Path(root_dir).rglob('*cropped/*.png')]
print(len(fn_ls))
fn_ls.sort()
for i in range(4):
    arr = []
    channel_names = []
    for fn in fn_ls:
        slide_name = fn.split('/')[-2].split('_')[0]
        s = fn.split('_')[-2]
        if s.endswith(str(i+1)):
            r = fn.split('_')[-1][0]
            channel = np.asarray(Image.open(fn).convert('L'))    # read image as single channel and convert to np array
            arr.append(channel)
            file_name = fn.split('/')[-1].split('.png')[0][:-5]  # record file name
            file_name = "-".join(file_name.split(' '))
            channel_names.append(file_name)
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
    
    
    
    
    
    
    

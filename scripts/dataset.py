import os
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
from PIL import Image
from skimage.io import imsave, imread
from skimage.transform import rescale, resize


class ZarrDataset(data.Dataset):
    '''
    This class definition creates handles for each zarr file as a dataset, with tiles as instances.
    The tile locations are created and save as a numpy file before data loading.
    '''
    def __init__(self, root_path = None, img_id = None, tile_size = None, overlap = None, threshold = None, mask_id = 'default', transform = None, label = None):
        self.root_path = root_path
        self.img_id = img_id
        self.tile_size = tile_size
        self.overlap = overlap
        self.threshold = threshold
        self.mask_id = mask_id
        self.transform = transform
        self.slide = self.read_slide()
        self.label = label
        if tile_size is not None and mask_id is not None:
            # Load tiles positions from disk
            self.tile_pos = self.load_tiles(mask_id, tile_size, overlap, threshold)
        
    def read_slide(self):
        ''' Read numpy file on disk mapped to memory '''
        import zarr
        zarr_path = f'{self.root_path}/{self.img_id}/slide.zarr'
        slide = zarr.open(zarr_path, mode = 'r')
        #print(f'Zarr dataset loaded from {zarr_path}')
        return slide

    def read_region(self, pos_x, pos_y, width, height):
        ''' Read a zarr slide region
            the width and height always refer to the original image
            spacing controls the final output size
        '''
        region = self.slide[:, pos_y:pos_y+height, pos_x:pos_x+width].copy()
        #region = region_np.swapaxes(0, 1) # Change to numpy format
        return region
    
    def get_slide_dimensions(self):
        ''' Get slide dimensions '''
        return self.slide.shape[2], self.slide.shape[1] # (x, y), (column, row) of array
    
    def __getitem__(self, index):
        image = self.read_region(self.tile_pos[index][0], self.tile_pos[index][1], self.tile_size, self.tile_size)
        image = torch.tensor(image, dtype = torch.float)    
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(self.label, dtype=torch.float),  self.img_id
    
    def __len__(self):
        return len(self.tile_pos)
    
    def generate_mask(self, save_path=None, threshold=0):
        slide_array = np.array(self.slide)
        mask = np.sum(slide_array, axis=0) > threshold
        mask_img = Image.fromarray(mask)
        if save_path is None:
            save_path = f'{self.root_path}/{self.img_id}/masks'
        os.makedirs(save_path, exist_ok=True)
        mask_img.save(f'{save_path}/threshold-{threshold}.png')
    
    def load_tiling_mask(self, mask_path):
        ''' Load tissue mask to generate tiles '''
        # Get slide dimensions
        slide_width, slide_height = self.get_slide_dimensions()
        # Create mask
        if mask_path is not None: # Load mask from existing file
            mask_temp = np.array(imread(mask_path))    #.swapaxes(0, 1)
            print(mask_temp.shape)
            print(f'slide width: {slide_width}; slide height: {slide_height}')
            assert abs(mask_temp.shape[1] / mask_temp.shape[0] - slide_width / slide_height) < 0.01 , 'Mask shape does not match slide shape'
        else:
            mask_temp = np.ones((slide_height, slide_width)) # Tile all regions
        return mask_temp

    def generate_tiles(self, mask_path, mask_id, tile_size, overlap, threshold = 0.99):
        ''' 
        Generate tiles from a slide
        threshold: minimum percentage of tissue mask in a tile
        '''
        # Load mask
        mask = self.load_tiling_mask(mask_path)
        h, w = mask.shape
        step_size = tile_size - overlap
        # Generate tile coordinates according to masked grid
        nx = w // step_size + 1
        ny = h // step_size + 1
        mask_grid = np.zeros((ny, nx))
        for i in range(nx):
            for j in range(ny):
                x = i * step_size
                y = j * step_size
                x_end = x + tile_size
                y_end = y + tile_size
                if x_end <= w and y_end <= h and (np.mean(mask[y :y_end, x : x_end]) >= threshold):
                    mask_grid[j, i] = 1
        hs, ws = np.where(mask_grid == 1)
        
        positions = (np.array(list(zip(ws, hs))) * step_size)
        # Save tile top left positions
        tile_path = f'{self.root_path}/{self.img_id}/tiles/{mask_id}'
        save_path = f'{tile_path}/{tile_size}-{overlap}-{threshold}'
        os.makedirs(save_path, exist_ok=True)
        np.save(f'{save_path}/tile_positions_top_left.npy', positions)
        # Save mask image
        imsave(f'{save_path}/mask.png', (mask_grid * 255).astype(np.uint8))
    
    def load_tiles(self, mask_id, tile_size, overlap, threshold):
            ''' load tiles positions from disk '''
            tile_path = f'{self.root_path}/{self.img_id}/tiles/{mask_id}/{tile_size}-{overlap}-{threshold}'
            tile_pos = np.load(f'{tile_path}/tile_positions_top_left.npy') #, mmap_mode = 'r', allow_pickle = True)
            return tile_pos
    

class ZarrSlidesDataset(data.Dataset):
    '''
    This class creates one dataset from multiple datasets of zarr images
    '''
    def __init__(self, slides_root_path = None, img_ids = None, tile_size = None, overlap = None, threshold = None, mask_id = 'default', transform = None, labels = None, dataset_class = None):
        self.slides_root_path = slides_root_path
        self.tile_size = tile_size
        self.overlap = overlap
        self.threshold = threshold
        self.mask_id = mask_id
        self.transform = transform
        self.labels = labels

        # Get id and path for all slides
        if img_ids is None:
            img_ids = self.get_slide_from_paths(slides_root_path)
        self.slides_dict, self.lengths = self.get_slides(img_ids, labels, dataset_class)

    def __getitem__(self, index):
        for slide_idx, (img_id, slide) in enumerate(self.slides_dict.items()):
            if index < self.lengths[slide_idx]:
                return slide[index]
            else:
                index -= self.lengths[slide_idx]

    def __len__(self):
        return sum(self.lengths)

    def get_slide_from_paths(self, slides_root_path):
        ''' Get slides from a directory '''
        slide_ids = []
        for slide_id in os.listdir(slides_root_path):
            if os.path.isdir(os.path.join(slides_root_path, slide_id)) and not slide_id.startswith('.'):
                slide_ids.append(slide_id)
        return slide_ids

    def get_slides(self, img_ids, labels, dataset_class):
        from tqdm import tqdm
        img_dict = {}
        lengths = []
        print('Loading slides...')
        for img_id, img_label in tqdm(zip(img_ids, labels)):
            #print(f'Loading {slide_id}')
            slide = dataset_class(self.slides_root_path, img_id, self.tile_size, self.overlap, self.threshold, self.mask_id, self.transform, img_label)
            img_dict[img_id] = slide
            lengths.append(len(slide))
        return img_dict, lengths
    
    def get_tile_idx(self, subset=None):
        """
        return a combined array of all tile_pos for each slide.
        """
        slides_idx = []
        for slide_id, slide in self.slides_dict.items():
            tile_pos = slide.tile_pos
            idx = {'img_id': slide_id}
            
            idx['x'] = tile_pos[:, 0]
            idx['y'] = tile_pos[:, 1]
            idx = pd.DataFrame(idx)
            slides_idx.append(idx)
        slides_idx = pd.concat(slides_idx)
        if subset is not None:
            slides_idx = slides_idx.iloc[:150]

        return slides_idx
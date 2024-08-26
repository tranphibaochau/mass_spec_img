import dataset
import os
import torch
import torchmetrics
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


def main():
    '''
    Model training with tensor in RAM
    '''
    # load data from npy
    # generate tiles and create trn/tst split on sample level
    # dataset and dataloader
    # instantiate model and train

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = 10 
    clean_up = True

    all_imgs = ['SQ1631_s1_R', 'SQ1631_s2_R', 'SQ1631_s3_N', 'SQ1631_s4_N',
                'SQ1632_s1_R', 'SQ1632_s2_N', 'SQ1632_s3_R', 'SQ1632_s4_N',
                'SQ1633_s1_R', 'SQ1633_s2_R', 'SQ1633_s3_N', 'SQ1633_s4_N',
                'SQ1634_s1_N', 'SQ1634_s2_R', 'SQ1634_s3_R', 'SQ1634_s4_N',
                'SQ1635_s1_N', 'SQ1635_s2_R', 'SQ1635_s3_R', 'SQ1635_s4_N',
                'SQ1636_s1_R', 'SQ1636_s2_N', 'SQ1636_s3_N', 'SQ1636_s4_R']
    all_imgs = np.array(all_imgs)

    label_ls = [1 if sample_name[-1] == 'R' else 0 for sample_name in all_imgs]
    label_ls = np.array(label_ls)

    # generate tiles 
    data_root_path = '../data/zarr'
    mask_threshold = 0
    area_threshold = 0.2
    tile_size = 32
    overlap = 8
    mask_id = f'threshold-{mask_threshold}'
    for img_id in all_imgs:
        img_ds = dataset.ZarrDataset(root_path='../data/zarr', img_id=img_id)
        mask_path = f'{data_root_path}/{img_id}/masks/{mask_id}.png'
        tile_path = f'{data_root_path}/{img_id}/tiles/{mask_id}/{tile_size}-{overlap}-{area_threshold}/tile_positions_top_left.npy'
        if clean_up:
            os.remove(mask_path)
            os.remove(tile_path)
        if not os.path.isfile(mask_path):
            img_ds.generate_mask()
            print(f'Mask {mask_id} generated for {img_id}.')
        if not os.path.isfile(tile_path):
            img_ds.generate_tiles(mask_path=mask_path, mask_id=mask_id, 
                                  tile_size=tile_size, overlap=overlap, threshold=area_threshold)
            print(f'Tiles from {mask_id} generated for {img_id}, tile_size: {tile_size}; overlap: {overlap}; area_threshold: {area_threshold}.')

    def collate_fn(batch):
        inputs, labels, img_ids = zip(*batch)
        inputs = torch.stack(inputs, 0)
        labels = torch.stack(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)
        return inputs, labels
        
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    
    kfold_accuracy = []
    kfold_auc = []
    
    # data augmentation transforms for training
    trn_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45)
            ])

    for i, (trn_idx, tst_idx) in enumerate(skf.split(X=np.zeros(len(label_ls)), y=label_ls)):
        trn_idx = np.array(trn_idx, dtype=int)
        tst_idx = np.array(tst_idx, dtype=int)
        print(all_imgs[trn_idx])
        print(f'Fold {i}')
        print('--------------------------------')
        
        trn_dataset = dataset.ZarrSlidesDataset(slides_root_path=data_root_path, img_ids = all_imgs[trn_idx], 
                                                tile_size = tile_size, overlap = overlap, threshold = area_threshold, mask_id = mask_id, 
                                                transform = trn_transform, labels = label_ls[trn_idx], dataset_class = dataset.ZarrDataset)
        tst_dataset = dataset.ZarrSlidesDataset(slides_root_path=data_root_path, img_ids = all_imgs[tst_idx], 
                                                tile_size = tile_size, overlap = overlap, threshold = area_threshold, mask_id = mask_id, 
                                                transform = None, labels = label_ls[tst_idx], dataset_class = dataset.ZarrDataset)

        # Define data loaders for training and testing data in this fold
        trn_loader = DataLoader(trn_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        tst_loader = DataLoader(tst_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
        
        # Initialize the model for this fold
        model = SimpleCNN(in_channels=917).to(device)
        auc, accuracy = train_and_evaluate(n_epochs, model, trn_loader, tst_loader)
        kfold_accuracy.append(accuracy)
        kfold_auc.append(auc)
    
    print('K Fold Accuracy:')
    print(kfold_accuracy)
    print('K Fold AUROC:')
    print(kfold_auc)


class SimpleCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, 16, 1),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 8, 3, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(8, 8, 3, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(8, 8, 3, stride=2),
                                   nn.ReLU()
                                  )
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        x = self.conv(x)
        #print(x.shape)
        x = torch.mean(x, dim = (2, 3), keepdim = False) # Global pooling
        #print(x.shape)
        out = self.fc(x)
        out = torch.squeeze(out)
        if out.ndim == 0:
            out = out.unsqueeze(0)
        return out

   
def train_and_evaluate(n_epochs, model, trn_loader, tst_loader):
    criterion = nn.BCEWithLogitsLoss()
    accuracy_fun = torchmetrics.Accuracy(task='binary')
    roc_fun = torchmetrics.AUROC(task='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(n_epochs):
        print('EPOCH {}:'.format(epoch + 1))
        model.train()
        running_loss = 0.
        for i, (inputs, labels) in enumerate(trn_loader):
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 9:  # Print every 100 batches
                print(f'Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{len(trn_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
        print(f'Epoch [{epoch + 1}/{n_epochs}] final loss: {loss}')
    print('Finished Training')

    # test model and report metric
    model.eval()
    running_loss = 0.
    all_probs = []
    all_labels = []
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tst_loader):
            logits = model(inputs)
            all_labels.append(labels.cpu())
            all_probs.append(nn.Sigmoid()(logits).cpu())
            loss = criterion(logits, labels)
            running_loss += loss
    all_probs = torch.cat(all_probs)
    #print(all_probs)
    all_labels = torch.cat(all_labels)
    #print(all_labels)
    avg_loss = running_loss / (i + 1)
    accuracy = accuracy_fun(all_probs, all_labels)
    auc = roc_fun(all_probs, all_labels)
    print(f'Test loss: {avg_loss}; AUROC: {auc}; Accuracy: {accuracy}')
    return accuracy, auc

    
if __name__ == '__main__':
    main()
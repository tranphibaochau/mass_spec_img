import torch
import torchmetrics
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, Subset

def main():
    '''
    Model training with tensor in RAM
    '''
    # load data from npy
    # generate tiles and create trn/tst split on sample level
    # dataset and dataloader
    # instantiate model and train

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = 15 

    sample_ls = ['SQ1631_s1_R', 'SQ1631_s2_R', 'SQ1631_s3_N', 'SQ1631_s4_N',
                 'SQ1632_s1_R', 'SQ1632_s2_N', 'SQ1632_s3_R', 'SQ1632_s4_N',
                 'SQ1633_s1_R', 'SQ1633_s2_R', 'SQ1633_s3_N', 'SQ1633_s4_N']

    tile_ls = []
    label_ls = []
    tile_grp = []
    tile_coords = []

    for sample_name in sample_ls:
        slide = np.load(f'../data/npy/{sample_name}.npy', mmap_mode='r')
        tiles, coords = tile_array(slide, tile_size=32, overlap=8, threshold=0.2)
        labels = [1 if sample_name[-1] == 'R' else 0] * len(tiles)
        print(f'{len(tiles)} tiles generated in sample {sample_name}.')
        tile_ls.extend(tiles)
        label_ls.extend(labels)
        tile_grp.extend([sample_name] * len(tiles))
        tile_coords.extend(coords)

    tile_ls = np.asarray(tile_ls)
    label_ls = np.asarray(label_ls)
    print(tile_ls.shape)
    print(label_ls.shape)

    tile_dat = TileDataset(inputs=tile_ls, labels=label_ls)

    def collate_fn(batch):
        inputs, labels = zip(*batch)
        #print(inputs[0].shape)
        inputs = torch.stack(inputs, 0)
        labels = torch.stack(labels)
        #print(inputs.shape)
        #print(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)
        return inputs, labels
        
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    
    kfold_accuracy = []
    kfold_auc = []

    for i, (trn_idx, tst_idx) in enumerate(sgkf.split(X=np.zeros(len(label_ls)), y=label_ls, groups=tile_grp)):
        print(f'Fold {i}')
        print('--------------------------------')
        trn_subset = Subset(tile_dat, trn_idx)
        tst_subset = Subset(tile_dat, tst_idx)

        # Define data loaders for training and testing data in this fold
        trn_loader = DataLoader(trn_subset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        tst_loader = DataLoader(tst_subset, batch_size=4, shuffle=False, collate_fn=collate_fn)
        tst_grp = np.asarray(tile_grp)[tst_idx]
        # Initialize the model for this fold
        model = SimpleCNN(in_channels=917).to(device)
        auc, accuracy = train_and_evaluate(n_epochs, model, trn_loader, tst_loader)
        kfold_accuracy.append(accuracy)
        kfold_auc.append(auc)
    
    print('K Fold Accuracy:')
    print(kfold_accuracy)
    print('K Fold AUROC:')
    print(kfold_auc)

# create tiles
def tile_array(array, tile_size, overlap, threshold):
    c, h, w = array.shape
    step_size = tile_size - overlap
    nx = w // step_size + 1
    ny = h // step_size + 1
    mask = np.sum(array, axis=0) > 0
    tiles = []
    coords = []
    for i in range(nx):
        for j in range(ny):
            
            x = i * step_size
            y = j * step_size
            x_end = min((x + tile_size), w)
            y_end = min((y + tile_size), h)
            if np.mean(mask[y :y_end, x : x_end]) > threshold:
                tile = np.zeros((c, tile_size, tile_size))
                tile[:,: y_end-y, :x_end-x] = array[:, y :y_end, x : x_end]
                tiles.append(tile)
                coords.append([x, y])
            else:
                pass
    return tiles, coords

class TileDataset(Dataset):
    '''
    Dataset class for numpy arrays in memory
    '''
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        from torchvision.transforms import v2
        self.transform = v2.Compose([
                                        #v2.ToImage(),
                                        v2.ToDtype(torch.float, scale=True),
                                        #v2.Lambda(lambda x: (x - 0.5)*2)
                                        ])
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        #x = np.expand_dims(self.inputs[index], 0)
        x = torch.Tensor(self.inputs[index])
        x = self.transform(x)
        #print(x.shape)
        y = torch.tensor(self.labels[index], dtype=torch.float)
        return x, y
    
    
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
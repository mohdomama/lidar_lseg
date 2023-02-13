import torch
from lidar_fields.dataset import LidarLsegDataset
from lidar_fields.model import LidarLsegModel
from torch.utils.data import random_split, DataLoader
from dataclasses import dataclass, field 
import tyro
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


@dataclass
class Args:
    dataset_dir: str = 'data/lego_loam_map1/' # 'data/dataset_beta/'
    weights_dir: str = 'data/lidar_fields' 
    val_split: float = 0.2
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    mseloss: bool = field(default=False)

    seed: int = 35
    exp_id: str = 'exp1'

args = tyro.cli(Args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    os.makedirs(args.weights_dir, exist_ok=True)


    # Data Prep
    dataset = LidarLsegDataset(args.dataset_dir, device=device)
    total_size = len(dataset)
    val_size = int(args.val_split*total_size)
    train_size = total_size-val_size

    dataset_train, dataset_val = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=0)

    ### Model Load
    model = LidarLsegModel().to(device)

    ### Optimizer ###
    params = list(model.parameters())
    if args.mseloss:
        criterion = nn.MSELoss()
    else: 
        criterion = nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max = 400, eta_min = 1e-5, verbose=True)

    best_vloss = 999999999
    for epoch in range(args.num_epochs):

        ###### Training ########
        running_loss = 0
        t_iter = 0
        model.train()

        for iter, sample in enumerate(tqdm(dataloader_train)):
            t_iter += 1
            pos, feat = sample
            pred_feat = model(pos)

            optimizer.zero_grad()
            if args.mseloss:
                loss = criterion(pred_feat, feat)
            else:
                loss = criterion(pred_feat, feat, torch.ones(feat.shape[0], device=device))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        average_tloss = running_loss/(t_iter)

        ###### Validation ########
        running_vloss = 0
        v_iter = 0
        model.eval()

        with torch.no_grad():
            for iter, vsample in enumerate(dataloader_val):
                v_iter += 1
                pos, feat = sample
                pred_feat = model(pos)
                optimizer.zero_grad()
                if args.mseloss:
                    loss = criterion(pred_feat, feat)
                else:
                    loss = criterion(pred_feat, feat, torch.ones(feat.shape[0], device=device))
                running_vloss += loss.item()

        average_vloss = running_vloss/(v_iter)

        print("Epoch: {}. Loss train: {}. Loss Validation: {}".format(epoch, average_tloss, average_vloss))
        if(average_vloss < best_vloss): # Saving best model
        #    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
           model_path = 'data/model_{}.pt'.format(args.exp_id)
           torch.save(model.state_dict(), model_path)
           best_vloss = average_vloss
    
    breakpoint()


if __name__=='__main__':
    main()
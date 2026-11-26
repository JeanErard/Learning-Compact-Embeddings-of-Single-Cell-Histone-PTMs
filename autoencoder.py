#For fun, try to create autoencoder with pytorch
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from clustering import clustering
from test_clustering import clust_score
from functools import partial
import torch.nn.functional as F


class AutoEncoder(nn.Module) :
    def __init__(self, input_dim, latent_dim=20, hidden_dim=(256, 128, 64), dropout = 0.1) :
        super(AutoEncoder, self).__init__()
        
        #Encoder
        layers, previous = [], input_dim
        for h_dim in hidden_dim :
            layers += [nn.Linear(previous, h_dim), nn.ReLU(), nn.Dropout(dropout)]
            previous = h_dim
        layers.append(nn.Linear(previous, latent_dim))
        self.encoder = nn.Sequential(*layers)
        
        #Decoder
        layers, previous = [], latent_dim
        for h_dim in reversed(hidden_dim) :
            layers += [nn.Linear(previous, h_dim), nn.ReLU(), nn.Dropout(dropout)]
            previous = h_dim
        layers.append(nn.Linear(previous, input_dim))
        self.decoder = nn.Sequential(*layers)
      
    def forward(self, x) :
          emb = self.encoder(x)
          out = self.decoder(emb)
          return out, emb

def train_AE_epoch(model, optimizer, scheduler, criterion, data, batch_idx, epoch, device, input_transform=None):
    model.train()
    loss_history = []
    lr_history = []
    log_every = max(1, len(batch_idx)) // 10
    
    for i, idx in enumerate(batch_idx):
        batch_dense = torch.tensor(data[idx.numpy()].toarray(), dtype=torch.float32)
        batch_dense = batch_dense.to(device)
        x_in = input_transform(batch_dense) if input_transform else batch_dense
        optimizer.zero_grad()
        output,_ = model(x_in)
        loss = criterion(output, batch_dense)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_float = loss.item()

        loss_history.append(loss_float)
    
        if i % log_every == 0:
            print(
                f"Train Epoch: {epoch}-{i:03d} "
                f"batch_loss={loss_float:0.2e} "
            )

    return loss_history

@torch.no_grad()
def extract_emb(model, device, data, batch_size, input_transform=None):
    model.eval()  # Important: eval mode (affects dropout, batch norm etc)
    emb = []
    n_cells = data.shape[0]
    
    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
      
        batch_tensor = torch.tensor(data[start:end].toarray(), dtype=torch.float32).to(device)
        if input_transform:
            batch_tensor = input_transform(batch_tensor)
        _, z = model(batch_tensor)
        
        emb.append(z.cpu().numpy())
        
    return np.vstack(emb)
    
def weighted_MSE(x_reconstruct, x_true, weight_no_zero=10.0) :
    weights = 1.0 + (weight_no_zero - 1.0) * (x_true > 0).float()
    
    loss = weights * (x_true - x_reconstruct)**2
    return torch.mean(loss)

def run_training(model_class, model_kwargs, num_epochs, optimizer_kwargs, criterion, anndata, batch_size, device="cuda", weight_no_zero=10.0, plot=True, input_transform=None):
    if device == 'cuda' and not torch.cuda.is_available() :
        device = 'cpu'
        
    train = anndata[anndata.obs["split_AE"] == "train"].X
    n_train = train.shape[0]
    batch_idx = DataLoader(np.arange(n_train), batch_size=batch_size, shuffle=True)
    
    # ===== Model, Optimizer and Criterion =====
    model = model_class(**model_kwargs).to(device)
    model = model.to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max= len(batch_idx) * num_epochs,
    )
    # ===== Train Model =====
    train_loss_history = []
    emb_eval_history = []
    for epoch in range(1, num_epochs + 1):
        train_loss = train_AE_epoch(
            model, optimizer, scheduler, criterion, train, batch_idx, epoch, device, input_transform=input_transform
        )
        train_loss_history.extend(train_loss)

        anndata.obsm['X_AE'] = extract_emb(model, device, anndata.X, batch_size, input_transform=input_transform)
        emb_eval = eval_emb(anndata)
        emb_eval_history.append(emb_eval)
        
    if not plot :
        return model, train_loss_history, emb_eval_history
    # ===== Plot training curves =====
    n_train = len(train_loss_history)
    t_train = num_epochs * np.arange(n_train) / n_train
    t_eval = np.arange(1, num_epochs+1)

    plt.figure(figsize=(6.4 * 3, 4.8))
    plt.subplot(1, 2, 1)
    plt.plot(t_train, train_loss_history, label="Train loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss (weighted MSE)")
    
    # ===== Plot eval curves =====    
    plt.figure(figsize=(6.4 * 3, 4.8))
    plt.subplot(1, 2, 2)
    plt.plot(t_eval, emb_eval_history, label="Emb. evaluation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    
    plt.tight_layout()
    plt.show()
    
    return model, train_loss_history, emb_eval_history
    
def eval_emb(anndata) :
    clustering(anndata, Kmeans=False, type_obs='X_AE')
    scores = clust_score(anndata, nmi_ari=True, asw_lisi=True, clust_name='leiden', type_obs='X_AE') 
    score_evaluation = np.mean(scores[:2]) - scores[3]
    return score_evaluation

def poisson_loss(out, x_true):
    return F.poisson_nll_loss(out.clamp(max=10), x_true, log_input=True)



import scvi

def AE_peakVI(anndata, seed=0) :
    scvi.settings.seed = seed
    scvi.model.PEAKVI.setup_anndata(anndata)
    peakVI = scvi.model.PEAKVI(anndata)
    peakVI.train()
    anndata.obsm['X_peakVI'] = peakVI.get_latent_representation()
    
def AE_poissonVI(anndata, seed=0) :
    scvi.settings.seed = seed
    scvi.external.POISSONVI.setup_anndata(anndata)
    poissonVI = scvi.external.POISSONVI(anndata)
    poissonVI.train()
    anndata.obsm['X_poissonVI'] = poissonVI.get_latent_representation()
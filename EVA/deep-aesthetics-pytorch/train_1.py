import os
import argparse
import multiprocessing as mp

import nni
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch_aesthetics
from torch_aesthetics.losses import RegRankLoss
from torch_aesthetics.models import RegressionNetwork
from torch_aesthetics.eva import EVA, load_transforms,load_transforms_flip
#from models.u_model import NIMA
#from models.e_model import NIMA
#from models.relic_model import NIMA
#from models.relic1_model import NIMA
from models.relic2_model import NIMA

# Make reproducible
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_model(cfg: DictConfig):
    # Load model
    model = RegressionNetwork(
        backbone=cfg.models.backbone,
        num_attributes=cfg.data.num_attributes,
        pretrained=cfg.models.pretrained
    )
    model = model.to(cfg.device).to(torch.float32)
    return model
  
def opt_loss_fn(cfg,model,optimizer="SGD"):
    # Setup optimizer and loss func
    if(optimizer=="SGD"):
        opt = torch.optim.SGD(
            params=model.parameters(),
            lr=cfg.train.lr,
            momentum=cfg.train.momentum
        )
    if(optimizer=="Adam"):
        opt = torch.optim.Adam(
            params=model.parameters(),
            lr=cfg.train.lr
            )
    loss_fn = RegRankLoss(margin=cfg.train.margin)
    return opt, loss_fn

def create_data_part(cfg: DictConfig):
    # Load datasets
    train_dataset = EVA(
        image_dir=cfg.data.image_dir,
        labels_dir=cfg.data.labels_dir,
        transforms=load_transforms(input_shape=cfg.data.input_shape)
    )
 
    train_dataset_aug = EVA(
        image_dir=cfg.data.image_dir,
        labels_dir=cfg.data.labels_dir,
        
        transforms=load_transforms_flip(input_shape=cfg.data.input_shape)
    )
    
    train_dataset = train_dataset
    
    train_df, test_df = train_test_split(train_dataset, test_size=0.2, random_state=42)

    # Setup dataloaders
    train_dataloader = DataLoader(
        train_df,
        batch_size=cfg.data.batch_size * 2,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=mp.cpu_count(),
        prefetch_factor=cfg.train.num_prefetch
    )
    val_dataloader = DataLoader(
        #val_dataset,
        test_df,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=mp.cpu_count(),
        prefetch_factor=cfg.train.num_prefetch
    )
    return train_dataloader, val_dataloader

def train(model,opt,loss_fn,train_dataloader,writer,epoch,n_iter):
    model.train()

    train_losses = AverageMeter()
    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for batch, (x, y,image_name) in pbar:
        opt.zero_grad()
        x = x.to(cfg.device).to(torch.float32)
        y = y.to(cfg.device).to(torch.float32)[:, 0:]
        x1, x2 = torch.split(x, cfg.data.batch_size, dim=0)
        y1, y2 = torch.split(y, cfg.data.batch_size, dim=0)

        y_pred1 = model(x1)
        y_pred2 = model(x2)
        
        loss, loss_reg, loss_rank = loss_fn(
            y_pred=(y_pred1, y_pred2),
            y_true=(y1, y2)
        )
        #loss = loss_rank    
        loss.backward()
        opt.step()
        pbar.set_description("Epoch {}, Reg Loss: {:.4f}, Rank Loss: {:.4f} ".format(
            epoch, float(loss_reg), float(loss_rank)))

        if n_iter % cfg.train.log_interval == 0:
            writer.add_scalar(
                tag="loss", scalar_value=float(loss), global_step=n_iter
            )
            writer.add_scalar(
                tag="loss_reg", scalar_value=float(loss_reg), global_step=n_iter
            )
            writer.add_scalar(
                tag="loss_rank", scalar_value=float(loss_rank), global_step=n_iter
            )

        train_losses.update(loss.item(), x.size(0))
    return train_losses.avg

def validate(model,val_dataloader,writer,epoch,n_iter):
    # Evaluate
    model.eval()
    test_loss = 0.0
    validate_losses = AverageMeter()
    torch.set_printoptions(precision=3)
    true_score = []
    pred_score = []
    pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))

    for batch, (x, y,image_name) in pbar:
        x = x.to(cfg.device).to(torch.float32)
        y = y.to(cfg.device).to(torch.float32)[:, 0:]
          
        with torch.no_grad():
            y_pred = model(x)
            pscore_np = y_pred.data.cpu().numpy().astype('float')
            tscore_np = y.data.cpu().numpy().astype('float')

            pred_score += pscore_np.mean(axis=1).tolist()
            true_score += tscore_np.mean(axis=1).tolist()
            
            test_loss += F.mse_loss(y_pred, y)
            validate_losses.update(test_loss.item(), x.size(0))
    test_loss /= len(val_dataloader)
    writer.add_scalar(
        tag="test_loss_reg", scalar_value=test_loss, global_step=n_iter
    )
    writer.add_scalar(
        tag="epoch", scalar_value=epoch, global_step=n_iter
    )
    srcc_mean, _ = spearmanr(pred_score, true_score)
    lcc_mean, _ = pearsonr(pred_score, true_score)
    true_score = np.array(true_score)
    true_score_lable = np.where(true_score <= 0.50, 0, 1)
    pred_score = np.array(pred_score)
    pred_score_lable = np.where(pred_score <= 0.50, 0, 1)
    acc = accuracy_score(true_score_lable, pred_score_lable)
    return validate_losses.avg, acc, srcc_mean, lcc_mean


def start_train(cfg):
    writer = SummaryWriter()
    # 1.create_model
    #model = create_model(cfg)
    model = NIMA()
    model = model.to(cfg.device).to(torch.float32)

    opt,loss_fn=opt_loss_fn(cfg,model,"Adam")
    writer = SummaryWriter()
    # 2.create dateset
    train_dataloader, val_dataloader = create_data_part(cfg)
    # 3.start training
    n_iter = 0
    for epoch in range(cfg.train.epochs):
        train_loss = train(model,opt,loss_fn,train_dataloader,writer,epoch,n_iter)
        n_iter += 1

        with open('train.txt', 'a') as f:
            f.write(f"epoch: {epoch}, train_loss: {train_loss:.4f} \n")

        val_loss,vacc,vsrcc,vlcc = validate(model, val_dataloader,writer,epoch,n_iter)

        with open('train.txt', 'a') as f:
            f.write(f"epoch: {epoch}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, vacc: {vacc}, vsrcc: {vsrcc}, vlcc: {vlcc} \n")

        nni.report_intermediate_result(
                {'vacc': vacc, "vsrcc": vsrcc,"vlcc":vlcc, "val_loss": val_loss})

        # 4.save checkpoint
        if cfg.train.save_dir is None:
            cfg.train.save_dir = writer.log_dir

        filename = "{}_epoch_{}_srcc_{:.3f}_lcc_{:.3f}_loss_{:.4f}_.pt".format(
            cfg.data.dataset,epoch,vsrcc,vlcc, val_loss
        )
        torch.save(model.state_dict(), os.path.join(cfg.train.save_dir, filename))

        nni.report_final_result({'vacc': vacc, "vsrcc": vsrcc})

    writer.close()

def check(cfg):
    model = NIMA()
    model = model.to(cfg.device).to(torch.float32)
    model.eval()
    model.load_state_dict(torch.load(cfg.data.model_path),'cuda:1')
    tran_,test_dataloader = create_data_part(cfg)
    loss = 0.0
    xs = []
    y_preds = []
    y_trues = []
    with torch.no_grad():
        for batch, (x, y,file_name) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            x = x.to(cfg.device)
            y = y.to(cfg.device).to(torch.float32)
            y_pred = model(x).cpu()
    #        loss += F.mse_loss(y_pred, y)
            y_preds.append(y_pred)
            xs.append(file_name)
            y_trues.append(y)
        fatten = [item for sublist in xs for item in sublist]
        xs = [[item] for item in fatten]
        y_trues = torch.cat(y_trues,dim=0).cpu().numpy()
        y_preds = torch.cat(y_preds, dim=0).cpu().numpy()
        #y_preds = np.clip(y_preds, -1.0, 1.0)
        print("\n Test preds")

        #print("\nTest Loss: {:.4f}".format(float(loss)))
        #y_true = [np.round(row[0],2) for row in y_trues]
        #y_pred = [np.round(row[0],2) for row in y_preds]
        merged_list = [[x, y.tolist(), z.tolist()] for x, y, z in zip(xs,y_trues, y_preds)]
        print(merged_list)
        print(len(xs))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Path to config.yaml file"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    start_train(cfg)
    #check(cfg)

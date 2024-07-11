import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v3 as imageio
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import torch
from transformers import SwinModel
import glob
import torchmetrics
import time
import psutil
import os
import logging
import random
import argparse



class Config():
    IMAGE_SIZE = 384
    BACKBONE = 'microsoft/swin-large-patch4-window12-384-in22k'
    TARGET_COLUMNS = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']
    N_TARGETS = len(TARGET_COLUMNS)
    BATCH_SIZE = 10
    LR_MAX = 1e-2
    WEIGHT_DECAY = 0.01
    N_EPOCHS = 50
    N_STEPS_PER_EPOCH = N_EPOCHS//BATCH_SIZE
    TRAIN_MODEL = True
    IS_INTERACTIVE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive'
    


CONFIG = Config()

class Dataset(Dataset):
    def __init__(self, X_jpeg_bytes,X_MLP, y, transforms=None):
        self.X_jpeg_bytes = X_jpeg_bytes
        self.X_MLP = X_MLP
        self.y = y
        self.transforms = transforms

    def __len__(self):
        return len(self.X_jpeg_bytes)

    def __getitem__(self, index):
        X_sample = self.transforms(
            image=imageio.imread(self.X_jpeg_bytes[index]),
        )['image']
        X_MLP_data = torch.tensor(self.X_MLP[index])

        y_sample = torch.tensor(self.y[index])

        return X_sample, X_MLP_data, y_sample


def save_model(file, m, o, s, e):
    torch.save({
        'epoch': e,
        'model_state_dict': m.state_dict(),
        'optimizer_state_dict': o.state_dict(),
        'scheduler_state_dict': s.state_dict(),
    }, file)


def load_model(file):
    global model, optimizer, LR_SCHEDULER, start_epoch
    checkpoint = torch.load(file)
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    LR_SCHEDULER.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"{file} loaded.")


def load_swin_model(m, file):
    # global model, optimizer, LR_SCHEDULER, start_epoch
    checkpoint = torch.load(file)
    # start_epoch = checkpoint['epoch'] + 1
    m.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # LR_SCHEDULER.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"{file} loaded.")

def load_mha_model(m, file):
    # global model, optimizer, LR_SCHEDULER, start_epoch
    checkpoint = torch.load(file)
    # start_epoch = checkpoint['epoch'] + 1
    m.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # LR_SCHEDULER.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"{file} loaded.")

def load_bestmodel(m, file):
    # global model, optimizer, LR_SCHEDULER, start_epoch
    checkpoint = torch.load(file)
    # start_epoch = checkpoint['epoch'] + 1
    m.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # LR_SCHEDULER.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"{file} loaded.")


class Swin(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SwinModel.from_pretrained(CONFIG.BACKBONE)
        self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(512, CONFIG.N_TARGETS)

    def forward(self, inputs):
        output = self.backbone(inputs).pooler_output
        output = self.fc1(output)
        return self.fc2(output)

class ModelSwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = Swin()

    def forward(self, image):
        vit = self.vit
        
        outvit = vit(image)
        
        return outvit
    
class ModelSelfAttention(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.mlp = MLP(d_in,d_in*2,d_in*3,d_in*2,d_in)
        
        self.selfmha = SelfAttention(d_in, 42)
        self.selfmha2 = SelfAttention(d_in, 42)
        self.selfmha3 = SelfAttention(d_in, 42)
        self.selfmha4 = SelfAttention(d_in, 43)
        
        self.layernorm = torch.nn.LayerNorm(d_in)

    def forward(self, metadata):
        mlp = self.mlp
        selfmha = self.selfmha
        outselfmha = selfmha(metadata)
        selfmha2 = self.selfmha2
        outselfmha2 = selfmha2(metadata)
        selfmha3 = self.selfmha3
        outselfmha3 = selfmha3(metadata)
        selfmha4 = self.selfmha4
        outselfmha4 = selfmha4(metadata)
        
        outselfmhaall = self.layernorm((torch.cat((outselfmha,outselfmha2, outselfmha3, outselfmha4),dim=-1)+metadata))
        
        outmlp = mlp(outselfmhaall)
        outmlp = self.layernorm((outmlp + outselfmhaall))
        return outmlp

class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_in = 169
        self.selfatt1 = ModelSelfAttention(169)
        self.layernorm = torch.nn.LayerNorm(self.d_in)
        self.fcc = torch.nn.Sequential(
            nn.Linear(self.d_in, 50),
            nn.GELU(),
            nn.Linear(50,6)
        )
    def forward(self, x):
        layernorm = self.layernorm
        att1 = self.selfatt1
        fcc = self.fcc
        out1 = att1(x)
        outall = layernorm(out1+x)
        return fcc(outall)
    
        
        
        

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = Swin()
        load_swin_model(self.vit, "./checkpoint/latest_model-001.pth")
        for param in self.vit.parameters():  # freeze the swin model
            param.requires_grad = False
        self.vit.fc2 = nn.Identity()  # remove the last layer
        
        # self.fc3 = nn.Linear(512, 256)
        
        self.mhaself = MultiHeadSelfAttention()
        load_mha_model(self.mhaself, "./checkpoint/latest_mha_model-001.pth")
        for param in self.mhaself.parameters():
            param.requires_grad=False
        self.mhaself.fcc = nn.Identity()
        # self.vitmlp = MLP(256,350,512,350,256)
        self.d_in = 169
        self.d_out_kq = 512
        self. d_out_v = 169
        self.d2_in = 512
        self.d2_out_kq = 169
        self.d2_out_v = 512
        self.mha1 = MultiHeadAttentionWrapper(
            self.d_in, self.d_out_kq, self.d_out_v, num_heads=4
        )
        self.mha2 = MultiHeadAttentionWrapper(
            self.d2_in, self.d2_out_kq, self.d2_out_v, num_heads=4
        )
        self.joining = JoiningTwoOutputnFlatten(self.d_out_v*4,self.d2_out_v*4, self.d_out_v,self.d2_out_v)
        self.fcc = FullyConnectedLayer(169+512, 6)
    def forward(self, image, metadata):
        vit = self.vit
        # fc3 = self.fc3
        # vitmlp = self.vitmlp
        joining = self.joining
        fcc = self.fcc
        mhaself = self.mhaself
        mha1 = self.mha1
        mha2 = self.mha2
        outvit = vit(image)
        outmlp = mhaself(metadata)
        # outvitmlp = vitmlp(outvit)
        outmha1 = mha1(outvit, outmlp)
        outmha2 = mha2(outmlp, outvit)
        out = joining(outmha1,outmha2, outmlp,outvit)
        return fcc(out)

# class ViT(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.backbone = timm.create_model(
#                 Config.BACKBONE,
#                 num_classes=256,
#                 pretrained=True)
#
#     def forward(self, inputs):
#         return self.backbone(inputs)

class MLP(nn.Module):
    def __init__(self, inputdim, outputdim1,outputdim2, outputdim3, outputdim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputdim, outputdim1),
            nn.GELU(),
            nn.Linear(outputdim1, outputdim2),
            # nn.ReLU(),
            # nn.Linear(outputdim2,outputdim3),
            nn.GELU(),
            nn.Linear(outputdim2, outputdim)
        )
    def forward(self,input):
        return self.layers(input)


class SelfAttention(nn.Module):
    def __init__(self, d_in,d_out):
        
        super().__init__()
        self.d_in = d_in
        self.W_query = nn.Parameter(torch.randn(d_in, d_out) * (1. / np.sqrt(d_out)))
        self.W_key = nn.Parameter(torch.randn(d_in, d_out) * (1. / np.sqrt(d_out)))
        self.W_value = nn.Parameter(torch.randn(d_in, d_out) * (1. / np.sqrt(d_out)))
        
    def forward(self, x_1):
        queries_1 = x_1 @ self.W_query #n x 300

        keys_2 = x_1 @ self.W_key
        values_2 = x_1 @ self.W_value
        
        attn_scores = queries_1.transpose(-2,-1) @ keys_2 #300 x 300
        attn_weights = torch.softmax(
            attn_scores / self.d_in**0.5, dim=-1) 
        context_vec =  values_2 @ attn_weights  #
        return context_vec



class CrossAttention(nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.randn(d_out_kq, d_in) * (1. / np.sqrt(d_out_kq)))
        self.W_key = nn.Parameter(torch.randn(d_in, d_in) * (1. / np.sqrt(d_in)))
        self.W_value = nn.Parameter(torch.randn(d_in, d_in) * (1. / np.sqrt(d_in)))
    def forward(self, x_1, x_2):
        queries_1 = x_1 @ self.W_query

        keys_2 = x_2 @ self.W_key
        values_2 = x_2 @ self.W_value

        attn_scores = queries_1.transpose(-2,-1) @ keys_2
        attn_weights = torch.softmax(
            attn_scores / self.d_out_kq**0.5, dim=-1)

        context_vec = values_2 @ attn_weights
        return context_vec

class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out_kq, d_out_v, num_heads):
        super().__init__()
        self.heads = nn.ModuleList(
            [CrossAttention(d_in, d_out_kq, d_out_v)
             for _ in range(num_heads)]
        )

    def forward(self, input1, input2):
        return torch.cat([head(input1,input2) for head in self.heads], dim=-1)


class JoiningTwoOutputnFlatten(nn.Module):
    def __init__(self,d1nhead1dim, d2nhead2dim, qx1dim,qx2dim ):
        super().__init__()
        self.linear = nn.Linear(d1nhead1dim, qx1dim)
        self.linear2 = nn.Linear(d2nhead2dim, qx2dim)
        self.layernorm1 = nn.LayerNorm(qx1dim)
        self.layernorm2 = nn.LayerNorm(qx2dim)
        

    def forward(self,x1,x2,qx1,qx2):
        # x1 (1, d1*head)and qx1 (1,d1) , x2 (1, d2*head) and qx2 (1,d2)
        x1 = self.linear(x1)
        x2 = self.linear2(x2)
        # x1 and qx1 (1,d1) , x2 and qx2 (1,d2)
        x = self.layernorm1(x1+qx1)
        y = self.layernorm2(x2+qx2)
        out = torch.cat((x,y), dim=1) # output (1, (d1+d2) )

        return out
class FullyConnectedLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim,256),
            nn.GELU(),
            # nn.Linear(1024,512),
            # nn.GELU(),
            nn.Linear(256,128),
            nn.GELU(),
            nn.Linear(128,output_dim)
        )
    def forward(self,x):
        return self.layers(x)
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val.sum()
        self.count += val.numel()
        self.avg = self.sum / self.count


def get_lr_scheduler(optimizer, CONFIG):
    # return torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer=optimizer,
    #     max_lr=CONFIG.LR_MAX,
    #     total_steps=CONFIG.N_STEPS,
    #     pct_start=0.1,
    #     anneal_strategy='cos',
    #     div_factor=1e1,
    #     final_div_factor=1e1,
    # )

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=CONFIG.N_EPOCHS,
        eta_min= 1e-5,

    )
    # return torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer=optimizer,
    #     factor = 0.5,
    #     patience=3,
    #     min_lr=1e-7
    # )

def r2_loss(y_pred, y_true, Y_MEAN, EPS):
    ss_res = torch.sum((y_true - y_pred)**2, dim=0)
    ss_total = torch.sum((y_true - Y_MEAN)**2, dim=0)
    ss_total = torch.maximum(ss_total, EPS)
    r2 = torch.mean(ss_res / ss_total)
    return r2

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args):
    logger = logging.getLogger(__name__)

    tqdm.pandas()

    # Configure logging
    logging.basicConfig(filename='output.txt', level=logging.INFO)

    # # In[3]:

    # parser = argparse.ArgumentParser(description='Resume training script.')

    # # Add the --resume argument
    # parser.add_argument('--resume', type=str, help='Path to the checkpoint file to resume training from')

    # # Parse the arguments
    # args = parser.parse_args()

    logging.info(f'GPU Name: {torch.cuda.get_device_name(0)}')

    EMB_SIZE = 512
    random_seed(100)
    logger.warning("started")

    train = pd.read_csv('./planttraits2024/train.csv')

    if(os.path.exists('./train.pkl')):
        logger.warning("Loading Previous train data")
        train = pd.read_pickle('./train.pkl')
    else:

        logger.warning("Reading data")
        train['file_path'] = train['id'].apply(lambda s: f'./planttraits2024/train_images/{s}.jpeg')
        train['jpeg_bytes'] = train['file_path'].progress_apply(lambda fp: open(fp, 'rb').read())
        train.to_pickle('./train.pkl')
        logger.warning("train data pkl saved")
    logger.warning("train data gaodim")
    for column in CONFIG.TARGET_COLUMNS:
        lower_quantile = train[column].quantile(0.005)
        upper_quantile = train[column].quantile(0.985)
        train = train[(train[column] >= lower_quantile) & (train[column] <= upper_quantile)]
    logger.warning("preprocess train data")
    CONFIG.N_TRAIN_SAMPLES = len(train)
    CONFIG.N_STEPS_PER_EPOCH = (CONFIG.N_TRAIN_SAMPLES // CONFIG.BATCH_SIZE)
    CONFIG.N_STEPS = CONFIG.N_STEPS_PER_EPOCH * CONFIG.N_EPOCHS + 1
    logger.warning("config gaodim")
    logger.warning("Reading test data")
    if(os.path.exists('./test.pkl')):
        logger.warning("Loading Previous test data")
        test = pd.read_pickle('./test.pkl')
    else:
        logger.warning("Reading test data")
        test = pd.read_csv('./planttraits2024/test.csv')
        test['file_path'] = test['id'].apply(lambda s: f'./planttraits2024/test_images/{s}.jpeg')
        test['jpeg_bytes'] = test['file_path'].progress_apply(lambda fp: open(fp, 'rb').read())
        test.to_pickle('./test.pkl')
        logger.warning("test data pkl saved")
    logger.warning(" test data gaodim")
    logger.info('N_TRAIN_SAMPLES:', len(train), 'N_TEST_SAMPLES:', len(test))
    LOG_FEATURES = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']
    logger.warning(" transforming y_train")
    y_train = np.zeros_like(train[CONFIG.TARGET_COLUMNS], dtype=np.float32)
    for target_idx, target in enumerate(CONFIG.TARGET_COLUMNS):
        v = train[target].values
        if target in LOG_FEATURES:
            v = np.log10(v)
        y_train[:, target_idx] = v

    SCALER = StandardScaler()
    y_train = SCALER.fit_transform(y_train)
    logger.warning(" y_train gaodim")
    logger.warning(" x_mlp_train")
    DROP_COLUMNS = ['id','X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean','file_path','jpeg_bytes']
    columns_to_keep = train.columns[~train.columns.isin(DROP_COLUMNS)]
    train.fillna(0,inplace=True)
    x_MLP_train = train[columns_to_keep]
    x_MLP_train = x_MLP_train.astype('float32')

    mlp_scaler_train = StandardScaler()
    x_MLP_scaled_train = mlp_scaler_train.fit_transform(x_MLP_train)
    logger.info(x_MLP_scaled_train.shape)
    logger.warning(" x_mlp_test")
    DROP_COLUMNS = ['id','X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean','file_path','jpeg_bytes']
    columns_to_keep = test.columns[~test.columns.isin(DROP_COLUMNS)]
    x_MLP_test = test[columns_to_keep]
    x_MLP_test = x_MLP_test.astype('float32')

    mlp_scaler_test = StandardScaler()
    x_MLP_scaled_test = mlp_scaler_test.fit_transform(x_MLP_test)
    logger.info(x_MLP_scaled_test.shape)
    logger.warning(" all gaodim")
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    TRAIN_TRANSFORMS = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomSizedCrop(
                [448, 512],
                CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE, w2h_ratio=1.0, p=0.75),
            A.Resize(CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.25),
            A.ImageCompression(quality_lower=85, quality_upper=100, p=0.25),
            A.ToFloat(),
            A.Normalize(mean=MEAN, std=STD, max_pixel_value=1),
            ToTensorV2(),
        ])

    TEST_TRANSFORMS = A.Compose([
            A.Resize(CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE),
            A.ToFloat(),
            A.Normalize(mean=MEAN, std=STD, max_pixel_value=1),
            ToTensorV2(),
        ])
    train_dataset = Dataset(
    train['jpeg_bytes'].values,
    x_MLP_scaled_train,
    y_train,
    TRAIN_TRANSFORMS,
    )

    train_dataloader = DataLoader(
            train_dataset,
            batch_size=CONFIG.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=1,
    )

    test_dataset = Dataset(
        test['jpeg_bytes'].values,
        x_MLP_scaled_test,
        test['id'].values,
        TEST_TRANSFORMS,
    )
    logger.warning("data gaodim")
    if args.train:

        if os.path.exists("./checkpoint/latest_model-001.pth"):
            if os.path.exists("./checkpoint/latest_mha_model-001.pth"):
                model = Model()
                model.cuda()
                total_params = sum(p.numel() for p in model.parameters())
                total_memory_for_weights = total_params * 4
                logger.warning("Total memory needed: "+ str(total_memory_for_weights))

                logger.info(model)

                MAE = torchmetrics.regression.MeanAbsoluteError().cuda()
                R2 = torchmetrics.regression.R2Score(num_outputs=CONFIG.N_TARGETS, multioutput='uniform_average').to('cuda')
                LOSS = AverageMeter()

                Y_MEAN = torch.tensor(y_train).mean(dim=0).cuda()
                EPS = torch.tensor([1e-6]).cuda()
                LOSS_FN = nn.SmoothL1Loss() # r2_loss

                optimizer = torch.optim.AdamW(
                    params=model.parameters(),
                    lr=CONFIG.LR_MAX,
                    weight_decay=CONFIG.WEIGHT_DECAY,
                )

                LR_SCHEDULER = get_lr_scheduler(optimizer, CONFIG)

                logger.warning("Training")
                # new
                logging.info("Start Training:")
                del train_dataset
                best_loss = 1000000000
                for epoch in range(CONFIG.N_EPOCHS):
                    MAE.reset()
                    R2.reset()
                    LOSS.reset()
                    model.train()
                    logging.info(f"Starting Epoch {epoch}...")

                    for step, (X_batch, MLP_batch, y_true) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{CONFIG.N_EPOCHS}")):

                        X_batch = X_batch.cuda()
                        mlp_batch = MLP_batch.cuda()
                        y_true = y_true.cuda()
                        t_start = time.perf_counter_ns()
                        y_pred = model(X_batch, mlp_batch)
                        loss = LOSS_FN(y_pred, y_true)
                        LOSS.update(loss)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        LR_SCHEDULER.step()
                        MAE.update(y_pred, y_true)
                        R2.update(y_pred, y_true)

                        if step%100 == 0  or step == CONFIG.N_STEPS_PER_EPOCH-1:
                            logging.info(f'EPOCH {epoch:02d}, {step + 1:04d}/{CONFIG.N_STEPS_PER_EPOCH} | ' +
                                            f'loss: {LOSS.avg:.4f}, mae: {MAE.compute().item():.4f}, r2: {R2.compute().item():.4f}, ' +
                                            f'step: {(time.perf_counter_ns() - t_start) * 1e-9:.3f}s, lr: {LR_SCHEDULER.get_last_lr()[0]:.2e}')

                            print(
                                    f'EPOCH {epoch+1:02d}, {step+1:04d}/{CONFIG.N_STEPS_PER_EPOCH} | ' +
                                    f'loss: {LOSS.avg:.4f}, mae: {MAE.compute().item():.4f}, r2: {R2.compute().item():.4f}, ' +
                                f'step: {(time.perf_counter_ns()-t_start)*1e-9:.3f}s, lr: {LR_SCHEDULER.get_last_lr()[0]:.2e}')
                    # elif CONFIG.IS_INTERACTIVE:
                    #     print(
                    #         f'\rEPOCH {epoch+1:02d}, {step+1:04d}/{CONFIG.N_STEPS_PER_EPOCH} | ' +
                    #         f'loss: {LOSS.avg:.4f}, mae: {MAE.compute().item():.4f}, r2: {R2.compute().item():.4f}, ' +
                    #         f'step: {(time.perf_counter_ns()-t_start)*1e-9:.3f}s, lr: {LR_SCHEDULER.get_last_lr()[0]:.2e}',
                    #         end='\n' if (step + 1) == CONFIG.N_STEPS_PER_EPOCH else '', flush=True,
                    #     )

                    if LOSS.avg < best_loss:
                        print("New Best mix model")
                        best_loss = LOSS.avg
                        save_model("./checkpoint/mix_model.pth", model, optimizer, LR_SCHEDULER, epoch)

                logger.warning("Evaluation")
                SUBMISSION_ROWS = []
                model.eval()

                for X_sample_test, mlp_sample_test, test_id in tqdm(test_dataset):
                    with torch.no_grad():
                        y_pred = model(X_sample_test.unsqueeze(0).cuda(), mlp_sample_test).detach().cpu().numpy()

                    y_pred = SCALER.inverse_transform(y_pred).squeeze()
                    row = {'id': test_id}

                    for k, v in zip(CONFIG.TARGET_COLUMNS, y_pred):
                        if k in LOG_FEATURES:
                            row[k.replace('_mean', '')] = 10 ** v
                        else:
                            row[k.replace('_mean', '')] = v

                    SUBMISSION_ROWS.append(row)

                submission_df = pd.DataFrame(SUBMISSION_ROWS)
                submission_df.to_csv('submission.csv', index=False)
                print("Submit!")
                # output = joining(mha1(input)) #try this

                # mlp = mlp.to('cuda')
                # print(mlp)
                # print(model)
                # print(mha1)
                # print(mha2)
                # print(joining)
            else:
                
                model = MultiHeadSelfAttention()
                model.cuda()
                total_params = sum(p.numel() for p in model.parameters())
                total_memory_for_weights = total_params * 4
                logger.warning("Total memory needed: "+ str(total_memory_for_weights))

                logger.info(model)

                MAE = torchmetrics.regression.MeanAbsoluteError().cuda()
                R2 = torchmetrics.regression.R2Score(num_outputs=CONFIG.N_TARGETS, multioutput='uniform_average').to('cuda')
                LOSS = AverageMeter()

                Y_MEAN = torch.tensor(y_train).mean(dim=0).cuda()
                EPS = torch.tensor([1e-6]).cuda()
                LOSS_FN = nn.SmoothL1Loss() # r2_loss

                optimizer = torch.optim.AdamW(
                    params=model.parameters(),
                    lr=CONFIG.LR_MAX,
                    weight_decay=CONFIG.WEIGHT_DECAY,
                )

                LR_SCHEDULER = get_lr_scheduler(optimizer, CONFIG)

                logger.warning("Pre-Training Self Attention model")
                # new
                logging.info("Start Pre-Training Self Attention model:")
                del train_dataset
                best_loss = 1000000000
                for epoch in range(CONFIG.N_EPOCHS):
                    MAE.reset()
                    R2.reset()
                    LOSS.reset()
                    model.train()
                    logging.info(f"Starting Epoch {epoch}...")

                    for step, (X_batch, MLP_batch, y_true) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{CONFIG.N_EPOCHS}")):

                        X_batch = X_batch.cuda()
                        mlp_batch = MLP_batch.cuda()
                        y_true = y_true.cuda()
                        t_start = time.perf_counter_ns()
                        y_pred = model(mlp_batch)
                        loss = LOSS_FN(y_pred, y_true)
                        LOSS.update(loss)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        LR_SCHEDULER.step()
                        MAE.update(y_pred, y_true)
                        R2.update(y_pred, y_true)

                        if step%100 == 0 or step == CONFIG.N_STEPS_PER_EPOCH-1:
                            logging.info(f'EPOCH {epoch:02d}, {step + 1:04d}/{CONFIG.N_STEPS_PER_EPOCH} | ' +
                                            f'loss: {LOSS.avg:.4f}, mae: {MAE.compute().item():.4f}, r2: {R2.compute().item():.4f}, ' +
                                            f'step: {(time.perf_counter_ns() - t_start) * 1e-9:.3f}s, lr: {LR_SCHEDULER.get_last_lr()[0]:.2e}')

                            print(
                                    f'EPOCH {epoch+1:02d}, {step+1:04d}/{CONFIG.N_STEPS_PER_EPOCH} | ' +
                                    f'loss: {LOSS.avg:.4f}, mae: {MAE.compute().item():.4f}, r2: {R2.compute().item():.4f}, ' +
                                f'step: {(time.perf_counter_ns()-t_start)*1e-9:.3f}s, lr: {LR_SCHEDULER.get_last_lr()[0]:.2e}')
                    # elif CONFIG.IS_INTERACTIVE:
                    #     print(
                    #         f'\rEPOCH {epoch+1:02d}, {step+1:04d}/{CONFIG.N_STEPS_PER_EPOCH} | ' +
                    #         f'loss: {LOSS.avg:.4f}, mae: {MAE.compute().item():.4f}, r2: {R2.compute().item():.4f}, ' +
                    #         f'step: {(time.perf_counter_ns()-t_start)*1e-9:.3f}s, lr: {LR_SCHEDULER.get_last_lr()[0]:.2e}',
                    #         end='\n' if (step + 1) == CONFIG.N_STEPS_PER_EPOCH else '', flush=True,
                    #     )

                    if LOSS.avg < best_loss:
                        print("New Best MHA model")
                        best_loss = LOSS.avg
                        save_model("./checkpoint/latest_mha_model-001.pth", model, optimizer, LR_SCHEDULER, epoch)


                
                main(args)
        else:
            if (os.path.exists("./checkpoint")==False):
                os.mkdir("./checkpoint")
            model = ModelSwin()
            model.cuda()
            total_params = sum(p.numel() for p in model.parameters())
            total_memory_for_weights = total_params * 4
            logger.warning("Total memory needed: "+ str(total_memory_for_weights))

            logger.info(model)

            MAE = torchmetrics.regression.MeanAbsoluteError().cuda()
            R2 = torchmetrics.regression.R2Score(num_outputs=CONFIG.N_TARGETS, multioutput='uniform_average').to('cuda')
            LOSS = AverageMeter()

            Y_MEAN = torch.tensor(y_train).mean(dim=0).cuda()
            EPS = torch.tensor([1e-6]).cuda()
            LOSS_FN = nn.SmoothL1Loss() # r2_loss

            optimizer = torch.optim.AdamW(
                params=model.parameters(),
                lr=CONFIG.LR_MAX,
                weight_decay=CONFIG.WEIGHT_DECAY,
            )

            LR_SCHEDULER = get_lr_scheduler(optimizer, CONFIG)

            logger.warning("Pre-Training Swin model")
            # new
            logging.info("Start Pre-Training Swin model:")
            del train_dataset
            best_loss = 1000000000
            for epoch in range(CONFIG.N_EPOCHS):
                MAE.reset()
                R2.reset()
                LOSS.reset()
                model.train()
                logging.info(f"Starting Epoch {epoch}...")

                for step, (X_batch, MLP_batch, y_true) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{CONFIG.N_EPOCHS}")):

                    X_batch = X_batch.cuda()
                    mlp_batch = MLP_batch.cuda()
                    y_true = y_true.cuda()
                    t_start = time.perf_counter_ns()
                    y_pred = model(X_batch)
                    loss = LOSS_FN(y_pred, y_true)
                    LOSS.update(loss)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    LR_SCHEDULER.step()
                    MAE.update(y_pred, y_true)
                    R2.update(y_pred, y_true)

                    if step%100 == 0 or step == CONFIG.N_STEPS_PER_EPOCH-1:
                        logging.info(f'EPOCH {epoch:02d}, {step + 1:04d}/{CONFIG.N_STEPS_PER_EPOCH} | ' +
                                        f'loss: {LOSS.avg:.4f}, mae: {MAE.compute().item():.4f}, r2: {R2.compute().item():.4f}, ' +
                                        f'step: {(time.perf_counter_ns() - t_start) * 1e-9:.3f}s, lr: {LR_SCHEDULER.get_last_lr()[0]:.2e}')

                        print(
                                f'EPOCH {epoch+1:02d}, {step+1:04d}/{CONFIG.N_STEPS_PER_EPOCH} | ' +
                                f'loss: {LOSS.avg:.4f}, mae: {MAE.compute().item():.4f}, r2: {R2.compute().item():.4f}, ' +
                            f'step: {(time.perf_counter_ns()-t_start)*1e-9:.3f}s, lr: {LR_SCHEDULER.get_last_lr()[0]:.2e}')
                # elif CONFIG.IS_INTERACTIVE:
                #     print(
                #         f'\rEPOCH {epoch+1:02d}, {step+1:04d}/{CONFIG.N_STEPS_PER_EPOCH} | ' +
                #         f'loss: {LOSS.avg:.4f}, mae: {MAE.compute().item():.4f}, r2: {R2.compute().item():.4f}, ' +
                #         f'step: {(time.perf_counter_ns()-t_start)*1e-9:.3f}s, lr: {LR_SCHEDULER.get_last_lr()[0]:.2e}',
                #         end='\n' if (step + 1) == CONFIG.N_STEPS_PER_EPOCH else '', flush=True,
                #     )

                if LOSS.avg < best_loss:
                    print("New Best Swin model")
                    best_loss = LOSS.avg
                    save_model("./checkpoint/latest_model-001.pth", model, optimizer, LR_SCHEDULER, epoch)


            
            main(args)
    if args.test:
        model = Model()
        if(os.path.exists("./checkpoint/mix_model.pth")==False):
            print("Best model weights not found! Start Training")
            exit()
        load_bestmodel(model,"./checkpoint/mix_model.pth")
        model.cuda()
        total_params = sum(p.numel() for p in model.parameters())
        total_memory_for_weights = total_params * 4
        logger.warning("Total memory needed: "+ str(total_memory_for_weights))

        logger.info(model)

        MAE = torchmetrics.regression.MeanAbsoluteError().cuda()
        R2 = torchmetrics.regression.R2Score(num_outputs=CONFIG.N_TARGETS, multioutput='uniform_average').to('cuda')
        LOSS = AverageMeter()

        Y_MEAN = torch.tensor(y_train).mean(dim=0).cuda()
        EPS = torch.tensor([1e-6]).cuda()
        LOSS_FN = nn.SmoothL1Loss() # r2_loss

        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=CONFIG.LR_MAX,
            weight_decay=CONFIG.WEIGHT_DECAY,
        )

        LR_SCHEDULER = get_lr_scheduler(optimizer, CONFIG)

        logger.warning("Evaluation")
        
        SUBMISSION_ROWS = []
        model.eval()

        for X_sample_test, mlp_sample_test, test_id in tqdm(test_dataset):
            with torch.no_grad():
                y_pred = model(X_sample_test.unsqueeze(0).cuda(), mlp_sample_test).detach().cpu().numpy()

            y_pred = SCALER.inverse_transform(y_pred).squeeze()
            row = {'id': test_id}

            for k, v in zip(CONFIG.TARGET_COLUMNS, y_pred):
                if k in LOG_FEATURES:
                    row[k.replace('_mean', '')] = 10 ** v
                else:
                    row[k.replace('_mean', '')] = v

            SUBMISSION_ROWS.append(row)

        submission_df = pd.DataFrame(SUBMISSION_ROWS)
        submission_df.to_csv('submission.csv', index=False)
        print("Submit!")
        # output = joining(mha1(input)) #try this

        # mlp = mlp.to('cuda')
        # print(mlp)
        # print(model)
        # print(mha1)
        # print(mha2)
        # print(joining)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="parameters")
    parser.add_argument('--train', action='store_true')
    parser.set_defaults(train=False)
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()
    print(args)
    main(args)










import argparse
import os
import os.path as osp
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--n_epochs', default=20, type=int)

parser.add_argument('--lamb', default=0.9, type=float)
parser.add_argument('--lr_G', default=1e-3, type=float)
parser.add_argument('--lr_D', default=2e-4, type=float)

parser.add_argument('--gpus', default='0', type=str)
parser.add_argument('--device', default='cuda', type=str)

parser.add_argument('--data_path', default='./data/', type=str)
parser.add_argument('--dataset', choices=['mnist', 'celeba'])
args = parser.parse_args()


def with_time(s):
    return s + '_' + str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))


def set_random_seed(seed=1000, use_cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed) 
    random.seed(seed) 
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
    print(f'Random state set:{seed}, cuda used: {use_cuda}')


@torch.no_grad()
def sample_img(model, control='cat'):
    model.eval()

    assert control in ['cat', 'cont']
    
    fix_z = torch.randn(1, 62).to(args.device)
    
    if control == 'cat':
        fix_code = torch.zeros(1, 2).to(args.device)
        fig, axes = plt.subplots(1, 10, figsize=(20, 5))
        for axe, i in zip(axes, range(10)):
            one_hot = torch.zeros(10, dtype=torch.long).to(args.device)
            one_hot[i] += 1
            one_hot = one_hot.unsqueeze(0)
            img = model(torch.cat([fix_z, one_hot, fix_code], dim=-1))
            img = img.squeeze(0)
            img = img.squeeze(0)
            axe.imshow(img.cpu().numpy(), cmap='gray') # CHW -> HWC
            axe.axis('off')

    elif control == 'cont':
        one_hot = torch.zeros(10, dtype=torch.long).to(args.device)
        one_hot[0] += 1
        one_hot = one_hot.unsqueeze(0)
        fig, axes = plt.subplots(1, 10, figsize=(20, 5))
        for axe in axes:
            code = torch.FloatTensor(1, 2).uniform_(-1, 1).to(args.device)
            img = model(torch.cat([fix_z, one_hot, code], dim=-1))
            img = img.squeeze(0)
            axe.imshow(img, cmap='gray')
            axe.axis('off')
        
    wandb.log({'images': wandb.Image(plt)})
            

class G(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.dataset == 'mnist':
            self.linear = nn.Sequential(
                nn.Linear(74, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=False),
                nn.Linear(1024, 7 * 7 * 128),
                nn.BatchNorm1d(7 * 7 * 128),
                nn.ReLU(inplace=False)
                )
            
            self.upconv = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=False),
                nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
            )
        else:
            pass
    
    def forward(self, x):
        """_summary_

        :param _type_ x: (B, 74)
        :return _type_: (B, 1, 28, 28)
        """
        x = self.linear(x)
        x = x.view(-1, 128, 7, 7)
        return self.upconv(x)


class DQ(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        if args.dataset == 'mnist':
            init_channel = 1
            Q_out_dim = 12

        # 1 x 28 x 28 -> 1024
        self.conv = nn.Sequential(
            nn.Conv2d(init_channel, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=False)
        )

        self.linear = nn.Sequential(
            nn.Linear(3200, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1, inplace=False)
        )
        
        # 1024 -> 1
        self.D = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        
        # 1024 -> 12 (categorical 10 + cont. 2)
        self.Q = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Linear(128, Q_out_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        B = x.shape[0]
        x = x.view(B, -1) #B, -1

        x = self.linear(x)
        
        fake_pred = self.D(x)

        mi_logits = self.Q(x)
        cat, cont = mi_logits[:, :10].clone(), mi_logits[:, 10:].clone()
        cat = F.softmax(cat, dim=-1)

        return fake_pred, cat, cont

        

def get_categorical(*size):
    B = size[0]
    D = size[-1]
    rands = torch.randint(0, D, (B, ))
    return F.one_hot(rands, num_classes=10)

        
def main(args):
    set_random_seed(args.seed)
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        torch.cuda.set_device(0)


    trans = torchvision.transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    data_path = osp.join(args.data_path, args.dataset)
    if args.dataset =='mnist':
        dataset = torchvision.datasets.MNIST(data_path, transform=trans)
    elif args.dataset =='celeba':
        dataset = torchvision.datasets.CelebA(data_path, transform=trans)
    
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    exp_name = f'infogan-{args.seed}_bs-{args.batch_size}_n_epoch-{args.n_epochs}'
    wandb.init(project='infogan-implement', name=with_time(exp_name))

    model_G = G(args).to(args.device)
    model_DQ = DQ(args).to(args.device)
    
    adv_loss = nn.BCELoss()
    cat_loss = nn.CrossEntropyLoss()
    cont_loss = nn.MSELoss()

    optim_G = torch.optim.Adam(model_G.parameters(), lr=args.lr_G)
    optim_DQ = torch.optim.Adam(model_DQ.parameters(), lr=args.lr_D)
    params = list(model_G.parameters()) + list(model_DQ.parameters())
    optim_MI = torch.optim.Adam(params, lr=args.lr_G)
    
    for epoch in range(args.n_epochs):
        model_G.train()
        model_DQ.train()

        pgs = tqdm(train_loader)
        for i, batch in enumerate(pgs):
            # optim_DQ.zero_grad()

            imgs, _ = batch[0], batch[1]
            imgs = imgs.to(args.device)
            
            B = imgs.shape[0]
            
            real_label = torch.ones(B, requires_grad=False).to(args.device)
            fake_label = torch.zeros(B, requires_grad=False).to(args.device)
                        
            """train D"""
            optim_DQ.zero_grad()

            z_code = torch.randn(B, 62).to(args.device)
            categ_code = get_categorical(B, 10).to(args.device)
            unif_code = torch.FloatTensor(B, 2).uniform_(-1, 1).to(args.device)
            input_G = torch.cat([z_code, categ_code, unif_code], dim=-1)

            fake_imgs = model_G(input_G).to(args.device)
            
            # forward discriminator real 
            real_logits, _, _ = model_DQ(imgs)
            # forward discriminator fake 
            fake_logits, cat_pred, cont_pred = model_DQ(fake_imgs.detach())
            real_logits = real_logits.squeeze(-1)
            fake_logits = fake_logits.squeeze(-1)
            
            D_loss1 = adv_loss(fake_logits, fake_label)
            D_loss2 = adv_loss(real_logits, real_label)
            D_loss = D_loss1 + D_loss2
            D_loss.backward()
            optim_DQ.step()

            """train G"""
            optim_G.zero_grad()

            fake_logits, cat_pred, cont_pred = model_DQ(fake_imgs)
            fake_logits = fake_logits.squeeze(-1)
            G_loss = adv_loss(fake_logits, real_label) #why real label(1.0)? section 3 https://arxiv.org/pdf/1406.2661.pdf
            G_loss.backward()
            optim_G.step()

            """train via Mutual Info term"""
            optim_MI.zero_grad()
            
            z_code = torch.randn(B, 62).to(args.device)
            categ_code = get_categorical(B, 10).to(args.device)
            unif_code = torch.FloatTensor(B, 2).uniform_(-1, 1).to(args.device)
            input_G = torch.cat([z_code, categ_code, unif_code], dim=-1)
            
            fake_imgs = model_G(input_G).to(args.device)
            fake_logits, cat_pred, cont_pred = model_DQ(fake_imgs)
            MI = cat_loss(cat_pred, categ_code.argmax(-1))
            MI += 0.1 * cont_loss(cont_pred, unif_code)
            MI.backward()
            optim_MI.step()
            
            pgs.set_description(f'Epoch: {epoch + 1:04d} | Iter: {i + 1:04d} | G Loss: {G_loss.item():.2f} | D Loss: {D_loss.item():.2f} | MI: {MI.item():.2f} | Real Acc: {(torch.round(real_logits) == real_label).sum() / real_label.shape[0] * 100:.2f}% | Fake Acc: {(torch.round(fake_logits) == fake_label).sum() / fake_label.shape[0] * 100:.2f}%')

        sample_img(model_G, control='cat')

if __name__ == '__main__':
    main(args)
        

    
from __future__ import print_function
import os 
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import argparse

from utils import load_data
from utils import load_hyperparameter
from utils import make_model
from utils import save_model
from utils import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='DTS-SNN')

parser.add_argument('--gpu', default=0, type=int, nargs='+', help='GPU id to use')
parser.add_argument('--dataset', type=str, default='DVS128-Gesture', help='which dataset to run (DVS128-Gesture or N-Cars or SHD)')
parser.add_argument('--temporal_kernel', type=str, default='ktzs', help='which temporal kernel to run (ktzs or kt)')
parser.add_argument('--mode', type=str, default='train', help='whether to train or eval')

# Hyperparameters
parser.add_argument('--ds', type=int, default=1, help='Spatial downsizing factor')
parser.add_argument('--dt', type=int, default=5, help='Temporal sampling factor')
parser.add_argument('--T', type=int, default=300, help='Number of time steps')
parser.add_argument('--num_workers', type=int, default=4, help='')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=200, help='Maximum number of epochs')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')

args = parser.parse_args()

def main():
    ngpus = len(args.gpu)
    train_dl, test_dl = load_data(args.dataset, args.num_workers, args.ds, args.dt, args.T, args.batch_size)   
    criterion = nn.MSELoss()
    
    if args.mode == 'train':
        model = make_model(args.dataset, args.temporal_kernel, args.dt, args.T)
        if ngpus > 1:
            model = nn.DataParallel(model, device_ids=args.gpu)
            model.to(device)
        else:
            model.to(device)       
            
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        acc_hist = list([])
        loss_train_hist = list([])
        loss_test_hist = list([])
        
        for epoch in range(args.num_epochs):
            start_time = time.time()
            
            train_loss = train(model, train_dl, criterion, epoch, optimizer)
            test_loss, acc = test(model, test_dl, criterion)
            
            acc_hist.append(acc)
            loss_train_hist.append(train_loss)
            loss_test_hist.append(test_loss)
            
            print("Epoch: {}/{}.. ".format(epoch+1, args.num_epochs),
                  "Train Loss: {:.5f}.. ".format(train_loss),
                  "Test Loss: {:.5f}.. ".format(test_loss),
                  "Test Accuracy: {:.3f}".format(acc))        
            print('Time elasped: %.2f' %(time.time()-start_time), '\n')
            
            save_model(model, acc, epoch, acc_hist, loss_train_hist, loss_test_hist, ngpus)
    
    elif args.mode == 'eval':
        model = make_model(args.dataset, args.temporal_kernel, args.dt, args.T)
        model = load_model(model, args.dataset, args.temporal_kernel)
        if ngpus > 1:
            model = nn.DataParallel(model, device_ids=args.gpu)
            model.to(device)
        else:
            model.to(device)    
        test_loss, acc = test(model, test_dl, criterion)
        print("Test Loss: {:.5f}.. ".format(test_loss), "Test Accuracy: {:.3f}".format(acc))
        
def train(model, train_loader, criterion, epoch, optimizer):
    ngpus = len(args.gpu)
    train_loss = 0
    
    with tqdm(total=len(train_loader)) as pbar:
        for i, (images, labels) in enumerate(train_loader):
            model.zero_grad()
            optimizer.zero_grad()
            
            if ngpus > 1:
                if model.module.dataset == 'DVS128-Gesture':
                    labels = torch.unique(labels, dim=1).reshape(args.batch_size, -1)
                elif model.module.dataset == 'N-Cars':
                    labels = torch.zeros(args.batch_size, model.module.outs).scatter_(1, labels.view(-1,1), 1)
                else:
                    labels = torch.zeros(args.batch_size, model.module.outs).scatter_(1, labels.to(torch.int64).view(-1,1), 1)                 
                outputs = model(images.to(device), args.batch_size // (torch.cuda.device_count())) 
            else:
                if model.dataset == 'DVS128-Gesture':
                    labels = torch.unique(labels, dim=1).reshape(args.batch_size, -1)
                elif model.dataset == 'N-Cars':
                    labels = torch.zeros(args.batch_size, model.outs).scatter_(1, labels.view(-1,1), 1)
                else:
                    labels = torch.zeros(args.batch_size, model.outs).scatter_(1, labels.to(torch.int64).view(-1,1), 1) 
                outputs = model(images.to(device), args.batch_size) 
                
            loss = criterion(outputs.cpu(), labels)
            train_loss += loss.item() / len(train_loader)
            loss.backward()
            optimizer.step()
            pbar.update(1)
            
    return train_loss

def test(model, test_loader, criterion):
    ngpus = len(args.gpu)
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if ngpus > 1:
                if model.module.dataset == 'DVS128-Gesture':
                    targets = torch.unique(targets, dim=1).reshape(args.batch_size, -1)
                elif model.module.dataset == 'N-Cars':
                    targets = torch.zeros(args.batch_size, model.module.outs).scatter_(1, targets.view(-1,1), 1)
                else:
                    targets = torch.zeros(args.batch_size, model.module.outs).scatter_(1, targets.to(torch.int64).view(-1,1), 1)
                outputs = model(inputs.to(device), args.batch_size // (torch.cuda.device_count()))
            else:
                if model.dataset == 'DVS128-Gesture':
                    targets = torch.unique(targets, dim=1).reshape(args.batch_size, -1)
                elif model.dataset == 'N-Cars':
                    targets = torch.zeros(args.batch_size, model.outs).scatter_(1, targets.view(-1,1), 1)
                else:
                    targets = torch.zeros(args.batch_size, model.outs).scatter_(1, targets.to(torch.int64).view(-1,1), 1)
                outputs = model(inputs.to(device), args.batch_size)
                
            loss = criterion(outputs.cpu(), targets)
            test_loss += loss.item() / len(test_loader)
            _, predicted = outputs.cpu().max(1)
            _, targets_idx = targets.max(1)
            
            total += float(targets.size(0))
            correct += float(predicted.eq(targets_idx).sum().item())
            
    acc = 100. * float(correct) / float(total)    
    return test_loss, acc

if __name__=='__main__':
    main()
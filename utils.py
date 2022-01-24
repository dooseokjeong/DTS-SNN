from __future__ import print_function
import os
import torch
from torch.utils.data import DataLoader

# for 'DVS128-Gesture' dataset_io
#import dataset_io.torchneuromorphic.dvs_gestures.create_hdf5 as create_hdf5
import dataset_io.torchneuromorphic.dvs_gestures.dvsgestures_dataloaders as dvsgestures_dataloaders

# for 'SHD' dataset_io
from dataset_io.spikevision import spikedata

# for 'N-cars' dataset_io
from dataset_io.ebdataset.vision import prophesee_ncars
from dataset_io.ebdataset.vision.transforms import ToDense
from quantities import ms

from network import DTS_SNN_2D
from network import DTS_SNN_1D

def load_data(dataset, num_workers, ds, dt, T, batch_size):
    path = './dataset/'    
    if dataset == 'DVS128-Gesture':
        # manually download from https://www.research.ibm.com/dvsgesture/ 
        # and place under dataset/DVS128-Gesture/
        # create_events_hdf5('./dataset/DVS128-gesture/DvsGesture',
        #'./dataset/DVS128-gesture/dvs128-gestures.hdf5')
        root = path  + dataset + '/dvs128-gestures.hdf5'        
        train_dl, test_dl= dvsgestures_dataloaders.create_dataloader(
        root= root,
        batch_size=batch_size,
        chunk_size_train = T,
        chunk_size_test = T,
        ds=ds,
        dt=dt*1000,
        num_workers=num_workers,
        sample_shuffle=True,
        time_shuffle=False,
        drop_last=True)
        
    elif dataset == 'N-Cars':
        root = path  + dataset + '/Prophesee_Dataset_n_cars' 
        # manually download from https://www.prophesee.ai/2018/03/13/dataset-n-cars/
        # and place under dataset/N-Cars
        train_ds = prophesee_ncars.PropheseeNCars(root, is_train=True, transforms=ToDense(dt*ms))
        test_ds = prophesee_ncars.PropheseeNCars(root, is_train=False, transforms=ToDense(dt*ms))
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
        
    elif dataset == 'SHD':
        root = path  + dataset
        train_ds = spikedata.SHD(root, dt=dt*1000, num_steps=T, train=True)
        test_ds = spikedata.SHD(root, dt=dt*1000, num_steps=T, train=False)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    
    return train_dl, test_dl
        
def load_hyperparameter(dataset, temporal_kernel, dt, T):
    # -- 2D (DVS128-Gesture, N-cars)
    # H, W, r, h_c, w_c, tem_kernel, T, dt, tau_trace, trace_scale_factor, tau_m, tau_s, thresh, hiddens, outs
    # -- 1D (SHD)
    # in_channels, r, in_c, tem_kernel, T, dt, tau_trace, trace_scale_factor, tau_m, tau_s, thresh, hiddens, outs
    if dataset == 'DVS128-Gesture': # 2d
        if temporal_kernel == 'ktzs':
            return 128, 128, 3, 16, 16, 'ktzs', T, dt, (50, 100), (1, 0.5), 60, 20, 0.05, 400, 11
        elif temporal_kernel == 'kt':
            return 128, 128, 3, 16, 16, 'kt', T, dt, 50, 1, 60, 20, 0.05, 400, 11
        
    elif dataset == 'N-Cars': # 2d 
        if temporal_kernel == 'ktzs':
            return 100, 120, 2, 10, 10, 'ktzs', T, dt, (10, 20), (1, 0.5), 12, 4, 0.05, 400, 2
        elif temporal_kernel == 'kt':
            return 100, 120, 2, 10, 10, 'kt', T, dt, 10, 1, 12, 4, 0.05, 400, 2
        
    elif dataset == 'SHD': # 1d
        if temporal_kernel == 'ktzs':
            return 700, 1, 20, 'ktzs', T, dt, (20, 40), (1, 0.5), 40, 20, 0.05, 128, 20
        elif temporal_kernel == 'kt':
            return 700, 1, 20, 'kt', T, dt, 20, 1, 40, 20, 0.05, 128, 20
        
        
def make_model(dataset, temporal_kernel, dt, T):
    if dataset == 'DVS128-Gesture':
        H, W, r, h_c, w_c, tem_kernel, T, dt, tau_trace, trace_scale_factor, tau_m, tau_s, thresh, hiddens, outs \
        = load_hyperparameter(dataset, temporal_kernel, dt, T)
        
        model = DTS_SNN_2D(dataset=dataset, H=H, W=W, r=r, h_c=h_c, w_c=w_c, tem_kernel=tem_kernel, 
                           T=T, dt=dt, tau_trace=tau_trace, trace_scale_factor=trace_scale_factor, tau_m=tau_m, tau_s=tau_s, thresh=thresh,
                           hiddens=hiddens, outs=outs)
    
    elif dataset == 'N-Cars':
        H, W, r, h_c, w_c, tem_kernel, T, dt, tau_trace, trace_scale_factor, tau_m, tau_s, thresh, hiddens, outs \
        = load_hyperparameter(dataset, temporal_kernel, dt, T)
        
        model = DTS_SNN_2D(dataset=dataset, H=H, W=W, r=r, h_c=h_c, w_c=w_c, tem_kernel=tem_kernel, 
                           T=T, dt=dt, tau_trace=tau_trace, trace_scale_factor=trace_scale_factor, tau_m=tau_m, tau_s=tau_s, thresh=thresh,
                           hiddens=hiddens, outs=outs)
        
    elif dataset == 'SHD':
        in_channels, r, in_c, tem_kernel, T, dt, tau_trace, trace_scale_factor, tau_m, tau_s, thresh, hiddens, outs \
        = load_hyperparameter(dataset, temporal_kernel, dt, T)
        
        model = DTS_SNN_1D(dataset=dataset, in_channels=in_channels, r=r, in_c=in_c, tem_kernel=tem_kernel, 
                           T=T, dt=dt, tau_trace=tau_trace, trace_scale_factor=trace_scale_factor, tau_m=tau_m, tau_s=tau_s, thresh=thresh, 
                           hiddens=hiddens, outs=outs)
    
    return model

def save_model(model, acc, epoch, acc_hist, loss_train_hist, loss_test_hist, ngpus):
    state = {
        'net': model.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'acc_hist': acc_hist,
        'loss_train_hist': loss_train_hist,
        'loss_test_hist': loss_test_hist
    } 
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint') 
    
    best_acc = 0
    best_acc = max(acc_hist)
    if ngpus > 1: # for multi-gpu  
        if acc == best_acc:
            torch.save(state, './checkpoint/' + model.module.dataset + '_' + model.module.tem_kernel + '_best'+ '.t7')
        else:
            torch.save(state, './checkpoint/' + model.module.dataset + '_' + model.module.tem_kernel + '.t7')
    else:
        if acc == best_acc:
            torch.save(state, './checkpoint/' + model.dataset + '_' + model.tem_kernel + '_best'+ '.t7')
        else:
            torch.save(state, './checkpoint/' + model.dataset + '_' + model.tem_kernel + '.t7')
        
def load_model(model, dataset, temporal_kernel):
    path =  './pretrained/' + dataset + '_' + temporal_kernel + '.t7'
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
    return model

    

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functions import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DTS_SNN_2D(nn.Module):   
    def __init__(self, dataset, H, W, r, h_c, w_c, tem_kernel, T, dt, tau_trace, trace_scale_factor, tau_m, tau_s, thresh, hiddens, outs):       
        super(DTS_SNN_2D, self).__init__()
        self.dataset = dataset
        self.H = H    # height
        self.W = W    # width
        self.r = r
        self.R = 2*r+1  # size of dynamic time surface 
        self.h_c = h_c  # size of grid cell
        self.w_c = w_c  # size of grid cell
        self.tem_kernel = tem_kernel  # 'ktzs' or 'kt'
        self.T = T    # num of time steps
        self.dt = dt  # temporal sampling time
        
        self.tau_trace = tau_trace
        self.trace_scale_factor = trace_scale_factor
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.thresh = thresh
        
        ### SNN classifier ###
        self.inputs = ((H // h_c)*self.R)*((W // w_c)*self.R)  # number of input neurons
        self.hiddens = hiddens                                 # number of hidden neurons
        self.outs = outs                                       # number of output neurons
        
        ### Learnable parameter ###
        self.dts_encoding = nn.Conv3d(2, 1, kernel_size=(1,h_c,w_c), stride=(1,h_c,w_c), bias=False)
        self.hidden = nn.Linear(self.inputs, self.hiddens, bias=False)
        self.output = nn.Linear(self.hiddens, self.outs, bias=False)
        
        nn.init.xavier_uniform_(self.dts_encoding.weight.data)
        nn.init.xavier_uniform_(self.hidden.weight.data)
        nn.init.xavier_uniform_(self.output.weight.data)
                       
    def forward(self, events, batch_size):
        # events dimension to (batch size, polarity, height, width, num of time steps(=T))
        if self.dataset == 'DVS128-Gesture':
            events_ = torch.clamp(events,0,1).permute(0,2,4,3,1)
        elif self.dataset == 'N-Cars':
            events_ = torch.clamp(events,0,1)
        
        # temporal kernel
        if self.tem_kernel == 'ktzs':
            trace1 = trace2 = torch.zeros(batch_size, 2, self.H, self.W, device=device)
        elif self.tem_kernel == 'kt':
            trace = torch.zeros(batch_size, 2, self.H, self.W, device=device)
                    
        membrane_inp = spike_inp = torch.zeros(batch_size, self.inputs, device=device)
        current_kernel_hdn = membrane_kernel_hdn = spike_hdn = torch.zeros(batch_size, self.hiddens, device=device)
        current_kernel_out = membrane_kernel_out = spike_out = torch.zeros(batch_size, self.outs, device=device)
        
        spike_sum_out = torch.zeros(batch_size, self.outs, device=device)    
        
        if self.tem_kernel == 'ktzs':
            for step in range(self.T):
                trace1, trace2, x = DTS_builder_ktzs_2D(trace1, 
                                                        trace2, 
                                                        events_[:,:,:,:,step],
                                                        self.H, 
                                                        self.W, 
                                                        self.r, 
                                                        self.dt,
                                                        self.tau_trace, 
                                                        self.trace_scale_factor, 
                                                        batch_size)
                
                membrane_inp, spike_inp = DTSbulider_to_input(membrane_inp, 
                                                              spike_inp, 
                                                              self.dts_encoding, 
                                                              x.detach(), 
                                                              self.dt, 
                                                              self.tau_m, 
                                                              self.thresh, 
                                                              batch_size)

                current_kernel_hdn, membrane_kernel_hdn, spike_hdn = Neuronstate_update(current_kernel_hdn,
                                                                                        membrane_kernel_hdn,
                                                                                        spike_hdn, 
                                                                                        self.hidden, 
                                                                                        spike_inp, 
                                                                                        self.dt, 
                                                                                        self.tau_m, 
                                                                                        self.tau_s, 
                                                                                        self.thresh)

                current_kernel_out, membrane_kernel_out, spike_out = Neuronstate_update(current_kernel_out, 
                                                                                        membrane_kernel_out, 
                                                                                        spike_out, 
                                                                                        self.output, 
                                                                                        spike_hdn,
                                                                                        self.dt, 
                                                                                        self.tau_m, 
                                                                                        self.tau_s, 
                                                                                        self.thresh)
                spike_sum_out += spike_out

            output = spike_sum_out / self.T
            
        elif self.tem_kernel == 'kt':
            for step in range(self.T):
                trace, x = DTS_builder_kt_2D(trace, 
                                             events_[:,:,:,:,step],
                                             self.H, 
                                             self.W, 
                                             self.r, 
                                             self.dt,
                                             self.tau_trace, 
                                             self.trace_scale_factor, 
                                             batch_size)

                membrane_inp, spike_inp = DTSbulider_to_input(membrane_inp, 
                                                              spike_inp, 
                                                              self.dts_encoding, 
                                                              x.detach(), 
                                                              self.dt, 
                                                              self.tau_m, 
                                                              self.thresh, 
                                                              batch_size)

                current_kernel_hdn, membrane_kernel_hdn, spike_hdn = Neuronstate_update(current_kernel_hdn, 
                                                                                        membrane_kernel_hdn, 
                                                                                        spike_hdn, 
                                                                                        self.hidden, 
                                                                                        spike_inp,
                                                                                        self.dt, 
                                                                                        self.tau_m, 
                                                                                        self.tau_s, 
                                                                                        self.thresh)

                current_kernel_out, membrane_kernel_out, spike_out = Neuronstate_update(current_kernel_out, 
                                                                                        membrane_kernel_out, 
                                                                                        spike_out, 
                                                                                        self.output, 
                                                                                        spike_hdn,
                                                                                        self.dt, 
                                                                                        self.tau_m, 
                                                                                        self.tau_s, 
                                                                                        self.thresh)
                spike_sum_out += spike_out
                
            output = spike_sum_out / self.T
        
        return output
    
class DTS_SNN_1D(nn.Module):   
    def __init__(self, dataset, in_channels, r, in_c, tem_kernel, T, dt, tau_trace, trace_scale_factor, tau_m, tau_s, thresh, hiddens, outs):       
        super(DTS_SNN_1D, self).__init__() 
        self.dataset = dataset
        self.in_channels = in_channels  # input dimension
        self.r = r
        self.R = 2*r+1  # size of dynamic time surface 
        self.in_c = in_c  # size of cell
        self.tem_kernel = tem_kernel  # 'ktzs' or 'kt'
        self.T = T    # num of time steps
        self.dt = dt  # temporal sampling time
        
        self.tau_trace = tau_trace
        self.trace_scale_factor = trace_scale_factor
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.thresh = thresh
        
        ### SNN classifier ###
        self.inputs = (in_channels // in_c) * self.R  # number of input neurons 
        self.hiddens = hiddens                        # number of hidden neurons
        self.outs = outs                              # number of output neurons
        
        ### Learnable parameter ###
        self.dts_encoding = nn.Conv3d(1, 1, kernel_size=(1,in_c,1), stride=(1,in_c,1), bias=False)
        self.hidden = nn.Linear(self.inputs, self.hiddens, bias=False)
        self.output = nn.Linear(self.hiddens, self.outs, bias=False)
        
        nn.init.xavier_uniform_(self.dts_encoding.weight.data)
        nn.init.xavier_uniform_(self.hidden.weight.data)
        nn.init.xavier_uniform_(self.output.weight.data)
        
    def forward(self, events, batch_size):
        # events dimension to (batch size, in_channels, num of time steps(=T))
        events_ = torch.clamp(events,0,1).permute(0,2,1)
        
        # temporal kernel
        if self.tem_kernel == 'ktzs':
            trace1 = trace2 = torch.zeros(batch_size, self.in_channels, device=device)
        elif self.tem_kernel == 'kt':
            trace = torch.zeros(batch_size, self.in_channels, device=device)
        
        membrane_inp = spike_inp = torch.zeros(batch_size, self.inputs, device=device)
        current_kernel_hdn = membrane_kernel_hdn = spike_hdn = torch.zeros(batch_size, self.hiddens, device=device)
        current_kernel_out = membrane_kernel_out = spike_out = torch.zeros(batch_size, self.outs, device=device)
        
        spike_sum_out = torch.zeros(batch_size, self.outs, device=device)     
        
        if self.tem_kernel == 'ktzs':
            for step in range(self.T):
                trace1, trace2, x = DTS_builder_ktzs_1D(trace1, 
                                                        trace2, 
                                                        events_[:,:,step],
                                                        self.in_channels,  
                                                        self.r, 
                                                        self.dt,
                                                        self.tau_trace, 
                                                        self.trace_scale_factor, 
                                                        batch_size)
                
                x = x.unsqueeze(-1).unsqueeze(-1).permute(0,3,2,1,4).contiguous()
                
                membrane_inp, spike_inp = DTSbulider_to_input(membrane_inp, 
                                                              spike_inp, 
                                                              self.dts_encoding, 
                                                              x.detach(), 
                                                              self.dt, 
                                                              self.tau_m, 
                                                              self.thresh, 
                                                              batch_size)

                current_kernel_hdn, membrane_kernel_hdn, spike_hdn = Neuronstate_update(current_kernel_hdn,
                                                                                        membrane_kernel_hdn,
                                                                                        spike_hdn, 
                                                                                        self.hidden, 
                                                                                        spike_inp, 
                                                                                        self.dt, 
                                                                                        self.tau_m, 
                                                                                        self.tau_s, 
                                                                                        self.thresh)

                current_kernel_out, membrane_kernel_out, spike_out = Neuronstate_update(current_kernel_out, 
                                                                                        membrane_kernel_out, 
                                                                                        spike_out, 
                                                                                        self.output, 
                                                                                        spike_hdn,
                                                                                        self.dt, 
                                                                                        self.tau_m, 
                                                                                        self.tau_s, 
                                                                                        self.thresh)
                spike_sum_out += spike_out

            output = spike_sum_out / self.T
            
        elif self.tem_kernel == 'kt':
            for step in range(self.T):
                trace, x = DTS_builder_kt_1D(trace, 
                                             events_[:,:,step],
                                             self.in_channels, 
                                             self.r, 
                                             self.dt,
                                             self.tau_trace, 
                                             self.trace_scale_factor, 
                                             batch_size)
                
                x = x.unsqueeze(-1).unsqueeze(-1).permute(0,3,2,1,4).contiguous()

                membrane_inp, spike_inp = DTSbulider_to_input(membrane_inp, 
                                                              spike_inp, 
                                                              self.dts_encoding, 
                                                              x.detach(), 
                                                              self.dt, 
                                                              self.tau_m, 
                                                              self.thresh, 
                                                              batch_size)

                current_kernel_hdn, membrane_kernel_hdn, spike_hdn = Neuronstate_update(current_kernel_hdn, 
                                                                                        membrane_kernel_hdn, 
                                                                                        spike_hdn, 
                                                                                        self.hidden, 
                                                                                        spike_inp,
                                                                                        self.dt, 
                                                                                        self.tau_m, 
                                                                                        self.tau_s, 
                                                                                        self.thresh)

                current_kernel_out, membrane_kernel_out, spike_out = Neuronstate_update(current_kernel_out, 
                                                                                        membrane_kernel_out, 
                                                                                        spike_out, 
                                                                                        self.output, 
                                                                                        spike_hdn,
                                                                                        self.dt, 
                                                                                        self.tau_m, 
                                                                                        self.tau_s, 
                                                                                        self.thresh)
                spike_sum_out += spike_out
                
            output = spike_sum_out / self.T
        
        return output
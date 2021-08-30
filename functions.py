import numpy as np
import torch
import torch.nn.functional as F

############################
### DTS_builder function ###
############################

# ktzs, kt = temporal zero sum kernel, single exponential temporal kernel.
# 2D (DVS128-Gesture, N-cars)
# 1D (SHD)

def DTS_builder_ktzs_2D(trace1, trace2, input_spike, H, W, r, dt, tau_trace, trace_scale_factor, batch_size):
    trace1 = trace1 * np.exp(-dt/tau_trace[0]) + (input_spike * trace_scale_factor[0])
    trace2 = trace2 * np.exp(-dt/tau_trace[1]) + (input_spike * trace_scale_factor[1])
    
    trace_pad = F.pad(trace1 - trace2, (r, r, r, r), "constant", 0)
    unfolded = trace_pad.unfold(2, 2*r+1, 1).unfold(3, 2*r+1, 1)
    dts = input_spike.unsqueeze(-1).unsqueeze(-1) * unfolded
    
    return trace1, trace2, F.normalize(dts.reshape(batch_size,2,H,W,-1).permute(0,1,4,2,3).contiguous(), dim=2)

def DTS_builder_kt_2D(trace, input_spike, H, W, r, dt, tau_trace, trace_scale_factor, batch_size):  
    trace = trace * np.exp(-dt/tau_trace) + (input_spike * trace_scale_factor)
    
    trace_pad = F.pad(trace, (r, r, r, r), "constant", 0)
    unfolded = trace_pad.unfold(2, 2*r+1, 1).unfold(3, 2*r+1, 1)
    dts = input_spike.unsqueeze(-1).unsqueeze(-1) * unfolded

    return trace, F.normalize(dts.reshape(batch_size,2,H,W,-1).permute(0,1,4,2,3).contiguous(), dim=2)

def DTS_builder_ktzs_1D(trace1, trace2, input_spike, in_channels, r, dt, tau_trace, trace_scale_factor, batch_size):
    trace1 = trace1 * np.exp(-dt/tau_trace[0]) + (input_spike * trace_scale_factor[0])
    trace2 = trace2 * np.exp(-dt/tau_trace[1]) + (input_spike * trace_scale_factor[1])
    
    trace_pad = F.pad(trace1 - trace2, (r, r), "constant", 0)
    unfolded = trace_pad.unfold(1, 2*r+1, 1)
    dts = input_spike.unsqueeze(-1) * unfolded
    
    return trace1, trace2, F.normalize(dts, dim=2)

def DTS_builder_kt_1D(trace, input_spike, in_channels, r, dt, tau_trace, trace_scale_factor, batch_size):  
    trace = trace * np.exp(-dt/tau_trace) + (input_spike * trace_scale_factor)

    trace_pad = F.pad(trace, (r, r), "constant", 0)
    unfolded = trace_pad.unfold(1, 2*r+1, 1)
    dts = input_spike.unsqueeze(-1) * unfolded

    return trace, F.normalize(dts, dim=2)

####################
### SNN function ###
####################

class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh):
        ctx.thresh = thresh
        ctx.save_for_backward(input)
        return input.gt(thresh).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        thresh = ctx.thresh
        a = thresh
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < a
        return grad_input * temp.float(), None
    
actfun = ActFun.apply
    
def DTSbulider_to_input(membrane, spike, ops, x, dt, tau_m, thresh, batch_size):
    """
    LIF neuron
    ----------
    ops: Conv3d
    x: time surface (current)
    """   
    membrane = membrane * np.exp(-dt/tau_m) * (1-spike) +  ops(x).reshape(batch_size, -1) 
    spike = actfun(membrane, thresh)       
    return membrane, spike

def Neuronstate_update(current_kernel, membrane_kernel, spike, ops, x, dt, tau_m, tau_s, thresh):
    """
    SRM neuron
    ----------
    ops: Linear
    x: spike
    """ 
    const = tau_m / (tau_m - tau_s)
    membrane_kernel = membrane_kernel * np.exp(-dt/tau_m) * (1-spike) +  ops(x)
    current_kernel = current_kernel * np.exp(-dt/tau_s) * (1-spike) +  ops(x)  
    membrane = (membrane_kernel - current_kernel) * const 
    spike = actfun(membrane, thresh)       
    return current_kernel, membrane_kernel, spike
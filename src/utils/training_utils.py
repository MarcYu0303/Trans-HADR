import os
import time
import shutil
import logging
import numpy as np
import torch
import torch.distributed as dist

def adjust_learning_rate(epoch, optimizer, init_lr, decay_gamma, nepoch_decay):
    lr = init_lr * (decay_gamma ** (epoch // nepoch_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def reduce_tensor(tensor, reduction='mean'):
    # clone tensor to avoid overwrite issue 
    rt = tensor.clone()
    # sum tensors from all procs and then distribute to all procs
    dist.all_reduce(rt,op=dist.ReduceOp.SUM)
    if reduction == 'mean':
        return rt / dist.get_world_size()
    elif reduction == 'sum':
        return rt
    else:
        raise ValueError('Reduction type not supported')

def restore(model, state_dict):
    net_state_dict = model.state_dict()
    restore_state_dict = state_dict
    restored_var_names = set()
    print('Restoring:')
    for var_name in restore_state_dict.keys():
        if var_name in net_state_dict:
            var_size = net_state_dict[var_name].size()
            restore_size = restore_state_dict[var_name].size()
            if var_size != restore_size:
                # pass
                print('Shape mismatch for var', var_name, 'expected', var_size, 'got', restore_size)
            else:
                if isinstance(net_state_dict[var_name], torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    net_state_dict[var_name] = restore_state_dict[var_name].data
                try:
                    net_state_dict[var_name].copy_(restore_state_dict[var_name])
                    # print(str(var_name) + ' -> \t' + str(var_size) + ' = ' + str(int(np.prod(var_size) * 4 / 10**6)) + 'MB')
                    restored_var_names.add(var_name)
                except Exception as ex:
                    print('While copying the parameter named {}, whose dimensions in the model are'
                          ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                              var_name, var_size, restore_size))
                    raise ex
    ignored_var_names = sorted(list(set(restore_state_dict.keys()) - restored_var_names))
    unset_var_names = sorted(list(set(net_state_dict.keys()) - restored_var_names))
    if len(ignored_var_names) == 0:
        print('Restored all variables')
    else:
        print('Did not restore:')
        # print('Did not restore:\n\t' + '\n\t'.join(ignored_var_names))
    if len(unset_var_names) == 0:
        print('No new variables')
    else:
        print('Initialized but did not modify')
        # print('Initialized but did not modify:\n\t' + '\n\t'.join(unset_var_names))


def debug_print(txt, debug=False):
    if debug:
        print(txt)

def create_dir(dir_name):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def to_gpu(batch, device):
    for k,v in batch.items():
        if torch.is_tensor(v):
            batch[k] = batch[k].to(device)
    return batch
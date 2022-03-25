import torch
import torch.nn as nn
import time
from models.tt.tt_linear import TTLinear
from torch.profiler import profile, record_function, ProfilerActivity, schedule

set_cuda = False
if set_cuda:
    device = torch.device('cuda:0')
else:
    torch.set_num_threads(4)
    device = torch.device('cpu')

tt_linear = TTLinear(in_features=[8, 20, 20, 18],
                     out_features=[4, 4, 4, 4],
                     ranks=[1, 4, 4, 4, 1],
                     dims=4).to(device)

# setting data size
n = 1
set_data = (True, True) # batch_size & seq_len
batch_size = 16
seq_len = 10
if not set_data[0] & set_data[1]:
    batch_size = 1
elif set_data[0] & (not set_data[1]):
    seq_len = 1
elif (not set_data[0]) & (not set_data[1]):
    batch_size = 1
    seq_len = 1

inputs = torch.rand(seq_len, batch_size, 57600).to(device)

#testing time
with torch.no_grad():
    if set_cuda:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     schedule=torch.profiler.schedule(
                         wait=1,
                         warmup=1,
                         active=5),
                     with_stack=True,
                     record_shapes=True,
                     profile_memory=True) as p:
            with record_function('model_inference'):
                for _ in range(n):
                    tt_linear(inputs)
                    p.step()
                
                torch.cuda.synchronize(device=device)
    else:
        with profile(activities=[ProfilerActivity.CPU],
                     with_stack=True,
                     record_shapes=True,
                     profile_memory=True) as p:
            with record_function('model_inference'):
                end = time.time()
                for _ in range(n):
                    tt_linear(inputs)
                    p.step()
                print('inference time : ', (time.time() - end)/n)

print(p.key_averages().table())

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from models.tt_lstm import TTLSTM

set_cuda = False
if set_cuda:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

tt_lstm = TTLSTM(input_size=[8, 20, 20, 18],
                 hidden_size=[4, 4, 4, 4],
                 ranks=[1, 4, 4, 4, 1],
                 dims=4).to(device)

# setting data size
n = 10
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
                     on_trace_ready=torch.profiler.tensorboard_trace_handler('gpu_log/'),
                     with_stack=True,
                     record_shapes=True,
                     profile_memory=True) as p:
            with record_function('model_inference'):
                for _ in range(n):
                    tt_lstm(inputs)
                    p.step()
                
                torch.cuda.synchronize(device=device)
    else:
        with profile(activities=[ProfilerActivity.CPU],
                     schedule=torch.profiler.schedule(
                         wait=1,
                         warmup=1,
                         active=5),
                     on_trace_ready=torch.profiler.tensorboard_trace_handler('cpu_log/'),
                     with_stack=True,
                     record_shapes=True,
                     profile_memory=True) as p:
            with record_function('model_inference'):
                for _ in range(n):
                    tt_lstm(inputs)
                    p.step()

print(p.key_averages().table())

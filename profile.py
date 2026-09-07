import argparse,json
import torch
from attention import generate,FA2Plan
from qualification import CORPUS
from evidence import save,environment
p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
plan=FA2Plan(generate(CORPUS['cases'][1]));plan();torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    plan();torch.cuda.synchronize()
kernels=sorted({e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA})
if not any('Prefill' in k or 'prefill' in k for k in kernels):raise AssertionError('FA2 kernel absent from CUDA trace')
save(args.output,{'status':'profiled','environment':environment(),'dispatch':plan.dispatch(),'cuda_kernel_names':kernels})

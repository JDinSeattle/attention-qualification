from __future__ import annotations
from dataclasses import dataclass
import copy,math
import torch

def validate(spec):
    if not spec['qo'] or len(spec['qo'])!=len(spec['kv']):raise ValueError('nonempty matching batch lengths required')
    if any(q<1 or k<q for q,k in zip(spec['qo'],spec['kv'])):raise ValueError('v1 requires 1 <= qo_len <= kv_len')
    if spec['hkv']<1 or spec['hq']%spec['hkv']:raise ValueError('GQA requires integral grouping')
    if spec['d'] not in (64,128) or spec['page'] not in (1,16):raise ValueError('unsupported head/page size')
    if spec['window'] < -1 or not math.isfinite(spec['scale']) or spec['scale']<=0:raise ValueError('invalid scale/window')

@dataclass
class Batch:
    spec:dict
    seed:int
    q:torch.Tensor
    keys:list
    values:list
    kpages:torch.Tensor
    vpages:torch.Tensor
    indices:torch.Tensor
    qindptr:torch.Tensor
    kindptr:torch.Tensor
    last:torch.Tensor

def generate(spec,seed=1729):
    validate(spec);spec=copy.deepcopy(spec);g=torch.Generator().manual_seed(seed)
    hq,hkv,d,page=(spec[x] for x in ('hq','hkv','d','page'))
    q=(torch.randn((sum(spec['qo']),hq,d),generator=g)*.7).half()
    keys=[(torch.randn((n,hkv,d),generator=g)*.7).half() for n in spec['kv']]
    values=[torch.randn((n,hkv,d),generator=g).half() for n in spec['kv']]
    counts=[math.ceil(n/page) for n in spec['kv']];npages=sum(counts)
    # Physical page order deliberately differs from logical token order.
    permutation=torch.randperm(npages,generator=g)
    kp=torch.full((npages,page,hkv,d),float('nan'),dtype=torch.float16);vp=kp.clone()
    offset=0
    for k,v,count in zip(keys,values,counts):
        for j in range(count):
            lo=j*page;valid=min(page,len(k)-lo);idx=permutation[offset+j]
            kp[idx,:valid]=k[lo:lo+valid];vp[idx,:valid]=v[lo:lo+valid]
        offset+=count
    def ptr(lengths):return torch.tensor([0]+list(__import__('itertools').accumulate(lengths)),dtype=torch.int32)
    return Batch(spec,seed,q,keys,values,kp,vp,permutation.int(),ptr(spec['qo']),ptr(counts),
                 torch.tensor([(n-1)%page+1 for n in spec['kv']],dtype=torch.int32))

def reference(batch,mutation=None,dtype=torch.float32):
    """Dense logical K/V, independent of page packing, FlashInfer and PyTorch SDPA."""
    spec=batch.spec; outputs=[];lses=[];offset=0;group=spec['hq']//spec['hkv']
    for qn,kn,k,v in zip(spec['qo'],spec['kv'],batch.keys,batch.values):
        q=batch.q[offset:offset+qn].to(dtype);k=k.to(dtype);v=v.to(dtype);offset+=qn
        heads=torch.arange(spec['hq'])//group
        if mutation=='head-map':heads=torch.arange(spec['hq'])%spec['hkv']
        kh=k[:,heads,:].permute(1,0,2);vh=v[:,heads,:].permute(1,0,2)
        scores=(q.permute(1,0,2)@kh.transpose(-1,-2))*(1.0 if mutation=='scale' else spec['scale'])
        pos=torch.arange(qn)+(0 if mutation=='causal-alignment' else kn-qn)
        tokens=torch.arange(kn);mask=tokens[None,:]<=pos[:,None]
        if spec['window']>=0 and mutation!='window':mask &= tokens[None,:]>=pos[:,None]-spec['window']
        if mutation=='tail':mask[:,-1]=False
        scores=scores.masked_fill(~mask[None,:,:],float('-inf'))
        lse=torch.logsumexp(scores,dim=-1)/math.log(2)
        probs=torch.softmax(scores,dim=-1)
        outputs.append((probs@vh).permute(1,0,2));lses.append(lse.T)
    return torch.cat(outputs),torch.cat(lses)

def compare(actual,expected,atol=.003,rtol=.004):
    actual=actual.detach().cpu().float();expected=expected.cpu().float()
    if actual.shape!=expected.shape:raise AssertionError('shape mismatch')
    error=(actual-expected).abs();scaled=error/(atol+rtol*expected.abs())
    if not torch.isfinite(actual).all() or not (scaled<=1).all():
        raise AssertionError(f'max_abs={error.max().item()}, max_scaled={scaled.max().item()}')
    return {'max_abs':error.max().item(),'max_scaled_error':scaled.max().item(),
            'p99_abs':torch.quantile(error.flatten(),.99).item(),'rms':error.square().mean().sqrt().item()}

class FA2Plan:
    def __init__(self,batch):
        import flashinfer
        if not torch.cuda.is_available():raise RuntimeError('real CUDA device required')
        if flashinfer.__version__!='0.6.18.post1':raise RuntimeError('unqualified FlashInfer version; update contract explicitly')
        s=batch.spec;self.q=batch.q.cuda();self.k=batch.kpages.cuda();self.v=batch.vpages.cuda()
        self.workspace=torch.empty(32*1024*1024,dtype=torch.uint8,device='cuda')
        self.wrapper=flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace,kv_layout='NHD',backend='fa2')
        self.wrapper.plan(batch.qindptr,batch.kindptr,batch.indices,batch.last,s['hq'],s['hkv'],s['d'],s['page'],
                          causal=True,sm_scale=s['scale'],window_left=s['window'],q_data_type=torch.float16,
                          kv_data_type=torch.float16,use_fp16_qk_reduction=False,disable_split_kv=True)
        if self.wrapper._backend!='fa2':raise RuntimeError('unexpected backend dispatch')
        shape=self.q.shape;self.storage=torch.full((self.q.numel()+32,),123,device='cuda',dtype=self.q.dtype)
        self.out=self.storage[16:-16].view(shape)
        self.lse_storage=torch.full((shape[0]*shape[1]+32,),123,device='cuda',dtype=torch.float32)
        self.lse=self.lse_storage[16:-16].view(shape[:2])

    def __call__(self):
        result=self.wrapper.run(self.q,(self.k,self.v),out=self.out,lse=self.lse,return_lse=True)
        if result[0].data_ptr()!=self.out.data_ptr() or result[1].data_ptr()!=self.lse.data_ptr():
            raise AssertionError('caller buffer identity violated')
        return result

    def check_guards(self):
        for buffer in [self.storage,self.lse_storage]:
            assert (buffer[:16]==123).all() and (buffer[-16:]==123).all(), 'output guard overwritten'

    def dispatch(self):
        module=self.wrapper._cached_module
        return {'requested':'fa2','actual':self.wrapper._backend,'module':str(type(module)),
                'run_callable':str(getattr(module,'paged_run',None)),
                'flashinfer':'0.6.18.post1','dtype':'float16','layout':'NHD','lse_base':2,'disable_split_kv':True}

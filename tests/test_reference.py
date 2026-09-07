import copy,math
import pytest,torch
from attention import generate,reference,compare,validate
from qualification import CORPUS
from reduce import ddmin,reduce_mutation

@pytest.mark.parametrize('spec',CORPUS['cases'])
def test_fp32_against_fp64(spec):
    b=generate(spec);single=reference(b);double=reference(b,dtype=torch.float64)
    compare(single[0],double[0],atol=2e-6,rtol=2e-6);compare(single[1],double[1],atol=2e-6,rtol=2e-6)

@pytest.mark.parametrize('spec',CORPUS['cases'])
def test_page_reconstruction_and_poison(spec):
    b=generate(spec);offset=0
    for i,n in enumerate(spec['kv']):
        ids=b.indices[b.kindptr[i]:b.kindptr[i+1]].long()
        k=b.kpages[ids].flatten(0,1);v=b.vpages[ids].flatten(0,1)
        assert torch.equal(k[:n],b.keys[i]) and torch.equal(v[:n],b.values[i])
        if len(k)>n:assert torch.isnan(k[n:]).all() and torch.isnan(v[n:]).all()

def test_singleton_analytic():
    b=generate(CORPUS['cases'][0]);out,lse=reference(b)
    for h in range(b.spec['hq']):
        assert torch.equal(out[0,h],b.values[0][0,0].float())
        dot=sum(float(x)*float(y) for x,y in zip(b.q[0,h],b.keys[0][0,0]))
        assert lse[0,h].item()==pytest.approx(dot*b.spec['scale']/math.log(2),abs=2e-6)

def test_window_zero_selects_aligned_value():
    b=generate(CORPUS['cases'][2]);out,_=reference(b);offset=0
    for qn,kn,value in zip(b.spec['qo'],b.spec['kv'],b.values):
        for i in range(qn):
            for h in range(b.spec['hq']):
                expected=value[kn-qn+i,h//(b.spec['hq']//b.spec['hkv'])].float()
                assert torch.equal(out[offset+i,h],expected)
        offset+=qn

def test_gqa_constant_value_groups():
    b=generate(CORPUS['cases'][1])
    for v in b.values:
        for h in range(b.spec['hkv']):v[:,h,:]=h+1
    out,_=reference(b)
    for h in range(b.spec['hq']):
        assert torch.allclose(out[:,h,:],torch.full_like(out[:,h,:],h//4+1),atol=1e-6,rtol=1e-6)

@pytest.mark.parametrize('mutation',['head-map','scale','causal-alignment','tail','window'])
def test_detector_mutations(mutation):
    spec=CORPUS['cases'][3] if mutation=='window' else CORPUS['cases'][1]
    b=generate(spec)
    with pytest.raises(AssertionError):compare(reference(b,mutation)[0],reference(b)[0])

@pytest.mark.parametrize('change',[{'qo':[0],'kv':[1]},{'qo':[2],'kv':[1]},{'hq':3,'hkv':2},{'scale':float('nan')},{'window':-2}])
def test_unsupported_rejected(change):
    spec=copy.deepcopy(CORPUS['cases'][0]);spec.update(change)
    with pytest.raises(ValueError):validate(spec)

def test_reducer():
    assert ddmin([1,2,3,4],lambda x:3 in x)==[3]
    result=reduce_mutation(CORPUS['cases'][1],1729,'head-map')
    assert len(result['spec']['qo'])==1

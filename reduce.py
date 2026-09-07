"""Delta debugging over batch members, retaining a reproducible seed and contract."""
import copy,json,argparse
from attention import generate,reference,compare

def ddmin(items,fails):
    items=list(items)
    if not fails(items):raise ValueError('initial input does not fail')
    granularity=2
    while len(items)>1:
        width=(len(items)+granularity-1)//granularity
        for start in range(0,len(items),width):
            candidate=items[:start]+items[start+width:]
            if candidate and fails(candidate):items=candidate;granularity=max(2,granularity-1);break
        else:
            if granularity>=len(items):break
            granularity=min(len(items),granularity*2)
    return items

def reduce_mutation(spec,seed,mutation):
    def subset(indices):
        s=copy.deepcopy(spec);s['qo']=[spec['qo'][i] for i in indices];s['kv']=[spec['kv'][i] for i in indices];return s
    def fails(indices):
        b=generate(subset(indices),seed)
        try:compare(reference(b,mutation)[0],reference(b)[0]);return False
        except AssertionError:return True
    indices=ddmin(range(len(spec['qo'])),fails)
    return {'kind':'synthetic mutation, not an upstream defect','mutation':mutation,'seed':seed,'spec':subset(indices),'retained_indices':indices}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    corpus=json.load(open('manifests/corpus.json'))
    with open(a.output,'w') as f:json.dump(reduce_mutation(corpus['cases'][1],1729,'head-map'),f,indent=2)

import argparse,json,time,subprocess,sys,tempfile
from pathlib import Path
from evidence import ROOT,environment,save,require_gate,torch_measure
from attention import generate,reference,compare,FA2Plan
CORPUS=json.loads((ROOT/'manifests/corpus.json').read_text())

def check_case(spec):
    rows=[]
    for seed in CORPUS['seeds']:
        batch=generate(spec,seed);start=time.perf_counter();expected=reference(batch);reference_ms=(time.perf_counter()-start)*1000
        plan=FA2Plan(batch);actual=plan();plan.check_guards()
        rows.append({'case':spec['id'],'seed':seed,'output':compare(actual[0],expected[0]),
                     'lse':compare(actual[1],expected[1],atol=.002,rtol=.002),'dispatch':plan.dispatch(),
                     'reference_cpu_ms':reference_ms,'physical_page_indices':batch.indices.tolist(),'guards':'passed'})
    return {'status':'passed','environment':environment(),'records':rows}

def bench_case(spec,gate,samples):
    require_gate(gate);rows=[]
    batch=generate(spec);plan=FA2Plan(batch)
    timed=torch_measure({'fa2-return-lse':plan},samples=samples,inner=10)
    rows.append({'case':spec['id'],'timing':timed,'dispatch':plan.dispatch()})
    return {'status':'measured','environment':environment(),'results':rows}

def isolated(phase,gate=None,samples=31):
    """One case per child; the automatic split-KV merge path is not qualified."""
    rows=[];workers=[]
    for spec in CORPUS['cases']:
        with tempfile.TemporaryDirectory(prefix='fa2-case-') as tmp:
            output=Path(tmp)/'result.json'
            command=[sys.executable,str(Path(__file__).resolve()),phase+'-case','--case',spec['id'],'--output',str(output),'--samples',str(samples)]
            if gate:command+=['--gate',str(Path(gate).resolve())]
            subprocess.run(command,check=True,timeout=600)
            result=json.loads(output.read_text())
            if result['status']!=('passed' if phase=='check' else 'measured'):raise AssertionError('worker did not qualify')
            expected_count=len(CORPUS['seeds']) if phase=='check' else 1
            entries=result['records' if phase=='check' else 'results']
            if len(entries)!=expected_count:raise AssertionError('missing worker cases')
            rows.extend(entries);workers.append({'case':spec['id'],'environment':result['environment']})
    return {'status':'passed' if phase=='check' else 'measured','environment':environment(),
            'records' if phase=='check' else 'results':rows,'workers':workers,
            'process_contract':'one case per process; disable_split_kv=True; automatic split-KV remains unqualified',
            'claim':'FA2 runtime only; no speedup claim against CPU reference'}

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['check','bench','check-case','bench-case']);p.add_argument('--gate');p.add_argument('--case')
    p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=31);a=p.parse_args()
    try:
        if a.phase.endswith('-case'):
            spec=next(s for s in CORPUS['cases'] if s['id']==a.case)
            result=check_case(spec) if a.phase=='check-case' else bench_case(spec,a.gate,a.samples)
        else:
            if a.phase=='bench':require_gate(a.gate)
            result=isolated(a.phase,a.gate,a.samples)
        save(a.output,result)
    except Exception as exc:
        save(a.output,{'status':'failed','environment':environment(),'error':repr(exc)});raise
if __name__=='__main__':main()

"""Численные свидетельства для вопросов о двойственности и альтернативных оптимумах."""
import numpy as np
from scipy import sparse
from Lab1.common import ROOT,SEED,TOL,load_json,save_json,solve
import lab2

def run():
    data=load_json(ROOT/'data/lab2.json'); res=load_json(ROOT/'results/lab2/results.json'); base=res['base']
    c,A,l,u=lab2.model(data); f=np.asarray(base['x']); d=np.asarray(base['row_dual'])
    nbal=10*90; rhs=l[:nbal]; caps=np.asarray([v['capacity'] for v in data['arcs']])
    dual_obj=float(rhs@d[:nbal]+caps@d[nbal:])
    reduced=np.asarray(c-A.T@d).ravel()
    dual_check={'objective':dual_obj,'gap':dual_obj-base['objective'],
      'maximum_dual_violation':max(0.,float(-reduced.min()),float(d[nbal:].max())),
      'max_complementarity':float(max(np.max(abs(f*reduced)),np.max(abs((caps-np.asarray(base['flows']).sum(axis=0))*d[nbal:]))))}
    assert abs(dual_check['gap'])<=TOL and dual_check['maximum_dual_violation']<=TOL
    rng=np.random.default_rng(SEED+999); random_cost=rng.normal(size=len(c))
    augmented=sparse.vstack([A,sparse.csr_matrix(c.reshape(1,-1))],format='csr')
    alternative=solve(random_cost,augmented,np.r_[l,base['objective']],np.r_[u,base['objective']],
                      log=ROOT/'results/lab2/logs/alternative_optimum.log')
    assert alternative['status']=='optimal'
    flows=alternative['x'].reshape(90,28)
    checked=lab2.check(data,flows,base['objective']); assert checked['passed']
    difference=float(np.max(abs(alternative['x']-f)))
    result={'lab2_dual':dual_check,'alternative_optimum':difference>1e-6,
       'maximum_flow_difference':difference,'alternative_flow_verification':checked,
       'alternative_flows':flows}
    save_json(ROOT/'results/diagnostics.json',result)
    print({k:v for k,v in result.items() if k!='alternative_flows'})

if __name__=='__main__': run()

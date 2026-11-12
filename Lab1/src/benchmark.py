"""Задание со звёздочкой: 10 ЛП растущего размера, 5 замеров каждого метода."""
import numpy as np
from scipy import sparse
from common import ROOT,SEED,TOL,solve,save_json,save_csv

def run():
    out=ROOT/'results/benchmark'; out.mkdir(parents=True,exist_ok=True)
    sizes=[(6,3),(12,5),(24,8),(48,12),(96,20),(192,30),(384,50),(768,80),(1536,120),(3072,180)]
    rows=[]; raw=[]
    # Прогрев обеих библиотечных ветвей не входит в измерения.
    for method in ('simplex','ipx'):
        solve([-1.,-2.],[[1.,1.]],[-np.inf],[1.],method=method)
    for index,(n,m) in enumerate(sizes):
        seed=SEED+1000+index; rng=np.random.default_rng(seed)
        A=rng.uniform(.02,1.,(m,n)); witness=rng.dirichlet(np.ones(n)); b=A@witness+rng.uniform(.005,.03,m)
        r=rng.uniform(.03,.2,n); mat=sparse.csr_matrix(np.vstack([np.ones(n),A]))
        objective_values=[]
        for method in ('simplex','ipx'):
            runs=[]
            for repeat in range(5):
                sol=solve(-r,mat,[1,*[-np.inf]*m],[1,*b],method=method,
                          log=out/f'logs/task_{index+1:02d}_{method}_{repeat+1}.log',time_limit=60)
                assert sol['status']=='optimal'; x=sol['x']
                err=max(abs(x.sum()-1),max(0.,-x.min()),max(0.,float(np.max(A@x-b))),abs(r@x+sol['objective']))
                assert err<=TOL
                item={'task':index+1,'seed':seed,'variables':n,'risk_constraints':m,'method':method,
                  'repeat':repeat+1,'time_s':sol['time_s'],'simplex_iterations':sol['simplex_iterations'],
                  'ipm_iterations':sol['ipm_iterations'],'crossover_iterations':sol['crossover_iterations'],
                  'objective':-sol['objective'],'max_violation':err}
                raw.append(item); runs.append(item); objective_values.append(-sol['objective'])
            rows.append({'task':index+1,'seed':seed,'variables':n,'risk_constraints':m,'method':method,
                'median_time_s':float(np.median([v['time_s'] for v in runs])),
                'min_time_s':min(v['time_s'] for v in runs),'max_time_s':max(v['time_s'] for v in runs),
                'simplex_iterations':runs[0]['simplex_iterations'],'ipm_iterations':runs[0]['ipm_iterations'],
                'crossover_iterations':runs[0]['crossover_iterations'],'objective':runs[0]['objective']})
        assert max(objective_values)-min(objective_values)<=TOL
        print(f'Benchmark {index+1}/10: n={n}, m={m}',flush=True)
    save_json(out/'results.json',{'repeats':5,'presolve':True,'threads':1,'rows':rows,'raw':raw})
    save_csv(out/'summary.csv',rows); save_csv(out/'runs.csv',raw)
    return rows


"""ЛР 3. Недельное расписание, LP-релаксация, big-M и жадная эвристика."""
import copy,time
import numpy as np
from scipy import sparse
from common import SEED,TOL,ROOT,solve,save_json,save_csv,load_json

def generate(n=10):
    rng=np.random.default_rng(SEED if n==10 else SEED+n)
    wages=rng.integers(150,231,n); fixed=rng.integers(700,1601,n)
    available=(rng.random((n,7,3))>.08).astype(int)
    cost=np.rint(wages[:,None,None]*8*np.array([1.25,1.,1.10])[None,None,:]
                  +rng.integers(0,151,(n,7,3))).astype(int)
    # Масштабируем одинаковый недельный профиль; базовые требования 1/2/2.
    demand=np.tile(np.maximum(1,np.rint(np.array([1,2,2])*n/10)).astype(int),(7,1))
    return {'seed':SEED if n==10 else SEED+n,'variant':5,'synthetic':True,'operators':n,'days':7,
      'shifts':['Н 00-08','У 08-16','В 16-24'],'shift_hours':8,'weekly_max_shifts':5,
      'regular_hours':32,'availability':available.tolist(),'assignment_cost':cost.tolist(),
      'fixed_week_cost':fixed.tolist(),'overtime_rate_rub_hour':(.5*wages).tolist(),
      'demand':demand.tolist(),'rest_rule':'между соседними рабочими днями следующий индекс смены не меньше предыдущего; это даёт не менее 16 часов отдыха'}

def model(data,M=None):
    n=data['operators']; D=7; S=3; nx=n*D*S; nv=nx+2*n
    def idx(i,d,s): return (i*D+d)*S+s
    rows=[]; cols=[]; vals=[]; low=[]; high=[]; labels=[]
    def add(terms,l,u,label):
        row=len(low)
        for col,value in terms: rows.append(row); cols.append(col); vals.append(value)
        low.append(l); high.append(u); labels.append(label)
    av=np.asarray(data['availability']); demand=np.asarray(data['demand']); cap=data['weekly_max_shifts']
    for d in range(D):
        for s in range(S): add([(idx(i,d,s),1) for i in range(n)],demand[d,s],np.inf,f'cover_{d}_{s}')
    for i in range(n):
        for d in range(D): add([(idx(i,d,s),1) for s in range(S)],-np.inf,1,f'daily_{i}_{d}')
        assignments=[(idx(i,d,s),1) for d in range(D) for s in range(S)]
        add(assignments,-np.inf,cap,f'weekly_{i}')
        big_m=cap if M is None else M
        add(assignments+[(nx+i,-big_m)],-np.inf,0,f'link_{i}')
        add([(col,8) for col,_ in assignments]+[(nx+n+i,-1)],-np.inf,data['regular_hours'],f'overtime_{i}')
        for d in range(D-1):
            for s in range(S):
                for t in range(s): add([(idx(i,d,s),1),(idx(i,d+1,t),1)],-np.inf,1,f'rest_{i}_{d}_{s}_{t}')
    c=np.r_[np.asarray(data['assignment_cost']).ravel(),data['fixed_week_cost'],data['overtime_rate_rub_hour']]
    ub=np.r_[av.ravel(),np.ones(n),np.full(n,max(0,8*cap-data['regular_hours']))]
    integ=np.r_[np.ones(nx+n),np.zeros(n)].astype(int)
    A=sparse.csr_matrix((vals,(rows,cols)),shape=(len(low),nv))
    return c,A,np.array(low),np.array(high),ub,integ,labels

def optimize(data,log,relax=False,M=None,time_limit=60):
    c,A,l,u,ub,integ,labels=model(data,M)
    sol=solve(c,A,l,u,ub=ub,integrality=None if relax else integ,log=log,time_limit=time_limit)
    if sol['x'] is not None:
        n=data['operators']; nx=n*21
        sol['assignments']=sol['x'][:nx].reshape(n,7,3)
        sol['activated']=sol['x'][nx:nx+n]; sol['overtime']=sol['x'][nx+n:]
    return sol

def check(data,assignments,activated,overtime,reported,integer=True,M=None):
    """Проверяем каждое исходное правило без обращения к матрице построителя."""
    x=np.asarray(assignments); y=np.asarray(activated); o=np.asarray(overtime); n=data['operators']
    if x.shape!=(n,7,3) or y.shape!=(n,) or o.shape!=(n,) or not all(np.isfinite(v).all() for v in (x,y,o)) or not np.isfinite(reported):
        return {'passed':False,'reason':'invalid vectors'}
    av=np.asarray(data['availability']); demand=np.asarray(data['demand']); cap=data['weekly_max_shifts']
    load=x.sum(axis=(1,2)); coverage=x.sum(axis=0); errors={}
    def pos(v): return max(0.,float(np.max(v)))
    errors['nonnegative']=max(pos(-x),pos(-y),pos(-o))
    errors['availability']=pos(x-av); errors['binary_upper']=max(pos(x-1),pos(y-1))
    errors['coverage']=pos(demand-coverage); errors['one_shift_per_day']=pos(x.sum(axis=2)-1)
    errors['weekly_limit']=pos(load-cap); errors['big_m']=pos(load-(cap if M is None else M)*y)
    errors['overtime']=pos(8*load-data['regular_hours']-o)
    errors['overtime_upper']=pos(o-max(0,8*cap-data['regular_hours']))
    rest=0.
    for i in range(n):
        for d in range(6):
            for s in range(3):
                for t in range(s): rest=max(rest,float(x[i,d,s]+x[i,d+1,t]-1))
    errors['rest']=max(0.,rest)
    errors['integrality']=max(float(np.max(abs(x-np.rint(x)))),float(np.max(abs(y-np.rint(y))))) if integer else 0.
    objective=float(np.sum(x*np.asarray(data['assignment_cost']))+y@np.asarray(data['fixed_week_cost'])+o@np.asarray(data['overtime_rate_rub_hour']))
    errors['objective']=abs(objective-reported)
    return {'passed':max(errors.values())<=TOL,'max_violation':max(errors.values()),'checks':errors,
      'objective_recomputed':objective,'coverage':coverage,'workload':load,'tolerance':TOL}

def greedy(data,restarts=30):
    start=time.perf_counter(); n=data['operators']; rng=np.random.default_rng(SEED+300)
    av=np.asarray(data['availability']); cost=np.asarray(data['assignment_cost']); demand=np.asarray(data['demand'])
    fixed=np.asarray(data['fixed_week_cost']); rates=np.asarray(data['overtime_rate_rub_hour']); cap=data['weekly_max_shifts']
    tasks=[(d,s) for d in range(7) for s in range(3) for _ in range(demand[d,s])]
    best=None; successes=0
    for attempt in range(restarts):
        order=list(tasks)
        if attempt==0: order.sort(key=lambda ds:(int(av[:,ds[0],ds[1]].sum()),ds[0],ds[1]))
        else: rng.shuffle(order)
        schedule=np.full((n,7),-1,dtype=int); load=np.zeros(n,dtype=int); feasible=True
        for d,s in order:
            candidates=[]
            for i in range(n):
                if not av[i,d,s] or schedule[i,d]>=0 or load[i]>=cap: continue
                if d>0 and schedule[i,d-1]>s: continue
                if d<6 and schedule[i,d+1]>=0 and schedule[i,d+1]<s: continue
                extra=cost[i,d,s]+(fixed[i] if load[i]==0 else 0)
                extra+=rates[i]*(max(0,8*(load[i]+1)-data['regular_hours'])-max(0,8*load[i]-data['regular_hours']))
                candidates.append((extra,i))
            if not candidates: feasible=False; break
            _,i=min(candidates); schedule[i,d]=s; load[i]+=1
        if feasible:
            successes+=1; x=np.zeros((n,7,3))
            for i in range(n):
                for d in range(7):
                    if schedule[i,d]>=0: x[i,d,schedule[i,d]]=1
            y=(load>0).astype(float); o=np.maximum(0,8*load-data['regular_hours'])
            objective=float(np.sum(x*cost)+y@fixed+o@rates)
            if best is None or objective<best['objective']:
                best={'objective':objective,'assignments':x,'activated':y,'overtime':o}
    if best is None: best={'feasible':False}
    else: best['feasible']=True
    best.update(time_s=time.perf_counter()-start,restarts=restarts,successful_restarts=successes)
    return best

def run():
    out=ROOT/'results/lab3'; out.mkdir(parents=True,exist_ok=True)
    data=load_json(ROOT/'data/lab3.json'); deficit=copy.deepcopy(data); stress=copy.deepcopy(data)
    # 20% операторов недоступны всю неделю: первые два идентификатора.
    for d in (deficit,stress):
        d['availability'][0]=np.zeros((7,3),dtype=int).tolist(); d['availability'][1]=np.zeros((7,3),dtype=int).tolist()
    stress['demand']=np.tile([2,2,2],(7,1)).tolist()
    scenarios={'base':data,'deficit':deficit,'stress':stress}; results={}; rows=[]; schedule_rows=[]
    for name,d in scenarios.items():
        sol=optimize(d,out/f'logs/{name}.log',time_limit=60)
        if sol['x'] is not None:
            sol['verification']=check(d,sol['assignments'],sol['activated'],sol['overtime'],sol['objective'])
            assert sol['verification']['passed'],sol['verification']
            for i in range(d['operators']):
                row={'scenario':name,'operator':i+1}
                for day in range(7):
                    selected=np.flatnonzero(sol['assignments'][i,day]>.5)
                    row[f'day_{day+1}']='-' if len(selected)==0 else ['Н','У','В'][selected[0]]
                row['total_shifts']=int(np.rint(sol['assignments'][i].sum())); row['overtime_hours']=sol['overtime'][i]
                schedule_rows.append(row)
        else:
            available_operators=sum(any(any(row) for row in person) for person in d['availability'])
            sol['infeasibility_certificate']={'required_shifts':int(np.sum(d['demand'])),
                 'available_operators':available_operators,'maximum_assignments':available_operators*d['weekly_max_shifts']}
            assert sol['status']=='infeasible'
            assert sol['infeasibility_certificate']['required_shifts']>sol['infeasibility_certificate']['maximum_assignments']
        results[name]=sol
        rows.append({'scenario':name,'status':sol['status'],'objective_rub':sol['objective'],
                     'time_s':sol['time_s'],'mip_gap':sol['mip_gap'],'nodes':sol['mip_nodes']})
    base=results['base']; assert base['status']=='optimal'
    relax=optimize(data,out/'logs/relaxation.log',relax=True)
    relax['verification']=check(data,relax['assignments'],relax['activated'],relax['overtime'],relax['objective'],False)
    assert relax['status']=='optimal' and relax['verification']['passed']
    relax['fractional_assignments']=int(np.sum((relax['assignments']>1e-7)&(relax['assignments']<1-1e-7)))
    relax['fractional_activations']=int(np.sum((relax['activated']>1e-7)&(relax['activated']<1-1e-7)))
    results['relaxation']=relax; results['integrality_gap_percent']=100*(base['objective']-relax['objective'])/base['objective']
    loose_lp=optimize(data,out/'logs/loose_M_lp.log',relax=True,M=10000)
    loose_mip=optimize(data,out/'logs/loose_M_mip.log',M=10000,time_limit=60)
    assert loose_lp['status']=='optimal'
    loose_lp['verification']=check(data,loose_lp['assignments'],loose_lp['activated'],loose_lp['overtime'],loose_lp['objective'],False,10000)
    assert loose_lp['verification']['passed']
    if loose_mip['x'] is not None:
        loose_mip['verification']=check(data,loose_mip['assignments'],loose_mip['activated'],loose_mip['overtime'],loose_mip['objective'],True,10000)
        assert loose_mip['verification']['passed']
    if loose_mip['status']=='optimal': assert abs(loose_mip['objective']-base['objective'])<1e-6
    results['loose_M']={'M':10000,'relaxation':loose_lp,'mip':loose_mip,
        'relaxation_gap_percent':100*(base['objective']-loose_lp['objective'])/base['objective']}
    g=greedy(data)
    if g['feasible']:
        g['verification']=check(data,g['assignments'],g['activated'],g['overtime'],g['objective'])
        assert g['verification']['passed']; g['gap_percent']=100*(g['objective']-base['objective'])/base['objective']
        assert g['gap_percent']>=-1e-7
    else: g['gap_percent']=None
    results['greedy']=g
    scaling=[]
    for n in (10,20,40):
        d=data if n==10 else generate(n); save_json(ROOT/f'data/lab3_scale_{n}.json',d)
        s=base if n==10 else optimize(d,out/f'logs/scale_{n}.log',time_limit=30)
        l=relax if n==10 else optimize(d,out/f'logs/scale_{n}_lp.log',relax=True)
        if s['x'] is not None:
            checked=check(d,s['assignments'],s['activated'],s['overtime'],s['objective']); assert checked['passed']
        else: checked={'passed':False}
        scaling.append({'operators':n,'binary_variables':n*22,'status':s['status'],
          'time_s':s['time_s'],'objective':s['objective'],'lp_bound':l['objective'],
          'mip_dual_bound':s['mip_dual_bound'],'solver_gap_percent':100*s['mip_gap'] if s['mip_gap'] is not None else None,
          'relaxation_gap_percent':100*(s['objective']-l['objective'])/s['objective'] if s['objective'] else None,
          'nodes':s['mip_nodes'],'verification_passed':checked['passed']})
        save_json(out/f'scale_{n}.json',{'mip':s,'lp':l,'verification':checked})
    results['scaling']=scaling
    save_json(out/'results.json',results); save_json(out/'scenario_data.json',scenarios)
    save_csv(out/'scenarios.csv',rows); save_csv(out/'solution.csv',schedule_rows); save_csv(out/'scaling.csv',scaling)
    save_csv(out/'lp_relaxation.csv',[{'operator':i+1,'day':d+1,'shift':['Н','У','В'][s],
        'value':relax['assignments'][i,d,s]} for i in range(10) for d in range(7) for s in range(3)])
    save_csv(out/'activation_comparison.csv',[{'operator':i+1,'integer_y':base['activated'][i],
       'relaxed_y':relax['activated'][i],'loose_M_relaxed_y':loose_lp['activated'][i]} for i in range(10)])
    return results


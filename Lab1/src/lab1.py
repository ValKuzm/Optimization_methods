"""ЛР 1. Портфель, сценарии, независимая проверка и двойственная задача."""
import copy
import numpy as np
from scipy.optimize import linprog
from common import SEED,TOL,ROOT,solve,save_json,save_csv,load_json

RISKS=('market_risk','currency_exposure','illiquid_exposure')

def generate():
    rng=np.random.default_rng(SEED)
    returns=np.array([.06,.09,.085,.15,.135,.17])+rng.integers(-6,7,6)/1000
    market=np.round(np.array([.08,.25,.22,.85,.75,.65])+rng.uniform(-.025,.025,6),3)
    illiquid=np.round(np.clip(np.array([0,.15,.05,.12,.25,.90])+rng.uniform(-.015,.015,6),0,1),3)
    names=['Защитные облигации','Корпоративные облигации','Валютные облигации',
           'Зарубежные акции','Акции внутреннего рынка','Малоликвидные бумаги']
    data={'seed':SEED,'variant':5,'budget_rub':100000,'horizon':'один год',
          'synthetic':True,'risk_limits':dict(zip(RISKS,[.4,.35,.28])), 'assets':[]}
    for i in range(6):
        data['assets'].append({'id':f'A{i+1}','name':names[i],'annual_return':round(float(returns[i]),3),
          'market_risk':float(market[i]),'currency_exposure':float(i in (2,3)),
          'illiquid_exposure':float(illiquid[i])})
    return data

def model(data):
    r=np.array([a['annual_return'] for a in data['assets']])
    A=np.array([[a[k] for a in data['assets']] for k in RISKS])
    b=np.array([data['risk_limits'][k] for k in RISKS])
    return r,A,b

def check(data,x,reported):
    """Только NumPy и исходный JSON; матрицы построителя и объект HiGHS не используются."""
    x=np.asarray(x); assets=data['assets']
    if x.shape!=(len(assets),) or not np.isfinite(x).all() or not np.isfinite(reported):
        return {'passed':False,'reason':'invalid vector or objective'}
    rows=[]; errors=[abs(float(x.sum())-1),max(0.,float(-x.min()))]
    for k,limit in data['risk_limits'].items():
        actual=float(sum(float(a[k])*v for a,v in zip(assets,x)))
        rows.append({'constraint':k,'actual':actual,'limit':limit,'slack':limit-actual})
        errors.append(max(0.,actual-limit))
    objective=float(sum(a['annual_return']*v for a,v in zip(assets,x)))
    errors.append(abs(objective-reported))
    return {'passed':max(errors)<=TOL,'max_violation':max(errors),'tolerance':TOL,
            'return_recomputed':objective,'profit_rub':objective*data['budget_rub'],'constraints':rows}

def optimize(data,log=None):
    r,A,b=model(data)
    return solve(-r,np.vstack([np.ones(len(r)),A]),[1,*[-np.inf]*3],[1,*b],log=log)

def run():
    out=ROOT/'results/lab1'; out.mkdir(parents=True,exist_ok=True)
    data=load_json(ROOT/'data/lab1.json'); r,A,b=model(data)
    scenarios={'base':copy.deepcopy(data),'deficit':copy.deepcopy(data),'stress':copy.deepcopy(data)}
    scenarios['deficit']['risk_limits']['market_risk']*=.8
    scenarios['stress']['risk_limits']['market_risk']*=.6
    scenarios['stress']['risk_limits']['illiquid_exposure']*=.6
    results={}; table=[]; all_rows=[]
    for name,d in scenarios.items():
        sol=optimize(d,out/f'logs/{name}.log'); assert sol['status']=='optimal',sol
        objective=-sol['objective']; test=check(d,sol['x'],objective); assert test['passed'],test
        sol['verification']=test; sol['annual_return']=objective; sol['profit_rub']=d['budget_rub']*objective
        sol['shadow_prices']=-sol['row_dual'][1:]; results[name]=sol
        table.append({'scenario':name,'status':sol['status'],'annual_return':objective,
                      'profit_rub':sol['profit_rub'],'max_violation':test['max_violation'],
                      **{f'x{i+1}':v for i,v in enumerate(sol['x'])}})
        for a,w in zip(d['assets'],sol['x']):
            all_rows.append({'scenario':name,'asset':a['id'],'name':a['name'],'weight':w,
                             'investment_rub':w*d['budget_rub'],'profit_rub':w*d['budget_rub']*a['annual_return']})
    base=results['base']; x=base['x']; mu=base['shadow_prices']
    # Двойственная задача: min b*y+z, A.T*y+z >= r, y>=0, z свободна.
    dual=solve([*b,1],np.column_stack([A.T,np.ones(6)]),r,np.full(6,np.inf),
               lb=[0,0,0,-np.inf],log=out/'logs/dual.log')
    y=dual['x'][:3]; z=dual['x'][3]; dual_slack=A.T@y+z-r
    dual_errors=[max(0.,-float(y.min())),max(0.,-float(dual_slack.min())),
                 abs(float(b@y+z-r@x)),float(np.max(abs(y*(b-A@x)))),
                 float(np.max(abs(x*dual_slack)))]
    dual['verification']={'passed':max(dual_errors)<=TOL,'max_violation':max(dual_errors),
        'duality_gap':float(b@y+z-r@x),'dual_slacks':dual_slack,
        'complementarity_risk':y*(b-A@x),'complementarity_assets':x*dual_slack}
    assert dual['status']=='optimal' and dual['verification']['passed']
    # Второй интерфейс решает ту же задачу; это дополнение к независимой проверке.
    cross=linprog(-r,A_ub=A,b_ub=b,A_eq=np.ones((1,6)),b_eq=[1],method='highs')
    assert cross.success and abs(-cross.fun-base['annual_return'])<=TOL
    base['scipy_crosscheck']={'objective':-float(cross.fun),'difference':abs(-float(cross.fun)-base['annual_return'])}
    # Диапазоны правых частей для текущего базиса. Здесь все 3 риска активны.
    support=np.flatnonzero(x>1e-8); B=np.vstack([np.ones(6),A])[:,support]
    assert B.shape==(4,4) and np.linalg.matrix_rank(B)==4
    inv=np.linalg.inv(B); ranges=[]
    for j,k in enumerate(RISKS):
        v=inv[:,j+1]; low=-np.inf; high=np.inf
        for value,slope in zip(x[support],v):
            if slope>1e-12: low=max(low,-value/slope)
            elif slope< -1e-12: high=min(high,-value/slope)
        delta=1e-4; changed=copy.deepcopy(data); changed['risk_limits'][k]+=delta
        s=optimize(changed,out/f'logs/shadow_{j}.log')
        predicted=mu[j]*delta; actual=-s['objective']-base['annual_return']
        assert abs(actual-predicted)<1e-8
        ranges.append({'risk':k,'base_limit':b[j],'shadow':mu[j],
          'allowable_limit_min':b[j]+low,'allowable_limit_max':b[j]+high,
          'test_delta':delta,'predicted_change':predicted,'actual_change':actual})
    for name in ('deficit','stress'):
        _,_,nb=model(scenarios[name]); results[name]['shadow_prediction']=float(base['annual_return']+mu@(nb-b))
    results['dual']=dual; results['sensitivity']=ranges; results['support']=support
    results['basis_rank']=int(np.linalg.matrix_rank(B)); results['unique_optimum']=bool(np.all(dual_slack[x<1e-8]>1e-8))
    save_json(out/'results.json',results); save_json(out/'scenario_data.json',scenarios)
    save_csv(out/'scenarios.csv',table); save_csv(out/'solution.csv',all_rows); save_csv(out/'sensitivity.csv',ranges)
    save_csv(out/'dual.csv',[{'variable':f'y{i+1}','value':v} for i,v in enumerate(y)]+[{'variable':'z','value':z}])
    return results


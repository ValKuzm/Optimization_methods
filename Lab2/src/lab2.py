"""ЛР 2. Все кратчайшие пути как ЛП, затем совместное распределение потоков."""
import copy, heapq, time
import numpy as np
from scipy import sparse
from common import SEED,TOL,ROOT,solve,save_json,save_csv,load_json

def floyd(data):
    """Независимый алгоритм для проверки расстояний, а не основной решатель."""
    n=len(data['nodes']); d=np.full((n,n),np.inf); np.fill_diagonal(d,0)
    for a in data['arcs']: d[a['u'],a['v']]=min(d[a['u'],a['v']],a['cost'])
    for k in range(n): d=np.minimum(d,d[:,k,None]+d[None,k,:])
    return d

def generate():
    rng=np.random.default_rng(SEED); arcs=[]
    def add(u,v,c): arcs.append({'id':f'E{len(arcs)+1:02d}','u':u,'v':v,'cost':float(c),'capacity':1000.})
    for offset in (0,5):
        for i in range(5):
            u=offset+i; v=offset+(i+1)%5
            add(u,v,int(rng.integers(2,8))); add(v,u,int(rng.integers(2,8)))
    for u,v in ((0,2),(2,0),(5,7),(7,5)): add(u,v,int(rng.integers(3,9)))
    add(4,5,2); add(5,4,2); add(0,9,20); add(9,0,20)
    commodities=[{'source':s,'target':t,'demand':float(rng.integers(1,4))}
                 for s in range(10) for t in range(10) if s!=t]
    data={'seed':SEED,'variant':5,'nodes':list(range(10)),'arcs':arcs,
          'commodities':commodities,'synthetic':True,
          'cost_unit':'условная денежная единица на единицу груза',
          'flow_unit':'единица груза в час'}
    # Внутренние дуги заведомо просторны, межгрупповые коридоры ограничены.
    forward=sum(k['demand'] for k in commodities if k['source']<5<=k['target'])
    backward=sum(k['demand'] for k in commodities if k['target']<5<=k['source'])
    for a in arcs: a['capacity']=200.
    arcs[-4]['capacity']=.75*forward; arcs[-3]['capacity']=.90*backward
    arcs[-2]['capacity']=.60*forward; arcs[-1]['capacity']=.50*backward
    return data

def model(data,capacitated=True,unit=False):
    n=len(data['nodes']); e=len(data['arcs']); k=len(data['commodities'])
    rows=[]; cols=[]; vals=[]; rhs=np.zeros(n*k)
    for j,pair in enumerate(data['commodities']):
        demand=1. if unit else pair['demand']
        rhs[j*n+pair['source']]=demand; rhs[j*n+pair['target']]=-demand
        for a,edge in enumerate(data['arcs']):
            rows.extend([j*n+edge['u'],j*n+edge['v']]); cols.extend([j*e+a]*2); vals.extend([1.,-1.])
    A=sparse.csr_matrix((vals,(rows,cols)),shape=(n*k,e*k)); lower=rhs.copy(); upper=rhs.copy()
    if capacitated:
        cap=sparse.hstack([sparse.eye(e,format='csr')]*k,format='csr')
        A=sparse.vstack([A,cap],format='csr'); lower=np.r_[lower,np.full(e,-np.inf)]
        upper=np.r_[upper,[a['capacity'] for a in data['arcs']]]
    return np.tile([a['cost'] for a in data['arcs']],k),A,lower,upper

def optimize(data,log,capacitated=True,unit=False):
    c,A,l,u=model(data,capacitated,unit)
    sol=solve(c,A,l,u,log=log)
    if sol['x'] is not None: sol['flows']=sol['x'].reshape(len(data['commodities']),len(data['arcs']))
    return sol

def check(data,flows,reported,capacitated=True,unit=False):
    """Проверка исходных дуг и каждой пары по отдельности; построитель матрицы не вызывается."""
    f=np.asarray(flows); k=len(data['commodities']); e=len(data['arcs']); n=len(data['nodes'])
    if f.shape!=(k,e) or not np.isfinite(f).all() or not np.isfinite(reported):
        return {'passed':False,'reason':'invalid flows or objective'}
    max_balance=0.; objective=0.; totals=np.zeros(e)
    for j,pair in enumerate(data['commodities']):
        balance=np.zeros(n)
        for a,edge in enumerate(data['arcs']):
            value=f[j,a]; balance[edge['u']]+=value; balance[edge['v']]-=value
            totals[a]+=value; objective+=edge['cost']*value
        demand=1. if unit else pair['demand']
        expected=np.zeros(n); expected[pair['source']]=demand; expected[pair['target']]=-demand
        max_balance=max(max_balance,float(np.max(abs(balance-expected))))
    cap_error=max(0.,max(totals[a]-edge['capacity'] for a,edge in enumerate(data['arcs']))) if capacitated else 0.
    error=max(max_balance,cap_error,max(0.,float(-f.min())),abs(objective-reported))
    return {'passed':error<=TOL,'max_violation':error,'max_balance_error':max_balance,
            'capacity_violation':cap_error,'objective_recomputed':objective,'tolerance':TOL}

def greedy(data):
    start=time.perf_counter(); e=len(data['arcs']); f=np.zeros((len(data['commodities']),e))
    remaining=np.array([a['capacity'] for a in data['arcs']]); adjacency=[[] for _ in data['nodes']]
    for j,a in enumerate(data['arcs']): adjacency[a['u']].append((a['v'],a['cost'],j))
    unserved=0.
    for k in sorted(range(len(data['commodities'])),key=lambda k:(-data['commodities'][k]['demand'],k)):
        pair=data['commodities'][k]; needed=pair['demand']
        while needed>1e-9:
            d=[np.inf]*len(adjacency); prev=[None]*len(adjacency); d[pair['source']]=0.; heap=[(0.,pair['source'])]
            while heap:
                cost,u=heapq.heappop(heap)
                if cost>d[u]+1e-9: continue
                if u==pair['target']: break
                for v,w,a in adjacency[u]:
                    if remaining[a]>1e-9 and cost+w<d[v]-1e-9:
                        d[v]=cost+w; prev[v]=(u,a); heapq.heappush(heap,(cost+w,v))
            if not np.isfinite(d[pair['target']]): unserved+=needed; break
            route=[]; v=pair['target']
            while v!=pair['source']: u,a=prev[v]; route.append(a); v=u
            amount=min(needed,min(remaining[a] for a in route))
            for a in route: f[k,a]+=amount; remaining[a]-=amount
            needed-=amount
    objective=float(np.sum(f*np.array([a['cost'] for a in data['arcs']])))
    return {'flows':f,'objective':objective,'unserved':unserved,'time_s':time.perf_counter()-start,
            'feasible':unserved<1e-7}

def extract_path(data,flow,source,target):
    """Один кратчайший путь из положительных дуг оптимального единичного LP-потока."""
    adjacency={v:[] for v in data['nodes']}
    for a,value in zip(data['arcs'],flow):
        if value>1e-8: adjacency[a['u']].append(a)
    def visit(node,seen):
        if node==target: return [node],[],0.
        for arc in adjacency[node]:
            if arc['v'] in seen: continue
            tail=visit(arc['v'],seen|{arc['v']})
            if tail is not None:
                nodes,arcs,cost=tail
                return [node]+nodes,[arc['id']]+arcs,arc['cost']+cost
        return None
    found=visit(source,{source})
    if found is None: raise ValueError('В положительном LP-потоке не найден путь')
    return found

def cut_certificate(data):
    """Простой достаточный сертификат несовместимости: перебираем все собственные разрезы."""
    n=len(data['nodes']); best=None
    for mask in range(1,(1<<n)-1):
        S={v for v in range(n) if mask>>v&1}
        required=sum(k['demand'] for k in data['commodities'] if k['source'] in S and k['target'] not in S)
        capacity=sum(a['capacity'] for a in data['arcs'] if a['u'] in S and a['v'] not in S)
        if required>capacity+1e-7 and (best is None or required-capacity>best['shortage']):
            best={'source_side':sorted(S),'required':required,'capacity':capacity,'shortage':required-capacity}
    return best

def run():
    out=ROOT/'results/lab2'; out.mkdir(parents=True,exist_ok=True)
    data=load_json(ROOT/'data/lab2.json'); e=len(data['arcs'])
    shortest=optimize(data,out/'logs/all_pairs.log',capacitated=False,unit=True)
    assert shortest['status']=='optimal'
    shortest['verification']=check(data,shortest['flows'],shortest['objective'],False,True)
    distances=floyd(data); costs=np.array([a['cost'] for a in data['arcs']]); rows=[]
    max_dist_error=0.
    for k,p in enumerate(data['commodities']):
        distance=float(shortest['flows'][k]@costs)
        max_dist_error=max(max_dist_error,abs(distance-distances[p['source'],p['target']]))
        nodes,arcs,path_cost=extract_path(data,shortest['flows'][k],p['source'],p['target'])
        assert abs(path_cost-distance)<=TOL
        rows.append({'source':p['source']+1,'target':p['target']+1,'distance_lp':distance,
                     'distance_floyd':distances[p['source'],p['target']],
                     'path_nodes':' -> '.join(str(v+1) for v in nodes),'path_arcs':' -> '.join(arcs)})
    assert shortest['verification']['passed'] and max_dist_error<=TOL
    shortest['floyd_max_difference']=max_dist_error
    base=optimize(data,out/'logs/base.log'); assert base['status']=='optimal'
    base['verification']=check(data,base['flows'],base['objective']); assert base['verification']['passed']
    totals=base['flows'].sum(axis=0); failed=int(np.argmax(totals))
    growth=copy.deepcopy(data)
    for k in growth['commodities']: k['demand']*=1.2
    failure=copy.deepcopy(data); failure['arcs'][failed]['capacity']=0.
    scenarios={'base':data,'growth_20':growth,'arc_failure':failure}; results={'shortest':shortest}
    scenario_rows=[]; flow_rows=[]
    for name,d in scenarios.items():
        sol=base if name=='base' else optimize(d,out/f'logs/{name}.log')
        if sol['status']=='optimal':
            sol['verification']=check(d,sol['flows'],sol['objective']); assert sol['verification']['passed']
            aggregate=sol['flows'].sum(axis=0)
            sol['capacity_shadow_prices']=-sol['row_dual'][-e:]
            for a,edge in enumerate(d['arcs']):
                flow_rows.append({'scenario':name,'arc':edge['id'],'from':edge['u']+1,'to':edge['v']+1,
                    'flow':aggregate[a],'capacity':edge['capacity'],'cost':edge['cost'],
                    'shadow_saving':sol['capacity_shadow_prices'][a]})
        else:
            sol['infeasibility_certificate']=cut_certificate(d)
            assert sol['status']=='infeasible' and sol['infeasibility_certificate'] is not None
        results[name]=sol
        scenario_rows.append({'scenario':name,'status':sol['status'],'cost':sol['objective'],'time_s':sol['time_s'],
                             'verification_passed':sol.get('verification',{}).get('passed','not applicable')})
    g=greedy(data)
    if g['feasible']:
        g['verification']=check(data,g['flows'],g['objective']); assert g['verification']['passed']
        g['gap_percent']=100*(g['objective']-base['objective'])/base['objective']
        assert g['gap_percent']>=-1e-7
    else: g['gap_percent']=None
    results['greedy']=g; results['failed_arc']=data['arcs'][failed]
    results['failed_arc_base_flow']=totals[failed]; results['distances']=distances
    # Малое увеличение насыщенной ёмкости и проверка теневой цены.
    active=np.flatnonzero(np.asarray(base['capacity_shadow_prices'])>1e-8); sensitivity=[]
    for a in active:
        d=copy.deepcopy(data); d['arcs'][a]['capacity']+=.01
        s=optimize(d,out/f'logs/shadow_{a}.log'); actual=base['objective']-s['objective']
        pred=.01*base['capacity_shadow_prices'][a]
        assert abs(actual-pred)<1e-6
        sensitivity.append({'arc':data['arcs'][a]['id'],'delta_capacity':.01,'predicted_saving':pred,'actual_saving':actual})
    results['sensitivity']=sensitivity
    save_json(out/'results.json',results); save_json(out/'scenario_data.json',scenarios)
    save_csv(out/'shortest_paths.csv',rows); save_csv(out/'solution.csv',flow_rows); save_csv(out/'scenarios.csv',scenario_rows)
    save_csv(out/'distance_matrix.csv',[{'node':i+1,**{str(j+1):distances[i,j] for j in range(10)}} for i in range(10)])
    save_csv(out/'commodity_flows.csv',[{'pair':k+1,'source':p['source']+1,'target':p['target']+1,
      'arc':edge['id'],'flow':base['flows'][k,a]} for k,p in enumerate(data['commodities'])
      for a,edge in enumerate(data['arcs']) if base['flows'][k,a]>1e-9])
    return results


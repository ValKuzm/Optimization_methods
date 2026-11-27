"""Общий интерфейс HiGHS и сохранение результатов для ЛР 1."""
from pathlib import Path
import json, csv, time, platform, importlib.metadata
import numpy as np
from scipy import sparse
import highspy

SEED = 10212
TOL = 1e-7
ROOT = Path(__file__).resolve().parents[1]

def clean(value):
    if isinstance(value, np.ndarray): return [clean(x) for x in value.tolist()]
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [clean(v) for v in value]
    return value

def save_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def save_csv(path, rows, fields=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    rows=list(rows)
    if fields is None: fields=list(rows[0]) if rows else []
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(clean(rows))

def load_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def solve(c, A, lower, upper, *, lb=None, ub=None, integrality=None,
          method='simplex', log=None, presolve=True, time_limit=60.0, gap=1e-8):
    """min c*x, lower <= A*x <= upper. Таймер измеряет только h.run()."""
    c=np.asarray(c,dtype=float); A=sparse.csr_matrix(A,dtype=float)
    lp=highspy.HighsLp(); lp.num_col_=len(c); lp.num_row_=A.shape[0]
    lp.col_cost_=c; lp.col_lower_=np.zeros(len(c)) if lb is None else np.asarray(lb,dtype=float)
    lp.col_upper_=np.full(len(c),highspy.kHighsInf) if ub is None else np.asarray(ub,dtype=float)
    lp.row_lower_=np.asarray(lower,dtype=float); lp.row_upper_=np.asarray(upper,dtype=float)
    lp.a_matrix_.format_=highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_=A.indptr.astype(np.int32)
    lp.a_matrix_.index_=A.indices.astype(np.int32); lp.a_matrix_.value_=A.data
    if integrality is not None:
        lp.integrality_=[highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous for v in integrality]
    h=highspy.Highs()
    options={'output_flag':bool(log),'log_to_console':False,'threads':1,
             'random_seed':SEED,'presolve':'on' if presolve else 'off',
             'solver':'choose' if integrality is not None and np.any(integrality) else method,
             'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,
             'mip_feasibility_tolerance':1e-8,'mip_rel_gap':gap,'time_limit':float(time_limit),
             'run_crossover':'on'}
    if log:
        log=Path(log); log.parent.mkdir(parents=True,exist_ok=True)
        if log.exists(): log.unlink()
        options['log_file']=str(log.resolve())
    for key,value in options.items():
        status=h.setOptionValue(key,value)
        if status != highspy.HighsStatus.kOk: raise RuntimeError(f'HiGHS option rejected: {key}={value}')
    if h.passModel(lp)!=highspy.HighsStatus.kOk: raise RuntimeError('HiGHS rejected model')
    start=time.perf_counter(); run_status=h.run(); elapsed=time.perf_counter()-start
    if run_status==highspy.HighsStatus.kError: raise RuntimeError('HiGHS run failed')
    st=h.getModelStatus(); info=h.getInfo(); sol=h.getSolution()
    names={highspy.HighsModelStatus.kOptimal:'optimal',highspy.HighsModelStatus.kInfeasible:'infeasible',
           highspy.HighsModelStatus.kUnbounded:'unbounded',highspy.HighsModelStatus.kTimeLimit:'time_limit'}
    valid=bool(sol.value_valid)
    result={'status':names.get(st,h.modelStatusToString(st)), 'time_s':elapsed,
            'objective':float(h.getObjectiveValue()) if valid else None,
            'x':np.asarray(sol.col_value) if valid else None,
            'row_dual':np.asarray(sol.row_dual) if sol.dual_valid else None,
            'col_dual':np.asarray(sol.col_dual) if sol.dual_valid else None,
            'simplex_iterations':int(info.simplex_iteration_count),
            'ipm_iterations':int(info.ipm_iteration_count),
            'crossover_iterations':int(info.crossover_iteration_count),
            'mip_nodes':int(info.mip_node_count),
            'mip_gap':float(info.mip_gap), 'mip_dual_bound':float(info.mip_dual_bound),
            'rows':A.shape[0], 'columns':len(c),'nonzeros':A.nnz}
    h.clear()
    return result

def environment():
    return {'python':platform.python_version(),'platform':platform.platform(),'highs':highspy.Highs().version(),
            'packages':{n:importlib.metadata.version(n) for n in ('numpy','scipy','highspy','matplotlib')},
            'seed':SEED,'group':527,'N':12,'Q':2,'Y':26,'variant':1+(17*12+7*2+26)%20}
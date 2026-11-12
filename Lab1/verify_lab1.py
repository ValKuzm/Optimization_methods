"""Независимая проверка ЛР № 1 и бенчмарка."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
from common import ROOT, load_json, save_json, TOL
import lab1

def main():
    checks = []
    def require(name, condition):
        checks.append({'check': name, 'passed': bool(condition)})
        if not condition:
            raise AssertionError(name)

    a = load_json(ROOT / 'results/lab1/results.json')
    ad = load_json(ROOT / 'results/lab1/scenario_data.json')

    for scenario in ('base', 'deficit', 'stress'):
        s = a[scenario]
        independently = lab1.check(ad[scenario], s['x'], s['annual_return'])
        require(f'lab1_{scenario}', independently['passed'] and abs(s['profit_rub'] - independently['profit_rub']) <= TOL)

    x = np.asarray(a['base']['x'])
    bad = x.copy(); bad[0] += .01
    require('lab1_reject_bad_budget', not lab1.check(ad['base'], bad, a['base']['annual_return'])['passed'])
    require('lab1_reject_bad_objective', not lab1.check(ad['base'], x, a['base']['annual_return'] + .001)['passed'])
    require('lab1_reject_bad_risk', not lab1.check(ad['base'], [0, 0, 0, 1, 0, 0], ad['base']['assets'][3]['annual_return'])['passed'])

    r, A, b = lab1.model(ad['base'])
    y = np.asarray(a['dual']['x'][:3]); z = a['dual']['x'][3]
    require('dual_bound', np.min(A.T @ y + z - r) >= -TOL and np.min(y) >= -TOL and abs(b @ y + z - r @ x) <= TOL)

    bench_path = ROOT / 'results/benchmark/results.json'
    if bench_path.exists():
        bench = load_json(bench_path)
        require('benchmark_10_tasks', len({row['task'] for row in bench['raw']}) == 10)
        require('benchmark_100_verified_runs', len(bench['raw']) == 100 and max(row['max_violation'] for row in bench['raw']) <= TOL)

    summary = {'passed': True, 'count': len(checks), 'checks': checks}
    save_json(ROOT / 'results/verification_summary.json', summary)
    print(f'{len(checks)} checks passed for Lab 1!')

if __name__ == '__main__':
    main()
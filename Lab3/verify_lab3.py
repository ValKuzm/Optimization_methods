"""Независимая проверка ЛР № 3 (MILP) на NumPy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
from src.common import ROOT, load_json, save_json, TOL
from src import lab3

def main():
    checks = []
    def require(name, condition):
        checks.append({'check': name, 'passed': bool(condition)})
        if not condition:
            raise AssertionError(name)

    c = load_json(ROOT / 'results/lab3/results.json')
    cd = load_json(ROOT / 'results/lab3/scenario_data.json')

    for scenario in ('base', 'deficit'):
        s = c[scenario]
        require(f'lab3_{scenario}', lab3.check(cd[scenario], s['assignments'], s['activated'], s['overtime'], s['objective'])['passed'])

    base = c['base']
    xx = np.asarray(base['assignments']); yy = np.asarray(base['activated']); oo = np.asarray(base['overtime'])
    require('lab3_reject_bad_activation', not lab3.check(cd['base'], xx, np.zeros(10), oo, base['objective'])['passed'])
    require('lab3_reject_missing_coverage', not lab3.check(cd['base'], np.zeros_like(xx), yy, oo, base['objective'])['passed'])
    require('lab3_reject_fractional', not lab3.check(cd['base'], xx * .9, yy, oo, base['objective'])['passed'])

    sc = c['stress']['infeasibility_certificate']
    require('lab3_stress_certificate', sc['required_shifts'] > sc['maximum_assignments'])

    for n in (10, 20, 40):
        scale = load_json(ROOT / f'results/lab3/scale_{n}.json')
        sd = load_json(ROOT / f'data/lab3_scale_{n}.json')
        ss = scale['mip']
        require(f'scaling_{n}_feasible', lab3.check(sd, ss['assignments'], ss['activated'], ss['overtime'], ss['objective'])['passed'])

    summary = {'passed': True, 'count': len(checks), 'checks': checks}
    save_json(ROOT / 'results/verification_summary.json', summary)
    print(f'{len(checks)} checks passed for Lab 3!')

if __name__ == '__main__':
    main()
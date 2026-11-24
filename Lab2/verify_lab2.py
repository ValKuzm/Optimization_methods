"""Независимая проверка ЛР № 2 на NumPy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
from common import ROOT, load_json, save_json, TOL
import lab2

def main():
    checks = []
    def require(name, condition):
        checks.append({'check': name, 'passed': bool(condition)})
        if not condition:
            raise AssertionError(name)

    b2 = load_json(ROOT / 'results/lab2/results.json')
    bd = load_json(ROOT / 'results/lab2/scenario_data.json')

    for scenario in ('base', 'growth_20'):
        s = b2[scenario]
        require(f'lab2_{scenario}', lab2.check(bd[scenario], s['flows'], s['objective'])['passed'])

    f = np.asarray(b2['base']['flows'])
    bad = f.copy(); bad[0, 0] += 1
    require('lab2_reject_bad_flow', not lab2.check(bd['base'], bad, b2['base']['objective'])['passed'])
    require('lab2_failure_certificate', lab2.cut_certificate(bd['arc_failure'])['shortage'] > 0)
    require('lab2_shortest_distance_crosscheck', b2['shortest']['floyd_max_difference'] <= TOL)

    summary = {'passed': True, 'count': len(checks), 'checks': checks}
    save_json(ROOT / 'results/verification_summary.json', summary)
    print(f'{len(checks)} checks passed for Lab 2!')

if __name__ == '__main__':
    main()
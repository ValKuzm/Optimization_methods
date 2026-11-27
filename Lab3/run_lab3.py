"""Запуск ЛР № 3 (MILP-модель расписания смен) одной командой."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.common import ROOT, save_json, environment
from src import lab3, figures, reports

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--regenerate', action='store_true', help='пересоздать data/lab3.json по seed 10212')
    args = p.parse_args()

    target = ROOT / 'data/lab3.json'
    if args.regenerate or not target.exists():
        save_json(target, lab3.generate())

    save_json(ROOT / 'results/environment.json', environment())
    print('Running Lab 3 (MILP Call-Center Scheduling)...', flush=True)
    lab3.run()
    print('Generating figures and summary report...', flush=True)
    figures.run()
    reports.run()
    print('Lab 3 finished successfully!', flush=True)

if __name__ == '__main__':
    main()
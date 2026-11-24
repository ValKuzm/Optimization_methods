"""Запуск ЛР № 2 (Транспортные и сетевые модели) одной командой."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from common import ROOT, save_json, environment
import lab2, figures, reports

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--regenerate', action='store_true', help='пересоздать data/lab2.json по seed 10212')
    args = p.parse_args()

    target = ROOT / 'data/lab2.json'
    if args.regenerate or not target.exists():
        save_json(target, lab2.generate())

    save_json(ROOT / 'results/environment.json', environment())
    print('Running Lab 2 (Network Shortest Paths & Multi-commodity Flow)...', flush=True)
    lab2.run()
    print('Generating figures and summary report...', flush=True)
    figures.run()
    reports.run()
    print('Lab 2 finished successfully!', flush=True)

if __name__ == '__main__':
    main()
"""Запуск ЛР № 1 (Портфель) + два задания со звёздочкой."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from common import ROOT, save_json, environment
import lab1, benchmark, figures, report

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--regenerate', action='store_true', help='пересоздать data/lab1.json по seed 10212')
    p.add_argument('--skip-benchmark', action='store_true', help='пропустить долгий бенчмарк 10 задач')
    args = p.parse_args()

    target = ROOT / 'data/lab1.json'
    if args.regenerate or not target.exists():
        save_json(target, lab1.generate())

    save_json(ROOT / 'results/environment.json', environment())
    
    print('Running Lab 1 (Portfolio & Dual)...', flush=True)
    lab1.run()
    
    if not args.skip_benchmark:
        print('Running Star Task 2 (Simplex vs IPX Benchmark)...', flush=True)
        benchmark.run()
        
    print('Generating figures and reports...', flush=True)
    figures.run()
    report.run()
    print('Lab 1 finished successfully!', flush=True)

if __name__ == '__main__':
    main()
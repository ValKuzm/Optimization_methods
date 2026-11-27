"""Краткая машинная сводка для ЛР 3."""
from common import ROOT, load_json

def write(path,text):
    p=ROOT/path; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(text.strip()+'\n',encoding='utf-8')

def run():
    c=load_json(ROOT/'results/lab3/results.json')
    write('results/lab3/REPORT.md',f'''
# Лабораторная работа 3. MILP-модель дискретного выбора (Расписание смен)

Вариант 5, группа 527, N=12, seed=10212. 10 операторов, 7 дней, 3 смены по 8 часов.
Данные синтетические; доступность, цены и спрос — data/lab3.json.

## Результаты

- Базовый оптимум (HiGHS MILP): `{c['base']['objective']:.6f}` ₽ за неделю, итоговый MIP gap = 0%.
- LP-релаксация: `{c['relaxation']['objective']:.6f}` ₽, интегральный зазор `{c['integrality_gap_percent']:.6f}%`.
- Влияние константы big-M: при M=5 нижняя граница LP равна `{c['relaxation']['objective']:.2f}` ₽, при M=10000 падает до `{c['loose_M']['relaxation']['objective']:.2f}` ₽ (зазор `{c['loose_M']['relaxation_gap_percent']:.2f}%`).
- Дефицитный сценарий (-20% персонала): status `optimal`, стоимость `{c['deficit']['objective']:.6f}` ₽.
- Стрессовый сценарий: status `infeasible` (требуется 42 выхода при максимуме 8*5=40).
- Жадная эвристика (30 запусков): стоимость `{c['greedy']['objective']:.6f}` ₽, проигрыш оптимуму `{c['greedy']['gap_percent']:.6f}%`.
''')

if __name__=='__main__': run()
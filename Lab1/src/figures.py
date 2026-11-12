"""Графики для ЛР 1 (портфель) и задания со звёздочкой (бенчмарк)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, load_json

NAVY='#20374F'; TEAL='#087F8C'; GOLD='#D89B30'; CORAL='#CB654D'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
 'axes.spines.right':False,'axes.labelcolor':NAVY,'text.color':NAVY,'axes.edgecolor':'#BCC8D2',
 'axes.titleweight':'bold','axes.titlepad':14,'savefig.facecolor':'white','figure.facecolor':'white'})
OUT=ROOT/'results/figures'

def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True); fig.savefig(OUT/f'{name}.png',dpi=240,bbox_inches='tight'); plt.close(fig)

def run():
    a=load_json(ROOT/'results/lab1/results.json')
    bench=load_json(ROOT/'results/benchmark/results.json')
    
    # 1. Структура портфеля по сценариям
    fig,ax=plt.subplots(figsize=(7.4,4.5)); positions=np.arange(6); width=.24
    for j,(name,label,color) in enumerate([('base','Базовый',TEAL),('deficit','Дефицитный',GOLD),('stress','Стрессовый',CORAL)]):
        ax.bar(positions+(j-1)*width,np.array(a[name]['x'])*100,width,label=label,color=color)
    ax.set(xticks=positions,xticklabels=[f'A{i+1}' for i in positions],ylabel='Доля бюджета, %',ylim=(0,55))
    ax.legend(frameon=False,ncol=3,loc='upper center'); ax.grid(axis='y',alpha=.15); fig.tight_layout(); save(fig,'lab1_allocation')
    
    # 2. Теневые цены ограничений риска
    fig,ax=plt.subplots(figsize=(7.4,4.1)); sh=np.array(a['base']['shadow_prices'])*1000
    bars=ax.barh(['Рыночный риск','Валютная доля','Неликвидность'],sh,color=[TEAL,GOLD,CORAL])
    for bar,value in zip(bars,sh): ax.text(value+.7,bar.get_y()+bar.get_height()/2,f'{value:.2f} ₽',va='center')
    ax.set(xlabel='Дополнительная годовая прибыль при увеличении лимита на 0,01',xlim=(0,max(sh)*1.28))
    ax.invert_yaxis(); ax.grid(axis='x',alpha=.15); fig.tight_layout(); save(fig,'lab1_shadows')
    
    # 3. Бенчмарк Симплекс vs IPX
    fig,axes=plt.subplots(1,2,figsize=(8.3,4))
    for method,color,label in [('simplex',TEAL,'Симплекс'),('ipx',CORAL,'Внутренняя точка IPX')]:
        rr=[r for r in bench['rows'] if r['method']==method]; xx=[r['variables'] for r in rr]
        axes[0].loglog(xx,[r['median_time_s'] for r in rr],'o-',color=color,label=label)
        axes[0].fill_between(xx,[r['min_time_s'] for r in rr],[r['max_time_s'] for r in rr],color=color,alpha=.12)
        axes[1].semilogx(xx,[r['simplex_iterations'] if method=='simplex' else r['ipm_iterations'] for r in rr],'o-',color=color,label=label)
    axes[0].set(xlabel='Переменных',ylabel='Медианное время, с'); axes[1].set(xlabel='Переменных',ylabel='Основных итераций')
    for ax in axes: ax.grid(alpha=.18); ax.legend(frameon=False,fontsize=8)
    fig.tight_layout(); save(fig,'benchmark')
    print('3 figures for Lab 1 generated')

if __name__=='__main__': run()
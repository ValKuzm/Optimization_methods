"""Графики для ЛР 3 (MILP-модель расписания смен операторов)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from common import ROOT, load_json

NAVY='#20374F'; TEAL='#087F8C'; GOLD='#D89B30'; CORAL='#CB654D'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
 'axes.spines.right':False,'axes.labelcolor':NAVY,'text.color':NAVY,'axes.edgecolor':'#BCC8D2',
 'axes.titleweight':'bold','axes.titlepad':14,'savefig.facecolor':'white','figure.facecolor':'white'})
OUT=ROOT/'results/figures'

def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True); fig.savefig(OUT/f'{name}.png',dpi=240,bbox_inches='tight'); plt.close(fig)

def run():
    c=load_json(ROOT/'results/lab3/results.json')

    for scenario in ('base','deficit'):
        x=np.asarray(c[scenario]['assignments']); arr=np.zeros((10,7),dtype=int)
        for i in range(10):
            for day in range(7):
                inds=np.flatnonzero(x[i,day]>.5)
                if len(inds): arr[i,day]=inds[0]+1
        fig,ax=plt.subplots(figsize=(8.1,5.3)); ax.imshow(arr,cmap=ListedColormap(['#EEF2F5','#7564A5','#E7B35A','#3699A0']),vmin=0,vmax=3,aspect='auto')
        for i in range(10):
            for day in range(7): ax.text(day,i,['-','Н','У','В'][arr[i,day]],ha='center',va='center',fontsize=13,color='white' if arr[i,day] in (1,3) else NAVY)
        ax.set(xticks=range(7),yticks=range(10),xticklabels=['Пн','Вт','Ср','Чт','Пт','Сб','Вс'],yticklabels=[f'Оператор {i+1:02d}' for i in range(10)])
        ax.set_xticks(np.arange(-.5,7,1),minor=True); ax.set_yticks(np.arange(-.5,10,1),minor=True); ax.grid(which='minor',color='white',lw=2)
        ax.tick_params(which='minor',bottom=False,left=False); ax.set_title('Н: 00-08  ·  У: 08-16  ·  В: 16-24  ·  -: выходной',fontsize=11)
        for spine in ax.spines.values(): spine.set_visible(False)
        fig.tight_layout(); save(fig,f'lab3_schedule_{scenario}')

    fig,ax=plt.subplots(figsize=(7.5,4.2)); labels=['LP, слишком большое M','LP, обоснованное M = 5','Целочисленный оптимум','Жадная эвристика']
    vals=[c['loose_M']['relaxation']['objective'],c['relaxation']['objective'],c['base']['objective'],c['greedy']['objective']]
    bars=ax.barh(labels,vals,color=['#A6B7C5',GOLD,TEAL,CORAL])
    for bar,value in zip(bars,vals): ax.text(value+300,bar.get_y()+bar.get_height()/2,f'{value:,.0f}'.replace(',',' '),va='center')
    ax.set(xlabel='Стоимость недели, ₽',xlim=(0,max(vals)*1.18)); ax.invert_yaxis(); ax.grid(axis='x',alpha=.15)
    fig.tight_layout(); save(fig,'lab3_bounds')

    scale=c['scaling']; fig,axes=plt.subplots(1,2,figsize=(8,3.7))
    axes[0].plot([v['operators'] for v in scale],[v['time_s'] for v in scale],'o-',color=TEAL)
    axes[0].set(xlabel='Операторов',ylabel='Время решения, с')
    axes[1].plot([v['operators'] for v in scale],[v['relaxation_gap_percent'] for v in scale],'o-',color=GOLD,label='Зазор LP-релаксации')
    axes[1].plot([v['operators'] for v in scale],[v['solver_gap_percent'] for v in scale],'o--',color=TEAL,label='Итоговый MIP gap')
    axes[1].set(xlabel='Операторов',ylabel='Зазор, %'); axes[1].legend(frameon=False,fontsize=8)
    for ax in axes: ax.grid(alpha=.18); ax.set_xticks([10,20,40])
    fig.tight_layout(); save(fig,'lab3_scaling')
    print('4 figures for Lab 3 generated')

if __name__=='__main__': run()
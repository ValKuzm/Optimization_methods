"""Графики для ЛР 2 (Кратчайшие пути и потоки в сети)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from common import ROOT, load_json

NAVY='#20374F'; TEAL='#087F8C'; CORAL='#CB654D'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
 'axes.spines.right':False,'axes.labelcolor':NAVY,'text.color':NAVY,'axes.edgecolor':'#BCC8D2',
 'axes.titleweight':'bold','axes.titlepad':14,'savefig.facecolor':'white','figure.facecolor':'white'})
OUT=ROOT/'results/figures'

def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True); fig.savefig(OUT/f'{name}.png',dpi=240,bbox_inches='tight'); plt.close(fig)

def run():
    b=load_json(ROOT/'results/lab2/results.json')
    d2=load_json(ROOT/'data/lab2.json')

    # 1. Тепловая карта кратчайших расстояний (Этап А)
    fig,ax=plt.subplots(figsize=(7.5,6)); im=ax.imshow(b['distances'],cmap='YlGnBu')
    for i in range(10):
        for j in range(10):
            val=b['distances'][i][j]; ax.text(j,i,f'{val:.0f}',ha='center',va='center',fontsize=10,color='white' if val>20 else NAVY)
    ax.set(xticks=range(10),yticks=range(10),xticklabels=range(1,11),yticklabels=range(1,11),xlabel='Куда',ylabel='Откуда')
    fig.colorbar(im,ax=ax,label='Минимальная стоимость единицы груза'); fig.tight_layout(); save(fig,'lab2_distances')

    # 2. Граф сети с потоками и пропускными способностями (Этап B)
    fig,ax=plt.subplots(figsize=(10.3,6.3))
    pos={0:(-1.3,1),1:(-2.8,1.1),2:(-3.2,-.3),3:(-2.2,-1.3),4:(-.9,-.6),
         5:(.9,-.6),6:(2.2,-1.3),7:(3.2,-.3),8:(2.8,1.1),9:(1.3,1)}
    G=nx.DiGraph(); G.add_nodes_from(range(10)); total=np.array(b['base']['flows']).sum(axis=0)
    for edge in d2['arcs']: G.add_edge(edge['u'],edge['v'])
    nx.draw_networkx_nodes(G,pos,ax=ax,node_color=NAVY,node_size=1000)
    nx.draw_networkx_labels(G,pos,ax=ax,labels={i:str(i+1) for i in range(10)},font_color='white',font_size=16)
    for j,edge in enumerate(d2['arcs']):
        u,v=edge['u'],edge['v']; active=abs(total[j]-edge['capacity'])<1e-7
        nx.draw_networkx_edges(G,pos,ax=ax,edgelist=[(u,v)],connectionstyle='arc3,rad=0.17',
          edge_color=CORAL if active else '#91A6B7',width=1.2+2*total[j]/max(total),arrowsize=15,node_size=1000)
        x0,y0=pos[u]; x1,y1=pos[v]; xx=(x0+x1)/2+.15*(y1-y0); yy=(y0+y1)/2-.15*(x1-x0)
        if (u,v)==(5,7): xx-=.15; yy+=.20
        label=f"{total[j]:g}/{edge['capacity']:g}"
        ax.text(xx,yy,label,ha='center',va='center',fontsize=10.5,color=CORAL if active else NAVY,
                bbox={'facecolor':'white','edgecolor':'none','pad':1,'alpha':.94})
    ax.text(0,1.7,'Поток / пропускная способность',ha='center',fontsize=13)
    ax.text(0,-1.9,'Оранжевые дуги насыщены. Узлы 1-5 и 6-10 связаны двумя коридорами.',ha='center',fontsize=11)
    ax.set(xlim=(-3.8,3.8),ylim=(-2.1,1.95)); ax.axis('off'); fig.tight_layout(); save(fig,'lab2_network')

    # 3. Загрузка межгрупповых коридоров
    fig,ax=plt.subplots(figsize=(7.5,4.2)); arcs=d2['arcs'][-4:]; posi=np.arange(4)
    ax.bar(posi-.2,[v['capacity'] for v in arcs],.4,color='#CBD6DE',label='Ёмкость')
    ax.bar(posi+.2,total[-4:],.4,color=TEAL,label='Базовый поток')
    for j,(edge,value) in enumerate(zip(arcs,total[-4:])):
        ax.text(j-.2,edge['capacity']+.5,f"{edge['capacity']:g}",ha='center',fontsize=10)
        ax.text(j+.2,value+.5,f'{value:g}',ha='center',fontsize=10)
    ax.set(xticks=posi,xticklabels=[f"{x['id']}\n{x['u']+1} → {x['v']+1}" for x in arcs],ylabel='Единиц груза в час',ylim=(0,65))
    ax.legend(frameon=False,ncol=2); ax.grid(axis='y',alpha=.15); fig.tight_layout(); save(fig,'lab2_corridors')
    print('3 figures for Lab 2 generated')

if __name__=='__main__': run()
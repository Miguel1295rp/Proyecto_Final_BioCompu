import requests as requests
import json as json
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
from google.colab import files

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os

import pandas as pd

%matplotlib inline
from google.colab import files
import networkx as nx

import bokeh.plotting as bpl
import bokeh.palettes as bpt



uploaded = files.upload()

datos = pd.read_csv('Fenotipo_Sano.txt',
            sep = '\t',
            index_col = 0)
datos

#Calcular correlación entre especies
correlaciones = datos.T.corr()
correlaciones

pos = plt.imshow(correlaciones, cmap = 'hot')
plt.colorbar(pos)

#Definir enlaces positivos y enlaces negativos
enlaces = correlaciones[(correlaciones>0.5)&(correlaciones !=1)].fillna(0)
antienlaces = -correlaciones[correlaciones<-0.5].fillna(0)

#Mostrar y descargar mapa de calor de los enlaces positivos
plt.imshow(enlaces, cmap = 'hot')
plt.savefig("Enlaces.png", dpi= 300)
files.download("Enlaces.png")

plt.plot()


#Mostrar y descargar mapa de calor de los enlaces positivos
plt.imshow(antienlaces, cmap = 'hot')
plt.savefig("Enlaces negativos.png", dpi= 300)
files.download("Enlaces negativos.png") 

plt.plot()


#Crear redes
G = nx.from_numpy_array(enlaces.to_numpy())
G = nx.relabel_nodes(G, {i:j for i,j in enumerate(list(enlaces.index))})

Ga = nx.from_numpy_array(antienlaces.to_numpy())
Ga = nx.relabel_nodes(Ga, {i:j for i,j in enumerate(list(antienlaces.index))})

G.edges(data = True)

plt.figure(figsize = (100,100))
pos = nx.spring_layout(G)


#Red de enlaces positivos
nx.draw_networkx_edges(Ga, node_size = 1000, pos = pos, edge_color = 'red', alpha = .4)
nx.draw(G, node_size = 1000, font_color= 'k', edge_color = 'blue')
plt.savefig("Red.png", dpi= 100)
files.download("Red.png") ### Comentar esta linea si no estás trabajando en colaboratory

# Crear listas con el grado de cada uno de los nodos, para cada red
list(G.degree) #enlaces positivos
list(Ga.degree) #enlaces negativos


#Características de las redes 
n = len(G)
m = len(G.edges)

k = 2*m/n

print('Número de nodos:\t n =', n, '\nNúmero de enlaces:\t m = ', m, '\nGrado promedio:\t\t <k> = ', round(k,2))

n2 = len(Ga)
m2= len(Ga.edges)
K2= 2*m2/n
print('Número de nodos:\t n =', n, '\nNúmero de antienlaces:\t m = ', m2, '\nGrado promedio:\t\t <k> = ', round(K2,2))


#Graficar distribución de grado de los enlaces positivos
grados = np.array(list(dict(G.degree()).values()))
log_bin = np.logspace(0, np.log10(max(grados)), 25)


y, x = np.histogram(grados, bins = log_bin, density = True)
x_med = 0.5*(x[:-1]+x[1:])



plt.plot(x_med, y, 'o')

plt.yscale('log')
plt.xscale('log')

plt.title('Distribución de grado\n Fenotipo Sano')
plt.xlabel('k (grado)')
plt.ylabel('Pk (distribución de grado)')
plt.plot()

plt.savefig("Distribución de Grado.png", dpi= 100)
files.download("Distribución de Grado.png") 

plt.plot()


#Graficar distribución de grado de los enlaces negativos
grados2 = np.array(list(dict(Ga.degree()).values()))
log_bin = np.logspace(0, np.log10(max(grados2)), 25)


y, x = np.histogram(grados2, bins = log_bin, density = True)
x_med = 0.5*(x[:-1]+x[1:])



plt.plot(x_med, y, 'o')

plt.yscale('log')
plt.xscale('log')

plt.title('Distribución de grado\n Fenotipo Sano')
plt.xlabel('k (grado)')
plt.ylabel('Pk (distribución de grado)')
plt.plot()

plt.savefig("Grado anti.png", dpi= 100)
files.download("Grado anti.png") 

plt.plot()


#Calcular componentes de la Red
componentes = [c for c in nx.connected_components(G)]
max_comp = max(componentes, key = len)
G_max = G.subgraph( max_comp )
pos = nx.drawing.layout.spring_layout(G_max)
nx.draw(G_max, pos = pos, node_size = 100)

from networkx.algorithms.community import greedy_modularity_communities
c = list(greedy_modularity_communities(G_max))
#sorted(c[0])

len(c)

colores = bpt.Category10[len(c)]


#Red de componentes

nx.draw(G_max, pos = pos, node_size = 0)

for i,cp in enumerate(c):
  nx.draw_networkx_nodes(G_max, pos = pos, nodelist = cp, node_size = 300, node_color = colores[i])

plt.savefig("Componentes.png", dpi= 100)
files.download("Componentes.png")

plt.plot()


#Componentes de la Red con enlaces negativos
componentes = [c1 for c1 in nx.connected_components(Ga)]
max_comp = max(componentes, key = len)
Ga_max = Ga.subgraph( max_comp )
pos = nx.drawing.layout.spring_layout(Ga_max)
nx.draw(Ga_max, pos = pos, node_size = 100)

from networkx.algorithms.community import greedy_modularity_communities
c1 = list(greedy_modularity_communities(Ga_max))
#sorted(c[0])

len(c1)

colores = bpt.Category10[len(c1)]

nx.draw(Ga_max, pos = pos, node_size = 0)

for i,cp in enumerate(c1):
  nx.draw_networkx_nodes(Ga_max, pos = pos, nodelist = cp, node_size = 300, node_color = colores[i])

plt.savefig("Componentes negativos.png", dpi= 100)
files.download("Componentes negativos.png") 

plt.plot()



#Medidas de Centralidad

#Clustering
nx.average_clustering(G)
nx.average_clustering(Ga)


etiquetas=dict(G.degree)

nx.draw(G,node_color="yellow",Node_size=1000,labels=etiquetas)


#Calcular Betweenness y mostrar Red
Tamaño=np.array(list(nx.betweenness_centrality(G).values()))
Tamaño

nx.draw(G,node_color="y",node_size=4000*Tamaño)#El tamaño de cada nodo se multiplica por su valor de betwenness
plt.savefig("Betweenneess.png", dpi= 100)
files.download("Betweenneess.png") 

plt.plot()



etiquetas2=dict(Ga.degree)
nx.draw(Ga,node_color="green",Node_size=1000,labels=etiquetas2)

Tamaño2=np.array(list(nx.betweenness_centrality(Ga).values()))
Tamaño2

nx.draw(Ga,node_color="y",node_size=4000*Tamaño2,labels=etiquetas2)


#Calcular Closeness y mostrar red
colores=np.array(list(nx.closeness_centrality(G).values()))
colores

nx.draw(G,node_color=colores,node_size=10000*Tamaño)
plt.savefig("Closesness.png", dpi= 100)
files.download("Closesness.png") 

plt.plot()
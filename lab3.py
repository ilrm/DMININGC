import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
import matplotlib as plt


iris=load_iris()
filename = 'iris.txt'
datar=[]
with open(filename,'r') as file_to_read:
    while (True):
        datat=[]
        lines = file_to_read.readline()
        if not lines:
            break
        att1, att2, att3, att4,\
        temp = [i for i in lines.split(",")]
        datat.append(float(att1))
        datat.append(float(att2))
        datat.append(float(att3))
        datat.append(float(att4))
        datar.append(datat)
D=np.array(datar)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(D, iris.target)
with open("tree.dot",'w')as file:
    file=tree.export_graphviz(clf,out_file=file)

import pydotplus
dota_data=tree.export_graphviz(clf,out_file=None)

graph = pydotplus.graph_from_dot_data(dota_data)

print(graph)  
graph.write_pdf("tree.pdf")

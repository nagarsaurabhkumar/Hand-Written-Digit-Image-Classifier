import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_digits
dataset = load_digits()

X = dataset.data
y = dataset.target

some_digit = X[1222]
some_digit_image = some_digit.reshape(8,8)

plt.imshow(some_digit_image)
plt.show()

from sklearn.tree import DecisionTreeClassifier
dts = DecisionTreeClassifier(max_depth = 13)

dts = dts.fit(X,y)
dts.score(X,y)

dts.predict(X[[1222],0:64])

from sklearn.tree import export_graphviz
export_graphviz(dts,out_file = 'tree.dot')


import graphviz
with open('tree.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

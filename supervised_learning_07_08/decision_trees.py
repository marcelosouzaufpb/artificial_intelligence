from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

import matplotlib.pyplot as plt
import numpy as np
import graphviz

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

# 1 - Visualizando Arvores
model = DecisionTreeClassifier(max_depth=6)
model = model.fit(x, y)
tree.plot_tree(model)
figura, ax = plt.subplots(figsize=(24, 12))
tree.plot_tree(model, max_depth=4, fontsize=10)
plt.show()

# 1.1 - Usando graphviz (alterando estilo)
dot_data = tree.export_graphviz(model, out_file=None,
                                feature_names=datasets.feature_names,
                                class_names=datasets.target_names,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph

# Plotando uma Learning Curve

# Plotando uma Validation Curve
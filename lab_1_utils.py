from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def print_decorated(description, obj):
    print("\n#######################################################################################")
    print(description, ": ", obj)


def get_iris_dataset():
    iris = load_iris()
    # conversione del dataset in dataframe panda (facilita operazioni sui dati)
    iris = pd.DataFrame(
        data=np.c_[iris['data'], iris['target']],
        columns=iris['feature_names'] + ['target']
    )

    # print(iris.head(10)) # pandas per vedere le prime 10 righe della matrice
    # classi = target: SETOSA = 0, VERSICOLOR = 1, VIRGINICA = 2
    # aggiunta della colonna "species" in cui mettiamo la classe in formato stringa
    species = []
    for i in range(len(iris['target'])):
        if iris['target'][i] == 0:
            species.append("setosa")
        elif iris['target'][i] == 1:
            species.append('versicolor')
        else:
            species.append('virginica')

    iris['species'] = species
    # print(iris.head(10))
    print_decorated("IRIS DATASET LOADED\nExamples", iris.groupby('species').size())
    # species: setosa = 50, versicolor = 50, virganica = 50
    return iris


def plotting_dataset_in_graph(iris):
    # plotting del dataset in un grafico
    setosa = iris[iris.species == "setosa"]
    versicolor = iris[iris.species == 'versicolor']
    virginica = iris[iris.species == 'virginica']
    fig, ax = plt.subplots()
    fig.set_size_inches(13, 7)  # adjusting the length and width of plot
    ax.scatter(setosa['petal length (cm)'], setosa['petal width (cm)'], label="Setosa", facecolor="blue")
    ax.scatter(versicolor['petal length (cm)'], versicolor['petal width (cm)'], label="Versicolor",
               facecolor="green")
    ax.scatter(virginica['petal length (cm)'], virginica['petal width (cm)'], label="Virginica", facecolor="red")
    ax.set_xlabel("petal length (cm)")
    ax.set_ylabel("petal width (cm)")
    ax.grid()
    ax.set_title("Iris petals")
    ax.legend()
    fig.show()


def get_data_and_target_arrays(iris):
    # split dataset in training e test set
    # X = dati senza l'etichetta di classe
    # Y = etichette di classe
    X = iris.drop(['target', 'species'], axis=1)
    Y = iris['target']
    return X, Y

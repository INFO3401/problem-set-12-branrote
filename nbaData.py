import pandas as pd
import seaborn as sns

from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

def loadData(datafile):
        with open(datafile, 'r', encoding='latin1') as csvfile:
            data = pd.read_csv(csvfile)

        print(data.columns.values)

        return data

def runKNN(dataset, prediction, ignore):
    X = dataset.drop(columns=[prediction, ignore])
    Y = dataset[prediction].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)

    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train, Y_train)

    score = knn.score(X_test, Y_test)
    print("Predicts " + prediction + "with" + str(score) + " accuracy")
    print("Chance is: " + str(1.0/len(dataset.groupby(prediction))))

    return knn

def classifyPlayer(targetRow, data, model, prediction, ignore):
    X = targetRow.drop(columns=[prediction, ignore])

    neighbors = model.kneighbors(X, n_neighbors=5, return_distance=False)

    for neighbor in neighbors[0]:
        print(data.iloc[neighbor])

def runKMeans(dataset, ignore):
    X = dataset.drop(columns=ignore)

    kmeans = KMeans(n_clusters=5)

    kmeans.fit(X)

    dataset['cluster'] = pd.Series(kmeans.predict(X), index=dataset.index)

    scatterMatrix = sns.pairplot(dataset.drop(columns=ignore), hue='cluster', palette='Set2')

    scatterMatrix.savefig("kmeanClusters.png")

    return kmeans

nbaData = loadData("nba_2013_clean.csv")
knnModel = runKNN(nbaData, "pos", "player")
classifyPlayer(nbaData.loc[nbaData['player']=='LeBron James'], nbaData, knnModel, 'pos', 'player')

kmeansmodel = runKMeans(nbaData, ['pos','player'])

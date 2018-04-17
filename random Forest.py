import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn


def main():
    """Loading CSV file with all the attributes"""
    data1=pd.read_csv("C:\Users\Aayush Goyal\Desktop\covtype.csv")
    headers=["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology",
             "Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points",
             "Wilderness_Area1","Wilderness_Area2","Wilderness_Area3","Wilderness_Area4","Soil_Type1","Soil_Type2","Soil_Type3","Soil_Type4","Soil_Type5","Soil_Type6","Soil_Type7","Soil_Type8","Soil_Type9",
              "Soil_Type10","Soil_Type11","Soil_Type12","Soil_Type13","Soil_Type14","Soil_Type15","Soil_Type16","Soil_Type17","Soil_Type18","Soil_Type19","Soil_Type20","Soil_Type21","Soil_Type22","Soil_Type23",
              "Soil_Type24","Soil_Type25","Soil_Type26","Soil_Type27","Soil_Type28","Soil_Type29","Soil_Type30","Soil_Type31","Soil_Type32","Soil_Type33","Soil_Type34","Soil_Type35","Cover_Type"]
    train_x,test_x,train_y,test_y=split_data(data1,0.90,headers[1:-1],headers[-1])
    training_model=random_forest(train_x,train_y)
    print "trained model",training_model
    predictions=training_model.predict(test_x)

    """Train/Test Accuracy"""
    print "Train Accuracy ::", accuracy_score(train_y,training_model.predict(train_x))
    print "Test Accuracy ::",accuracy_score(test_y,predictions)
    print "Confusion Matrix"
    print confusion_matrix(test_y,predictions)

    """Determining importance variables """
    A, b = make_classification(n_samples=1000,
                               n_features=53,
                               n_informative=4,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=2,
                               random_state=0,
                               shuffle=False)
    forest = ExtraTreesClassifier(n_estimators=53,
                                  random_state=0)
    forest.fit(A,b)
    imp = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
    indices = np.argsort(imp)[::-1]

    print("Feature ranking:")
    """Print feature ranking"""

    """Pair plots using seaborn"""
    #Target variable Classes 1 to 7
    data1["Cover_Type"][data1["Cover_Type"]==1] = "one"
    data1["Cover_Type"][data1["Cover_Type"]==2] = "two"
    data1["Cover_Type"][data1["Cover_Type"]==3] = "three"
    data1["Cover_Type"][data1["Cover_Type"]==4] = "four"
    data1["Cover_Type"][data1["Cover_Type"]==5] = "five"
    data1["Cover_Type"][data1["Cover_Type"]==6] = "six"
    data1["Cover_Type"][data1["Cover_Type"]==7] = "seven"

    #Predictor Variables
    data1["Elevation"] = data1["Elevation"]
    data1["Aspect"] = data1["Aspect"]
    data1["Slope"] = data1["Slope"]
    data1["Horizontal_Distance_To_Hydrology"] = data1["Horizontal_Distance_To_Hydrology"]
    data1["Vertical_Distance_To_Hydrology"] = data1["Vertical_Distance_To_Hydrology"]
    data1["Horizontal_Distance_To_Roadways"] = data1["Horizontal_Distance_To_Roadways"]

    plt.figure()
    seaborn.pairplot(data=data1[["Elevation","Slope","Aspect","Horizontal_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Vertical_Distance_To_Hydrology","Cover_Type"]],hue="Cover_Type",dropna=True)
    plt.savefig("Trial plot")

    for f in range(A.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], imp[indices[f]]))

    """Figuring out important variables for classification"""
    plt.figure()
    plt.title("Feature Importances")
    plt.bar(range(A.shape[1]), imp[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(A.shape[1]), indices)
    plt.xlim([-1, A.shape[1]])
    plt.show()

    """Principal Component Analysis to determine dimensionality could be reduced"""
    # scalar=StandardScaler().fit(train_x)
    # train_x_scaled = pd.DataFrame(scalar.transform(train_x),index=train_x.index.values , columns=train_x.columns.values)
    # test = pd.DataFrame(scalar.transform(test_x),index=test_x.index.values , columns=test_x.columns.values)
    Transform_xtrain = PCA().fit_transform(train_x)
    pca=PCA()
    pca.fit(np.array(Transform_xtrain))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel(' Total Explained variance')
    plt.show()
    # print pca.components_
    # points=pd.DataFrame(PCA.transform(np.array(train_x)))
    # x_axis = np.arange(1,PCA.n_components+1)
    # pcaScaled = PCA()
    # pcaScaled.fit(train_x)
    # points_scaled = pd.DataFrame(PCA.transform(train_x))


def split_data(data,train_percentage,feature_headers,target_header):

    train_x,test_x,train_y,test_y=train_test_split(data[feature_headers],data[target_header],train_size=train_percentage)
    return train_x,test_x,train_y,test_y

def random_forest(features,target):
    """return feature and target here to fit into our classifier"""
    rf=RandomForestClassifier()
    rf.fit(features,target)
    return rf


if __name__=="__main__":
    main()
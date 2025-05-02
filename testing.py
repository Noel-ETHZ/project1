from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.decomposition import KernelPCA, PCA
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from matplotlib import pyplot as plt
from tqdm import tqdm

# sklearn imports...
# SVRs are not allowed in this project.

if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()

    # Load dataset: images (X) and corresponding minimum distance values (y)
    #X = np.load("/Users/noel/Documents/SML Project/project1/images_train.npy")
    #y = np.load("/Users/noel/Documents/SML Project/project1/labels_train.npy")
    X, y = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(X)} samples.")

    images_test = load_test_dataset(config)

    # TODO: Your implementation starts here
    # possible preprocessing steps ... training the model


    params = {
            "model" : "KNN",
            "scaler": "StandardScaler",
            "pca" : "KernelPCA",
            "n_neighbors" : 2,
            "pca_components" : 200,
            "downsample_factor" : config["downsample_factor"],
            "test_size" : 0.15
        }
#     {'model': 'KNN', 'scaler': 'StandardScaler', 'pca': 'KernelPCA', 'n_neighbors': 2, 'pca_components': 200, 'downsample_factor': 3, 'test_size': 0.15}
# MAE: 12.526

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params["test_size"], random_state=42)
    # 1. Normalization

    if params["scaler"] == "MinMaxScaler":
        scaler = preprocessing.MinMaxScaler().fit(X_train)
    elif params["scaler"] == "StandardScaler":
        scaler = preprocessing.StandardScaler().fit(X_train)
    #scaler = preprocessing.MinMaxScaler().fit(X_train)

    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    if params["pca"] == "KernelPCA":
        pca = KernelPCA(n_components=params["pca_components"], kernel="rbf")
    else:
        pca = PCA(n_components=params["pca_components"], whiten=True)
    pca.fit(X_train_s)

    X_train_pp = pca.transform(X_train_s)
    X_test_pp = pca.transform(X_test_s)


    model = KNeighborsRegressor(n_neighbors=params["n_neighbors"], weights="distance", metric="manhattan")  


    
    model.fit(X_train_pp, y_train)
        
    # predict
    X_pred = model.predict(X_test_pp)
    
    #plt.plot(train_pred, label="train_pred")
    #plt.plot(distances_train, label="distances_train")
    #plt.legend()
    #plt.show()

    # evaluate
    print_results(y_test, X_pred)

    # Save the results DONT FORGET THE PREPROCESSING STEPS
    images_pred = model.predict(pca.transform(scaler.transform(images_test)))
    save_results(images_pred)
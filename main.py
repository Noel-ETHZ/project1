from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.model_selection import train_test_split
import numpy as np
import math

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
            "pca_components" : 20,
            "downsample_factor" : config["downsample_factor"],
            "test_size" : 0.15
        }
    
    downsample_factors = [1, 3, 10, 20, 30, 50]
    pca_components = [20, 50, 100, 200, 400]
    #test_sizes = [0.2, 0.15]
    n_neighborss = [2, 3]
    scalers = ["StandardScaler", "MinMaxScaler"]
    
    for ds_factor in reversed(downsample_factors):
        X, y = load_dataset(config, downsample_factor=ds_factor)
        #for ts in test_sizes:
        ts = params["test_size"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42) 
        for pca_c in pca_components:
            if pca_c > X.shape[1]:
                break

            for n_neighbors in n_neighborss:
                for sc in scalers:
                    if sc == "MinMaxScaler":
                        scaler = preprocessing.MinMaxScaler().fit(X_train)
                    else:
                        scaler = preprocessing.StandardScaler().fit(X_train)
                    X_train_s = scaler.transform(X_train)
                    X_test_s = scaler.transform(X_test)

                    pca = KernelPCA(n_components=pca_c, kernel="rbf")
                    pca.fit(X_train_s)

                    params["downsample_factor"] = ds_factor
                    params["test_size"] = ts
                    params["n_neighbors"] = n_neighbors
                    params["scaler"] = sc
                    params["pca_components"] = pca_c
                    # Save the parameters to a file
                    with open("test_cases.txt", "a") as f:
                        f.write(str(params) + "\n")
                    

                    X_train_pp = pca.transform(X_train_s)
                    X_test_pp = pca.transform(X_test_s)


                    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance", metric="manhattan")  

                    
                    model.fit(X_train_pp, y_train)
                        
                    # predict
                    X_pred = model.predict(X_test_pp)

                    # evaluate
                    MAE_test = print_results(y_test, X_pred)
                    with open("test_cases.txt", "a") as f:
                        f.write("MAE: " + str(MAE_test) + "\n")
                        f.write("======================\n")
                        f.write("\n")
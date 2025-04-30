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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 1. Normalization
    scaler = preprocessing.StandardScaler().fit(X_train)
    #scaler = preprocessing.MinMaxScaler().fit(X_train)

    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    params = {
            "model" : "MLPRegressor",
            "scaler": "StandardScaler",
            "hidden_layer_sizes" : (700,700,700),
            "pca_components" : 20,
            "downsample_factor" : config["downsample_factor"],
            "alpha" : 0.1
        }
    with open("test_cases.txt", "a") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

    

    #pca = KernelPCA(n_components=300, kernel="rbf")
    pca = PCA(n_components=params["pca_components"], whiten=True)
    pca.fit(X_train_s)

    X_train_pp = pca.transform(X_train_s)
    X_test_pp = pca.transform(X_test_s)

    #X_train_pp = X_train_s
    #X_test_pp = X_test_s

    # regression

    # define model
    #model = linear_model.Ridge(alpha=.5)
    #model = KNeighborsRegressor(n_neighbors=2, weights="distance", metric="manhattan")  
    #model = DecisionTreeRegressor()
    model = MLPRegressor(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        alpha=params["alpha"],
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        tol=1e-10,
        verbose=False,
        learning_rate="adaptive",
        learning_rate_init=0.000005,
    )


    # fine tuning loop
    last_MAE = 0
    last_iter = 0
    for i in range(100):
        for t in tqdm(range(50), desc="Training iterations"):
            model.partial_fit(X_train_pp, y_train)

        y_train_pred = model.predict(X_train_pp)
        y_test_pred = model.predict(X_test_pp)

        print(f"========= Iteration {i} =========")
        print("Train Score:")
        print_results(y_train, y_train_pred)
        print("Test Score:")
        last_iter = i
        last_MAE = print_results(y_test, y_test_pred)
        
    with open("test_cases.txt", "a") as f:
        f.write(f"Final Test Score: {last_MAE}, iteration: {last_iter}\n==============================\n")

    
    #model.fit(X_train_pp, y_train)
        
    # predict
    X_pred = model.predict(X_test_pp)
    
    #plt.plot(train_pred, label="train_pred")
    #plt.plot(distances_train, label="distances_train")
    #plt.legend()
    #plt.show()

    # evaluate
    print_results(y_test, X_pred)

    # Save the results DONT FORGET THE PREPROCESSING STEPS
    images_pred = model.predict(pca.transform(images_test))
    #save_results(images_pred)
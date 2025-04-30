from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.decomposition import KernelPCA, PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from matplotlib import pyplot as plt
from tqdm import tqdm
import time

# sklearn imports...
# SVRs are not allowed in this project.

if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()

    downsample_factors = [50, 10, 3]
    for downsample_factor in downsample_factors:
        # Load dataset: images (X) and corresponding minimum distance values (y)
        #X = np.load("/Users/noel/Documents/SML Project/project1/images_train.npy")
        #y = np.load("/Users/noel/Documents/SML Project/project1/labels_train.npy")
        X, y = load_dataset(config, downsample_factor)
        print(f"[INFO]: Dataset loaded with {len(X)} samples.")

        images_test = load_test_dataset(config)

        # TODO: Your implementation starts here
        # possible preprocessing steps ... training the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Test params:
        hidden_layer_sizes = ((300, 300, 300), (500, 500, 500), (500, 500), (700), (600, 600, 600), (700, 700, 700))
        alphas = (0.1, 0.01, 0.001)
        pca_component = 20
        #pca_components = (1,3,6)
        for hls in hidden_layer_sizes:
            for a in alphas:
                params = {
                    "scaler": "StandardScaler",                                    
                }

                params["hidden_layer_sizes"] = hls
                params["alpha"] = a
                params["pca_components"] = pca_component
                params["downsample_factor"] = downsample_factor


                # 1. Normalization
                scaler = preprocessing.StandardScaler().fit(X_train)
                #scaler = preprocessing.MinMaxScaler().fit(X_train)

                X_train_s = scaler.transform(X_train)
                X_test_s = scaler.transform(X_test)

                #pca = KernelPCA(n_components=300, kernel="rbf")
                pca = PCA(n_components=pca_component, whiten=True)
                pca.fit(X_train_s)

                X_train_pp = pca.transform(X_train_s)
                X_test_pp = pca.transform(X_test_s)

                #X_train_pp = X_train_s
                #X_test_pp = X_test_s

                # regression

                # define model
                #mmodel = RandomForestRegressor(
                    #n_estimators=100,
                    #max_depth=None,
                    #random_state=42,
                    #n_jobs=-1  # uses all available cores)
                #model = GradientBoostingRegressor(
                    #n_estimators=100,
                    #learning_rate=0.1,
                    #max_depth=3,
                    #random_state=42)
                #model = linear_model.Ridge(alpha=.5)
                #model = KNeighborsRegressor(n_neighbors=2, weights="distance", metric="manhattan")  
                #model = DecisionTreeRegressor()
                model = MLPRegressor(
                    hidden_layer_sizes=hls,
                    alpha=a,
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
                for i in range(20):
                    start_time = time.time()
                    for t in range(50):
                        model.partial_fit(X_train_pp, y_train)

                    y_train_pred = model.predict(X_train_pp)
                    y_test_pred = model.predict(X_test_pp)

                    
                    last_iter = i

                    print(f"========= Iteration {i} =========")
                    print("Train Score:")
                    print_results(y_train, y_train_pred)
                    print("Test Score:")
                    last_MAE = print_results(y_test, y_test_pred)
                    if time.time() - start_time > 120:
                        break
                
                #model.fit(X_train_pp, y_train)
                    
                # predict
                X_pred = model.predict(X_test_pp)
                
                # Save params and score to a text file
                with open("saved_parameters.txt", "a") as f:
                    for key, value in params.items():
                        f.write(f"{key}: {value}\n")
                    f.write(f"Final Test Score: {last_MAE}, iteration: {last_iter}\n==============================\n")

        
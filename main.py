from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from matplotlib import pyplot as plt

# sklearn imports...
# SVRs are not allowed in this project.

if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()

    # Load dataset: images (X) and corresponding minimum distance values (y)
    X = np.load("/Users/noel/Documents/SML Project/project1/images_train.npy")
    y = np.load("/Users/noel/Documents/SML Project/project1/labels_train.npy")
    print(f"[INFO]: Dataset loaded with {len(X)} samples.")

    images_test = load_test_dataset(config)

    # TODO: Your implementation starts here
    # possible preprocessing steps ... training the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 1. Normalization
    scaler = preprocessing.StandardScaler().fit(X)


    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=50)
    pca.fit(X_train_s)

    X_train_pp = pca.transform(X_train_s)
    X_test_pp = pca.transform(X_test_s)


    # regression

    # define model
    #model = linear_model.Ridge(alpha=.5)
    model = KNeighborsRegressor(n_neighbors=2, weights="distance")
    #model = DecisionTreeRegressor()
    '''model = MLPRegressor(
        hidden_layer_sizes=(500, 500, 500),
        activation="relu",
        solver="adam",
        max_iter=10000,
        random_state=42,
        tol=1e-10,
        verbose=True,
        learning_rate="adaptive",
        learning_rate_init=0.000005,
    )'''
   
    # 4. fit model
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
    images_pred = model.predict(pca.transform(images_test))
    #save_results(images_pred)
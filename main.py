from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np

if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()

    # Load training dataset: images and corresponding minimum distance values
    train_images, train_distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(train_images)} samples.")
    num_samples, num_features = train_images.shape

    # Load public dataset for testing
    X,y = load_dataset(config)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Load private dataset
    test_images = load_test_dataset(config)

    # Determine the appropriate number of components for PCA
    n = min(num_samples, num_features, 300)  # Ensure n_components is valid

    # KNeighbors with Pipeline including PCA
    knr_pipe = make_pipeline(RobustScaler(), PCA(n_components=n), KNeighborsRegressor())

    # Parameter grid for Grid Search
    param_grid = {
        'kneighborsregressor__n_neighbors': [2, 3, 4, 5, 6],
        'kneighborsregressor__weights': ['uniform', 'distance'],
        'kneighborsregressor__p': [2]  # p = 1 for Manhattan, p = 2 for Euclidean
    }

    # Grid Search
    grid_search = GridSearchCV(knr_pipe, param_grid, cv=5, n_jobs=-1)  # Adjust n_jobs to a reasonable number
    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_

    # Evaluation
    pred = best_estimator.predict(X_test)
    gt = y_test

    # print_results(gt, pred)
    #priv_pred = best_estimator.predict(priv_test_images)
    # Save the results
    print_results(gt, pred)

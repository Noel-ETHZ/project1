from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import cv2

if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()

    # Load training dataset: images and corresponding minimum distance values
    train_images, train_distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(train_images)} samples.")
    num_samples, num_features = train_images.shape

    def extract_fisher_features(images, n_components=16):
        # Reshape 1D vectors into images assuming square shape after downscaling
        downscale = config["downscaling_factor"]
        img_side = int(np.sqrt(90000 // downscale))
        images_reshaped = images.reshape((-1, img_side, img_side))

        # Optional greyscale conversion (images already 1-channel in this case)
        features = []
        for img in images_reshaped:
            # Extract dense SIFT or pixel patch features
            sift = cv2.SIFT_create()
            kp = [cv2.KeyPoint(x, y, 8) for y in range(0, img.shape[0], 8)
                                             for x in range(0, img.shape[1], 8)]
            _, desc = sift.compute(img.astype(np.uint8), kp)
            if desc is not None:
                features.append(desc)
        
        all_descriptors = np.vstack(features)
        
        # Fit a Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag')
        gmm.fit(all_descriptors)
        
        fisher_vectors = []
        for desc in features:
            if desc is None or len(desc) == 0:
                fisher_vectors.append(np.zeros(n_components * desc.shape[1] * 2))
                continue

            probs = gmm.predict_proba(desc)
            means = gmm.means_
            covariances = gmm.covariances_
            weights = gmm.weights_

            # First and second order statistics
            Q = probs.sum(axis=0)[:, np.newaxis]
            u = ((probs.T @ desc - Q * means) / np.sqrt(covariances)).flatten()
            v = ((probs.T @ (desc**2) - Q * (means**2 + covariances)) / np.sqrt(2 * covariances)).flatten()
            fisher_vectors.append(np.hstack((u, v)))
        
        return np.array(fisher_vectors)

    X_raw, y = load_dataset(config)
    X = extract_fisher_features(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    test_raw = load_test_dataset(config)
    test_images = extract_fisher_features(test_raw)

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
    # grid_search = GridSearchCV(knr_pipe, param_grid, cv=5, n_jobs=-1)  # Adjust n_jobs to a reasonable number
    # grid_search.fit(X_train, y_train)
    # best_estimator = grid_search.best_estimator_

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = KernelPCA(n_components=200, kernel="rbf")
    pca.fit(X_train_s)

    X_train_pp = pca.transform(X_train_s)
    X_test_pp = pca.transform(X_test_s)

    model = KNeighborsRegressor(n_neighbors=2, weights="distance", metric="manhattan")

    model.fit(X_train_s, y_train)

    # Evaluation
    #pred = best_estimator.predict(X_test)
    pred = model.predict(X_test_s)
    gt = y_test

    # print_results(gt, pred)
    #priv_pred = best_estimator.predict(priv_test_images)
    # Save the results
    print_results(gt, pred)

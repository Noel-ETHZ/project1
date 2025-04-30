from utils import load_config, load_dataset, load_test_dataset, print_results
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
import numpy as np

if __name__ == "__main__":
    config = load_config()
    images_test = load_test_dataset(config)

    params = {
        "model": "GradientBoosting",
        "scaler": "StandardScaler",
        "pca": "KernelPCA",
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "pca_components": 20,
        "downsample_factor": config["downsample_factor"],
        "test_size": 0.15
    }

    downsample_factors = [1, 3, 10, 20, 30]
    pca_components = [20, 50, 100, 200]
    test_size = params["test_size"]
    scalers = ["StandardScaler", "MinMaxScaler"]

    n_estimators_list = [50, 100, 200]
    learning_rates = [0.01, 0.05, 0.1]
    max_depths = [3, 5, 7]

    for ds_factor in reversed(downsample_factors):
        X, y = load_dataset(config, downsample_factor=ds_factor)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        for pca_c in pca_components:
            if pca_c > X.shape[1]:
                break

            for sc in scalers:
                if sc == "MinMaxScaler":
                    scaler = preprocessing.MinMaxScaler().fit(X_train)
                else:
                    scaler = preprocessing.StandardScaler().fit(X_train)

                X_train_s = scaler.transform(X_train)
                X_test_s = scaler.transform(X_test)

                pca = KernelPCA(n_components=pca_c, kernel="rbf")
                pca.fit(X_train_s)

                X_train_pp = pca.transform(X_train_s)
                X_test_pp = pca.transform(X_test_s)

                for n_est in n_estimators_list:
                    for lr in learning_rates:
                        for md in max_depths:
                            model = GradientBoostingRegressor(
                                n_estimators=n_est,
                                learning_rate=lr,
                                max_depth=md,
                                random_state=42
                            )

                            model.fit(X_train_pp, y_train)
                            y_pred = model.predict(X_test_pp)

                            mae = print_results(y_test, y_pred)

                            params.update({
                                "downsample_factor": ds_factor,
                                "pca_components": pca_c,
                                "scaler": sc,
                                "n_estimators": n_est,
                                "learning_rate": lr,
                                "max_depth": md
                            })

                            with open("test_cases.txt", "a") as f:
                                f.write(str(params) + "\n")
                                f.write("MAE: " + str(mae) + "\n")
                                f.write("======================\n\n")
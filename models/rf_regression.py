import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing


def run_rf(X_train, X_test, y_train, y_test):
    """
    运行随机森林回归模型 (包含 GridSearch 调参)
    """
    print("\n--- Start Training Random Forest (Advanced Tuning) ---")
    print("Searching for the optimal parameters. This may take 1-2 minutes....")

    param_grid = {
        'n_estimators': [100, 200],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=1)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=1, verbose=1, scoring='r2')

    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    print(f" optimal parameters: {grid_search.best_params_}")

    y_pred = best_rf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    print("\nSample predictions (Random Forest):")
    for true, pred in zip(y_test[:5], y_pred[:5]):
        print(f"True: {true:.3f} | Predicted: {pred:.3f}")

    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'model': best_rf,
        'predictions': y_pred
    }


# self test
if __name__ == "__main__":
    print("正在进行独立自我测试...")
    data = fetch_california_housing()
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    run_rf(X_train_d, X_test_d, y_train_d, y_test_d)
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

def run_svr(X_train, X_test, y_train, y_test):
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr_model.fit(X_train, y_train)
    y_pred = svr_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print("\nSample predictions:")
    for true, pred in zip(y_test[:5], y_pred[:5]):
        print(f"True: {true:.3f} | Predicted: {pred:.3f}")

    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'y_pred': y_pred
    }

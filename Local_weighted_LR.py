import numpy as np
import scipy.io
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

data=scipy.io.loadmat('data.mat')
#print("Top_level keys:",data.keys())
data=data['data']
X=data[:,0].reshape(-1,1)
y=data[:,-1].flatten()

EPSILON=1e-4


def lw_reg(x,x_i,y_train,h):
    n=x_i.shape[0]
    x_int=np.hstack((np.ones((n,1)), x_i)) ##adding intercept theta
    x_query=np.array([1.0,float(x)])

    W = np.diag(np.exp(-np.square(x_i.flatten() - float(x)) / (2 * h**2)))
    beta = np.linalg.inv(x_int.T @ W @ x_int + EPSILON * np.eye(2)) @ x_int.T @ W @ y_train
    return x_query @ beta


def predict(X,Y,h):
    predictions=[]
    for x_p in X:
        pred=lw_reg(x_p,X,Y,h)
        predictions.append(pred)
    return np.array(predictions)

##Cross-validation
h_range=np.linspace(0.01,1.5,20)
cv_errors=[]
kf=KFold(n_splits=5,shuffle=True,random_state=42)

for h in h_range:
    Errors=[]
    for train_index, vali_index in kf.split(X):
        x_train,x_vali=X[train_index],X[vali_index] ##training and validation datasets
        y_train,y_vali=y[train_index],y[vali_index]

        predits=[]
        for x_p in x_vali:
            pred=lw_reg(x_p,x_train,y_train,h)
            predits.append(pred)

        predictions=np.array(predits)
        error=np.mean((y_vali-predictions)**2)
        Errors.append(error)

    mean_error=np.mean(Errors)
    cv_errors.append(mean_error)


plt.figure(figsize=(8, 5))
plt.plot(h_range, cv_errors, marker='o')
plt.xlabel("Bandwidth $h$")
plt.ylabel("Cross-Validation Error")
plt.title("Cross-Validation Curve")
plt.grid(True)
plt.show()


optimal_h = h_range[np.argmin(cv_errors)]
print(f"Optimal bandwidth h: {optimal_h:.4f}")

###Make predictions using optimal h

x_query = -1
y_pred_at_x = lw_reg(x_query, X, y, optimal_h)
print(f"Predicted y at x = {x_query}: {y_pred_at_x:.4f}")


x_range = np.linspace(X.min() - 1, X.max() + 1, 400).reshape(-1, 1)
def predict(query_points, x_train, y_train, h):
    predictions = []
    for x_p in query_points:
        pred = lw_reg(x_p, x_train, y_train, h)
        predictions.append(pred)
    return np.array(predictions)

y_curve = predict(x_range, X, y, optimal_h)



plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='gray', alpha=0.6, label='Training Data')
plt.plot(x_range, y_curve, color='red', label='Prediction Curve')
plt.scatter(x_query, y_pred_at_x, color='green', s=100, zorder=5, label=f'Prediction at x={x_query}')
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"LWR Prediction at x = {x_query} (h = {optimal_h:.3f})")
plt.legend()
plt.grid(True)
plt.show()

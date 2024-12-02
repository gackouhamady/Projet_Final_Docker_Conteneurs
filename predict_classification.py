import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def predict(model_name):
  
    with open(model_name + '.pkl', 'rb') as f:
        model = pickle.load(f)
 
    data = np.genfromtxt('validation.csv', delimiter=',')
    X_val = data[:, :-1]
    y_val = data[:, -1]  

    
    y_pred = model.predict(X_val)

     
    plt.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap=plt.cm.coolwarm, label='Vrai Labels', edgecolor='k', s = [80 for _ in range(len(y_val))])
    plt.scatter(X_val[:, 0], X_val[:, 1], c=y_pred, cmap=plt.cm.Paired, marker='x', label='Prédictions')
    plt.legend()
    plt.title(model_name)
    plt.show()

    
    report = classification_report(y_val, y_pred)
    print(report)

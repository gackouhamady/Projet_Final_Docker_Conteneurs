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

     
    plt.figure(figsize=(10, 7))

     
    plt.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap='viridis', alpha=0.5, label="Vraies étiquettes")

    
    plt.scatter(X_val[:, 0], X_val[:, 1], c=y_pred, cmap='cool', alpha=0.5, s=20, label="Prédictions")

    plt.title("Visualisation des données de validation et des prédictions")
    plt.xlabel("Caractéristique 1")
    plt.ylabel("Caractéristique 2")
    plt.legend()
    plt.colorbar(label="Classes")

    
    report = classification_report(y_val, y_pred)
    plt.figtext(0.5, -0.1, report, wrap=True, horizontalalignment='center', fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict()

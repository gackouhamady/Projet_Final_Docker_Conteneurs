from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pickle
import numpy as np

import matplotlib.pyplot as plt

class Model:
    def __init__(self, model_name="SVM"):
        self.model_name = model_name
        self.model = None
        self.params = None
        self._init_model()

    def _init_model(self):
        if self.model_name == "SVM":
            self.model = SVC()
            self.params = {
                'C': [0.1, 1, 10, 100],            
                'kernel': ['rbf', 'poly', 'sigmoid'],  
            }

        elif self.model_name == "RandomForest":
            self.model = RandomForestClassifier()
            self.params = {
                    'n_estimators': [50, 100, 200],        
                    'max_depth': [None, 10, 20, 30],             
                }
        elif self.model_name == "DecisionTree":
            self.model = DecisionTreeClassifier()
            self.params = {
                    'criterion': ['gini', 'entropy', 'log_loss'],                         
                    'max_leaf_nodes': [None, 10, 20, 50],         
                }
       
        elif self.model_name == "LogisticRegression":
            self.model = LogisticRegression()
            self.params = {
                'penalty': ['l2', 'l1', 'elasticnet', None],
                'fit_intercept': [True, False],
            }  
        else:
            raise ValueError(f"Modèle {self.model_name} non supporté. Choisissez parmi : \
                            SVM, RandomForest, DecisionTree, LogisticRegression.")
    
    def get_params(self):
        return self.params
   
    def get_model_name(self):
        return self.model_name
    def get_model(self):
        return self.model


class Train:
    def __init__(self, dataset_instance, model, scoring, cv=5):
        """
        Initialise l'objet Train avec les données et le modèle sélectionné.
        """
        self.dataset = dataset_instance
        self.model = model
        self.scoring = scoring
        self.cv = cv
        self.best_model = None
        self.best_params = None
        self.best_score = None

    def train(self):
        """
        Sélectionne le meilleur modèle avec GridSearchCV.
        """
        X_train, y_train = self.dataset.get_train()
        best_result = None
        
        print(f"Recherche des meilleurs hyperparamètres pour {self.model.get_model_name()}...")
        grid_search = GridSearchCV(estimator=self.model.get_model(), param_grid=self.model.get_params(), scoring=self.scoring, cv=self.cv, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        if best_result is None or grid_search.best_score_ > self.best_score:
            self.best_model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_
            best_result = (self.model.get_model_name(), grid_search.best_params_, grid_search.best_score_)

        print(f"Meilleur modèle : {best_result[0]}")
        print(f"Meilleurs hyperparamètres : {best_result[1]}")
        print(f"Meilleur score ({self.scoring}) : {best_result[2]:.4f}")
        return self.best_model, self.best_score
    
    def get_model(self):
        return self.model




class Dataset:
    def __init__(self, n_informative = 2, n_classes = 2, n_clusters_per_class = 1, class_sep = 1.0, n_samples = 10000):
        self.n_informative = n_informative
        self.n_classes = n_classes
        self.n_clusters_per_class = n_clusters_per_class
        self.class_sep = class_sep
        self.n_samples = n_samples
        self.X, self.y = make_classification( n_classes=self.n_classes,
                                            n_informative=self.n_informative,
                                            n_clusters_per_class=self.n_clusters_per_class,
                                            class_sep=self.class_sep,
                                            n_samples=self.n_samples,
                                            scale=100,
                                            random_state=45)

        self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val = self._generate_split()

    def _generate_split(self):
        X_train, X_rest, y_train, y_rest = train_test_split(self.X, self.y, test_size=0.3, random_state=1)
        X_test, X_val, y_test, y_val = train_test_split(X_rest, y_rest, test_size=(1/3), random_state=1)

        np.savetxt("train.csv", np.hstack((X_train, y_train.reshape(-1, 1))), delimiter=",")
        np.savetxt("test.csv", np.hstack((X_test, y_test.reshape(-1, 1))), delimiter=",")
        np.savetxt("validation.csv", np.hstack((X_val, y_val.reshape(-1, 1))), delimiter=",")

        return X_train, y_train, X_test, y_test, X_val, y_val

    def get_train(self):
        return self.X_train, self.y_train
    def get_test(self):
        return self.X_test, self.y_test
    def get_validation(self):
        return self.X_val, self.y_val
    
    def reduce_dataset(self):
        pca = PCA(n_components=2, random_state=42)
        pca.fit(self.X_train)
        return pca.transform(self.X_train)
    
    def visualize(self):
        if self.X.shape[1] > 2:
            X = self.reduce_dataset()
        else:
            X = self.X_train
        plt.title("Dataset")
        plt.ylabel('X2')
        plt.xlabel('X1')
        c = ["red" if y == 1 else "blue" for y in self.y_train]
        plt.scatter(X[:,0], X[:,1], c=c)
        plt.show()

data = Dataset(n_informative=2, n_classes=2, n_clusters_per_class=1, class_sep=1.3, n_samples=10000)
data.visualize()

for model_name in {'SVM', "LogisticRegression", "RandomForest", "DecisionTree"}:
    model = Model(model_name=model_name)
    train = Train(dataset_instance=data, model=model, scoring='f1', cv=5)
    best_model, best_score = train.train()
    pickle.dump(best_model, open(model_name + ".pkl", 'wb')) 

    
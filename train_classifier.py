from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class model:
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
                'kernel': ['linear', 'rbf', 'poly'],  
                'degree': [2, 3, 4],      
                'gamma': ['scale', 'auto', 0.01, 0.1, 1], 
            }

        elif self.model_name == "RandomForest":
            self.model = RandomForestClassifier()
            self.params = {
                    'n_estimators': [50, 100, 200],        
                    'max_depth': [None, 10, 20, 30],    
                    'min_samples_split': [2, 5, 10],     
                    'min_samples_leaf': [1, 2, 4],        
                    'max_features': ['sqrt', 'log2'],    
                    'bootstrap': [True, False],           
                }
        elif self.model_name == "DecisionTree":
            self.model = DecisionTreeClassifier()
            self.params = {
                    'criterion': ['gini', 'entropy', 'log_loss'],  # Fonction de mesure de l'impureté
                    'splitter': ['best', 'random'],               # Méthode pour choisir la variable de split
                    'max_depth': [None, 5, 10, 20, 30],           # Profondeur maximale de l'arbre
                    'min_samples_split': [2, 5, 10],              # Minimum d'échantillons requis pour diviser un nœud
                    'min_samples_leaf': [1, 2, 4],                # Minimum d'échantillons dans une feuille
                    'max_features': [None, 'sqrt', 'log2'],       # Nombre de features à considérer pour chaque split
                    'max_leaf_nodes': [None, 10, 20, 50],         # Nombre maximum de feuilles
                }
       
        elif self.model_name == "LogisticRegression":
            self.model = LogisticRegression()
            self.params = {
                'penalty': ['l2', 'l1', 'elasticnet', None],  # Type de régularisation
                'C': [0.01, 0.1, 1, 10, 100],  # Paramètre de régularisation : inverse de la force de régularisation
                'solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg'],  # Méthode d'optimisation
                'max_iter': [100, 200, 300],  # Nombre d'itérations maximales pour l'optimisation
                'tol': [1e-4, 1e-3, 1e-2],  # Critère de tolérance pour la convergence
                'fit_intercept': [True, False],  # Ajouter ou non une constante au modèle
                'class_weight': [None, 'balanced'],  # Poids des classes (utile pour les jeux de données déséquilibrés)
                'multi_class': ['ovr', 'multinomial'],  # Stratégie pour la régression logistique multi-classes
                'warm_start': [True, False],  # Utiliser les solutions précédentes pour les nouvelles itérations
                'l1_ratio': [0, 0.5, 1],  # Si la pénalité est 'elasticnet', le ratio entre L1 et L2
            }  
        else:
            raise ValueError(f"Modèle {self.model_name} non supporté. Choisissez parmi : \
                            SVM, RandomForest, DecisionTree, LogisticRegression.")
    
    def get_model_parms(self):
        return self.params
   
    def get_model_name(self):
        return self.model_name


class Train:
    def __init__(self, dataset_instance, model, scoring, cv=5):
        """
        Initialise l'objet Train avec les données et le modèle sélectionné.
        
        Parameters:
        - dataset_instance: instance de la classe DataSet
        - model_name: nom du modèle choisi ("SVM", "RandomForest", "DecisionTree", "LogisticRegression")
        - model_params: dictionnaire contenant les paramètres spécifiques au modèle
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
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.model.get_params(), scoring=self.scoring, cv=self.cv, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        if best_result is None or grid_search.best_score_ > self.best_score:
            self.best_model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_
            best_result = (self.model.get_model_name(), grid_search.best_params_, grid_search.best_score_)

        print(f"Meilleur modèle : {best_result[0]}")
        print(f"Meilleurs hyperparamètres : {best_result[1]}")
        print(f"Meilleur score ({self.scoring}) : {best_result[2]:.4f}")
        return self.best_model
    
    def get_model(self):
        """
        Retourne le modèle entraîné.
        """
        return self.model

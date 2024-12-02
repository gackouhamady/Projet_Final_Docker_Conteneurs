from train_classifier import Model, Train, Dataset
from predict_classification import predict
import pickle

data = Dataset(n_informative=2, n_classes=2, n_clusters_per_class=1, class_sep=1.3, n_samples=10000)
data.visualize()

print("Choisir les modèles à entrainer (entrer leur nombres):")
print("1- SVM")
print("2- Regression logistique")
print("3- Forets aleatoires")
print("4-Arbre de décision")
choices = int(input())
MODELS = ['SVM', "LogisticRegression", "RandomForest", "DecisionTree"]
MODELS = [m for m in MODELS if MODELS.index(m) + 1 in choices ]

for model_name in MODELS:
    model = Model(model_name=model_name)
    train = Train(dataset_instance=data, model=model, scoring='f1', cv=5)
    best_model, best_score = train.train()
    pickle.dump(best_model, open(model_name + ".pkl", 'wb'))

    predict(model_name)
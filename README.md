# Projet_Final_Docker_Conteneurs  
## Déploiement:  
git clone https://github.com/gackouhamady/Projet_Final_Docker_Conteneurs.git  
docker pull omarnms/examen-docker:v1.0  
docker compose build  
docker compose run --rm app 
## Packages utilisées:
1. scikit-learn
2. numpy
3. pandas
4. pickle-mixin
5. matplotlib
## Features:  
L'application fonctionne selon les étapes suivantes:
1. Demander à l'utilisateur de choisir les modèles à entrainer parmi: SVM, regression logistique, arbre de decision et forets aléatoire.
2. Créer un dataset synthétique de 10000 samples avec **make_classification()** et exporter les trois parties entrainement, test et validation en *.csv*.
3. Pour chaque modèle choisi par utilisateur:
   1. Faire une sélection de modèle avec Gridsearch (avec une 5-fold cross-validation).
   2. Exporter le meilleur modèle entrainé en *.pkl*.
   3. Tester les modèles sur les données de validation.
   4. Afficher le rapport de classification et ploter les prédictions du modèles.
4. Jusqu'à l'utilisateur entre 0.

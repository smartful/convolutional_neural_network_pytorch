# Classification d'image avec un Réseau de Neurones Convolutionnel (CNN) avec Pytorch

Exemple de CNN avec PyTorch.

## Utilisation du dataset Fashion MNIST : 

**Fashion-MNIST** est un ensemble de données d'images d'articles de Zalando, composé d'**un ensemble d'apprentissage de 60 000 exemples** et d'**un ensemble de test de 10 000 exemples**.

Chaque exemple est une image en **niveaux de gris** de **28x28**, associée à une étiquette parmi **10 classes**.

Zalando souhaite que **Fashion-MNIST** remplace directement l'ensemble de données **MNIST original** pour l'évaluation comparative des algorithmes d'apprentissage automatique.

Il partage la même taille d'image et la même structure de divisions d'entraînement et de test.

[Lien du dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist)


## Étapes du Notebook : 
- Import du dataset.
- Visualisation des datas.
- Création des mini-batches sur la data.
- Classification avec un premier modèle : un réseau de neurones basique.
- Création du modèle, définition de la loss function, de l'optimizer, puis entraînement du modèle.
- Visualisation des courbes de loss et d'accuracy.
- Classification avec un deuxième modèle : un réseau de neurones convolutionnel (CNN).
- Création du modèle, définition de la loss function, de l'optimizer, puis entraînement du modèle.
- Visualisation des courbes de loss et d'accuracy.
- Comparaison des modèles.
- Visualisation des prédictions.
- Evaluation des prédictions avec une matrice de confusion.

# Cours Complet sur PyTorch

### Introduction à PyTorch

PyTorch est une bibliothèque open-source développée par Facebook pour le calcul scientifique et les applications de machine learning, notamment le deep learning. Elle est particulièrement appréciée pour sa flexibilité et sa facilité d'utilisation, permettant de créer des modèles de machine learning de manière dynamique.

---
### Installation de PyTorch

Après avoir activé votre environnement virtuel, installez PyTorch avec pip :

Sur Mac/Linux:
```bash
pip3 install torch torchvision torchaudio
```

Sur Windows (CPU):
```bash
pip3 install torch torchvision torchaudio
```
Sur Windows (CUDA):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
<ins>Nota Bene:</ins> 
- 124 correspnd à la version 12.4 de CUDA
- Il peut être préférable d'installer PyTorch avec Conda.


---

### Caractéristiques Principales de PyTorch

1. **Tensors** : Structures de données similaires aux tableaux NumPy, mais avec un support intégré pour les calculs sur GPU.
2. **Autograd** : Système de différenciation automatique qui permet de calculer facilement les gradients pour l'entraînement des modèles.
3. **Modules et Layers** : Structure modulaire permettant de construire des modèles de manière simple et réutilisable.
4. **Optim** : Fournit divers algorithmes d'optimisation (comme SGD, Adam) pour ajuster les poids du modèle.
5. **Dynamic Computational Graphs** : Les graphes computationnels sont créés dynamiquement, ce qui permet une flexibilité accrue.

---

### Tensors en PyTorch

Les Tensors sont la base de PyTorch. Ils peuvent être manipulés de manière similaire aux tableaux NumPy, mais avec la possibilité de fonctionner sur des GPU.

#### Création de Tensors

```python
import torch

# Tensor vide
tensor_vide = torch.empty(2, 3)
print(tensor_vide)

# Tensor de zéros
tensor_zeros = torch.zeros(2, 3)
print(tensor_zeros)

# Tensor de uns
tensor_ones = torch.ones(2, 3)
print(tensor_ones)

# Tensor aléatoire
tensor_random = torch.rand(2, 3)
print(tensor_random)
```

#### Opérations sur Tensors

```python
# Addition
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
z = x + y
print(z)

# Multiplication par un scalaire
z = x * 2
print(z)

# Produit matriciel
mat1 = torch.tensor([[1, 2], [3, 4]])
mat2 = torch.tensor([[5, 6], [7, 8]])
mat_result = torch.mm(mat1, mat2)
print(mat_result)
```

#### Tensors et GPU

```python
# Déplacer un Tensor vers le GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor_gpu = tensor_ones.to(device)
print(tensor_gpu)

# Déplacer de nouveau vers le CPU
tensor_cpu = tensor_gpu.to("cpu")
print(tensor_cpu)
```

---

### Autograd : Calcul Automatique des Gradients

`torch.autograd` est le moteur de différenciation automatique de PyTorch. Il calcule automatiquement les gradients des Tensors en fonction de leur chaîne d'opérations.

#### Exemple de Calcul de Gradient

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad) # Gradient de y par rapport à x, ici 2*x = 4
```

#### Désactivation de l'Autograd

```python
with torch.no_grad():
 y = x ** 2
 print(y.requires_grad) # False, car autograd est désactivé
```

---

### Modules et Layers en PyTorch

Les modèles dans PyTorch sont définis en tant que classes qui héritent de `nn.Module`.

#### Création d’un Module Simple

```python
import torch.nn as nn

class SimpleModel(nn.Module):
 def __init__(self):
 super(SimpleModel, self).__init__()
 self.layer1 = nn.Linear(2, 3) # Couche linéaire (entrée de taille 2, sortie de taille 3)
 self.layer2 = nn.ReLU() # Fonction d'activation ReLU
 self.layer3 = nn.Linear(3, 1) # Couche linéaire (entrée de taille 3, sortie de taille 1)

 def forward(self, x):
 x = self.layer1(x)
 x = self.layer2(x)
 x = self.layer3(x)
 return x

model = SimpleModel()
print(model)
```

---

### Optimisation en PyTorch

Le module `torch.optim` contient des algorithmes d'optimisation comme la descente de gradient stochastique (SGD) et Adam.

#### Exemple de Configuration de l'Optimiseur

```python
# Définir un modèle simple
model = SimpleModel()

# Définir un optimiseur
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Définir une fonction de perte
criterion = nn.MSELoss()

# Exemple de boucle d'entraînement
for epoch in range(10):
 # Entrée fictive et cible
 inputs = torch.rand(1, 2)
 targets = torch.rand(1, 1)

 # Réinitialiser les gradients
 optimizer.zero_grad()

 # Calculer la sortie
 outputs = model(inputs)

 # Calculer la perte
 loss = criterion(outputs, targets)

 # Calculer les gradients
 loss.backward()

 # Mettre à jour les poids
 optimizer.step()

 print(f'Epoch {epoch+1}, Loss: {loss.item()}')
```

---

### Sauvegarde et Chargement de Modèles

La **sauvegarde d'un modèle** en PyTorch permet de préserver l'état du modèle après son entraînement, pour plusieurs raisons pratiques :

#### 1. **Réutilisation du Modèle**

Après avoir entraîné un modèle, tu peux sauvegarder ses poids et ses paramètres pour :
- **Éviter de réentraîner** le modèle depuis le début, ce qui économise du temps et des ressources, surtout pour des modèles complexes ou entraînés sur de grandes quantités de données.

#### 2. **Déploiement**

- Une fois entraîné, un modèle peut être **déployé** dans un environnement de production où il sera utilisé pour faire des prédictions sur de nouvelles données sans avoir besoin de réentraînement.

#### 3. **Partage du Modèle**

- La sauvegarde permet de **partager** un modèle avec d'autres développeurs ou équipes sans partager les données d'entraînement. Cela facilite la collaboration.

#### 4. **Reprise d’Entraînement**

- Si l'entraînement d'un modèle est interrompu (par exemple, à cause d'une panne de courant), la sauvegarde permet de **reprendre l'entraînement** à partir de l'état sauvegardé plutôt que de recommencer depuis le début.

#### 5. **Versioning des Modèles**

- En sauvegardant différents états du modèle à différents moments (par exemple, après chaque époque), tu peux garder une **trace des performances** et revenir à une version précédente si nécessaire.

#### 6. **Validation et Évaluation**

- Tu peux charger un modèle sauvegardé pour effectuer des **tests** ou **évaluations** supplémentaires, ou pour **valider** ses performances sur de nouvelles données sans avoir à réentraîner.

### Sauvegarde du Modèle

```python
torch.save(model.state_dict(), 'model.pth')
```

### Chargement du Modèle

```python
model = SimpleModel()
model.load_state_dict(torch.load('model.pth'))
model.eval() # Passer en mode évaluation
```

---

### Conclusion

Ce cours présente les concepts fondamentaux de PyTorch : les Tensors, le module `autograd`, la construction de modèles avec `nn.Module`, et les optimisations avec `torch.optim`. PyTorch est puissant et flexible, ce qui le rend idéal pour le développement rapide et l'expérimentation en deep learning.



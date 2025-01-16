# Cours sur les environnements virtuels

### Introduction à `venv`

`venv` est un module intégré à Python à partir de la version 3.3 qui permet de créer des environnements virtuels isolés. Un environnement virtuel est un répertoire contenant une installation de Python et une copie de la structure de répertoire utilisée pour installer des modules tiers (comme `pip`).

L'utilisation d'environnements virtuels permet de gérer facilement les dépendances de différents projets, éviter les conflits de version et maintenir un environnement de développement propre.

---

### Pourquoi Utiliser `venv` ?

1. **Isolation** : Chaque projet peut avoir ses propres dépendances, même si elles ont des versions différentes d'un projet à l'autre.
2. **Contrôle des Dépendances** : Évitez les conflits de dépendances en ayant une version spécifique d'une bibliothèque pour chaque projet.
3. **Facilité de Déploiement** : Recréez l'environnement sur une autre machine avec les mêmes dépendances en utilisant un fichier `requirements.txt`.
4. **Propreté** : Gardez votre environnement global Python propre sans avoir à installer toutes les bibliothèques globalement.

---

### Commandes de Base avec `venv`

#### 1. **Créer un Environnement Virtuel**
Sur Mac/Linux :
```bash
python3 -m venv nom_env
```
Sur Windows :
```bash
python -m venv nom_env
```
Cela crée un répertoire `nom_env` contenant l'installation de Python et `pip`.

#### 2. **Activer l’Environnement Virtuel**
- **Sur Mac/Linux** :
 ```bash
 source nom_env/bin/activate
 ```
- **Sur Windows** :
 ```bash
 nom_env\Scripts\activate
 ```

Une fois activé, tu verras le nom de l'environnement au début de la ligne de commande, indiquant que tu travailles dans l'environnement virtuel.

#### 3. **Désactiver l’Environnement Virtuel**
Pour quitter l'environnement virtuel :
```bash
deactivate
```

#### 4. **Installer des Paquets**
Lorsque l'environnement est activé, utilise `pip` pour installer les paquets spécifiques à cet environnement :
```bash
pip install nom_du_paquet
```

#### 5. **Lister les Paquets Installés**
```bash
pip list
```

#### 6. **Créer un Fichier `requirements.txt`**
Ce fichier répertorie toutes les dépendances de ton projet avec leurs versions spécifiques. Il peut être généré comme suit :
```bash
pip freeze > requirements.txt
```

#### 7. **Recréer un Environnement à Partir de `requirements.txt`**
Sur une autre machine, recrée l'environnement virtuel et installe les dépendances :
```bash
pip install -r requirements.txt
```

---

### Structure d'un Environnement Virtuel

- **`bin` ou `Scripts`** : Contient les exécutables pour l'environnement (Python, `pip`, etc.).
- **`lib`** : Contient les bibliothèques Python spécifiques à cet environnement.
- **`pyvenv.cfg`** : Fichier de configuration de l'environnement virtuel.

---

### Cas d'Usage Avancés

#### 1. **Utilisation avec Différentes Versions de Python**
Tu peux créer un environnement virtuel avec une version spécifique de Python (si elle est installée sur ton système) :
```bash
python3.9 -m venv nom_env
```

#### 2. **Suppression d’un Environnement Virtuel**
Pour supprimer un environnement virtuel, il suffit de supprimer le répertoire de l'environnement :
```bash
rm -rf nom_env
```
ou
```bash
rmdir /S /Q nom_env
```

#### 3. **Environnement Virtuel dans un Projet Git**
Il est recommandé de ne **pas** inclure le répertoire de l'environnement virtuel dans ton dépôt Git. Ajoute-le à ton fichier `.gitignore` :
```
nom_env/
```

#### 4. **Automatisation avec des Scripts**
Tu peux écrire des scripts pour automatiser l'activation de l'environnement virtuel et l'installation des dépendances.

Exemple de script `Makefile` pour Mac/Linux :
```makefile
venv:
 python3 -m venv nom_env
 source nom_env/bin/activate
 pip install -r requirements.txt

activate:
 source nom_env/bin/activate
```

---

### Problèmes Courants et Solutions

1. **Commande `python` ou `pip` non trouvée après activation** :
 - Assure-toi d'avoir correctement activé l'environnement.
 - Vérifie que le répertoire `bin` ou `Scripts` est dans ton PATH.

2. **Problèmes de Permission** :
 - Si tu as des problèmes de permission lors de la création de l'environnement, essaie de le créer dans un répertoire où tu as les droits d'écriture.

3. **Incompatibilité de Version de Python** :
 - Si tu veux utiliser une version spécifique de Python, assure-toi qu'elle est installée sur ton système avant de créer l'environnement virtuel.

---

### Conclusion

`venv` est un outil puissant pour gérer des environnements de développement Python. Il assure une isolation des dépendances et facilite la gestion et la reproduction d'environnements de projet, rendant le développement plus propre et plus structuré. En utilisant `venv`, tu peux éviter les conflits de paquets et maintenir un flux de travail cohérent sur différents systèmes d'exploitation.
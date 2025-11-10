# Programmation GPU : La Puissance du Parallélisme au Service de l'Informatique

## Introduction (1 minute 30)  **_Bapt_**

### 1.1 Le "Pourquoi" du GPU (Accroche)

Carte graphique

**Constat :**  
Aujourd'hui, les problèmes informatiques modernes (IA, science, cryptographie) nécessitent des calculs massifs et répétitifs.  
Le CPU est le maître de l’ordinateur, mais il a besoin d’aide.

**Objectif de la présentation :**  
Comprendre comment le GPU a révolutionné l’informatique grâce à sa capacité unique de parallélisme massif.

### 1.2 L’Analogie Clé : Le Cerveau vs. L’Armée

- **Le CPU (Central Processing Unit)** : Le PDG.  
  Expert extrêmement intelligent, capable de résoudre n’importe quel problème de A à Z,  
  mais il ne fait qu’une chose à la fois → **séquentiel**.

- **Le GPU (Graphics Processing Unit)** : L’Armée.  
  Des milliers d’ouvriers peu spécialisés mais capables d’exécuter **la même tâche simple**,  
  **en même temps**, sur **des millions de données** → **parallèle**.

---

## I. Le Concept Fondamental : Parallélisme Massif (3 minutes) **_Elouan_** /  **_Bapt_**

### 2.1 La Différence Structurelle (Simple) **_Elouan_**

- CPU : quelques cœurs, gros, optimisés pour la **latence**.  
- GPU : des milliers de petits cœurs, optimisés pour le **débit (throughput)**.

C’est cette architecture qui permet un traitement simultané massif.
 
### 2.2 Quand utiliser le GPU ? **_Elouan_**

Le GPU excelle dans les problèmes **“embarrassingly parallel”**, c'est-à-dire où les calculs sont indépendants :

**Exemples :**
- **Rendu d’image** : chaque pixel est calculé indépendamment.
- **Entraînement d’IA** : chaque donnée passe dans les mêmes matrices.

### 2.3 Comment ça “Programme” ? **_Bapt_**

Principe :  
Le programmeur envoie **un seul kernel** au GPU et lui dit :  
**"Exécute ce kernel un million de fois, chacun sur une donnée différente."**

Langages permettant cela :  
- **CUDA** (NVIDIA)  
- **OpenCL** (générique, multi-constructeurs)

*(Pas besoin d’entrer dans la syntaxe.)*

---

## II. Applications Modernes (2 minutes) **_Elouan_**

### Intelligence Artificielle
Le Deep Learning est un gigantesque calcul matriciel.  
Les GPU sont les moteurs derrière ChatGPT, Stable Diffusion, etc.

### Cryptomonnaies
Le minage repose sur la recherche rapide et répétitive de solutions cryptographiques → parfait pour le parallélisme.

### Science et Ingénierie
Simulations physiques (fluides, molécules), modélisation scientifique, calcul intensif, etc.

---

## III. Exemple Original : Le Casseur de Code à 4 Chiffres (3 minutes 30) **_A définir_**

### 4.1 Le Défi
Un code secret à **4 chiffres** : de 0000 à 9999.  
Objectif : le retrouver le plus rapidement possible.

### 4.2 La Méthode CPU (Le PDG)
Le CPU teste les combinaisons **une par une** :  
0000 → 0001 → 0002 → …  

**Temps nécessaire :**  
Jusqu’à **10 000 tentatives séquentielles**.

### 4.3 La Méthode GPU (L’Armée)
Le GPU lance **10 000 ouvriers en parallèle**.  
Chaque ouvrier teste **une unique combinaison** :  
(l’ouvrier n°4732 teste la combinaison 4732)

Dès qu’un ouvrier trouve la bonne valeur → **la tâche est terminée**.

**Temps nécessaire :**  
En théorie : **le temps d’une seule tentative** (plus un faible temps de préparation).

Impact visuel :  
- **CPU** → prendrait plusieurs secondes  
- **GPU** → solution quasi instantanée

---

## Conclusion (30 secondes) **_A définir_**

Le GPU est devenu essentiel pour résoudre les problèmes à grande échelle.  
Il ne remplace pas le CPU :  
- le **CPU** gère la logique,  
- le **GPU** apporte la force brute parallèle.

Ensemble, ils forment le duo central de l’informatique moderne.




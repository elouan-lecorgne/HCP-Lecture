[Lien vers cours complet ](./Cours.md)
# Programmation GPU avec CUDA

## Qu'est-ce que la programmation GPU ?

La programmation GPU exploite la carte graphique pour **accélérer massivement les calculs parallèles**. Au lieu d'afficher des images, le GPU devient un accélérateur capable d'exécuter **des milliers d'opérations simultanément** sur de grandes quantités de données.

**Principe fondamental :** Quand une tâche peut être divisée en milliers de petits calculs **indépendants**, le GPU les traite tous en parallèle, offrant des gains de performance considérables par rapport au CPU.

---

## CPU vs GPU : Deux Philosophies de Calcul

### Le CPU : Le PDG (Latency Oriented)
- **Peu de cœurs** (4-32), mais très rapides et polyvalents
- **Excellente réactivité** : gère les tâches complexes, les branchements, les interruptions
- **Traitement séquentiel** : une tâche après l'autre
- **Idéal pour** : logique complexe, tâches dépendantes, coordination

**Analogie :** Un professeur brillant qui résout des problèmes complexes, mais doit corriger 10 000 copies simples une par une.

### Le GPU : L'Armée (Throughput Oriented)
- **Des milliers de cœurs** (2 000-10 000+), mais simplifiés
- **Débit massif** : tous exécutent la même instruction sur des données différentes (SIMD)
- **Traitement parallèle** : milliers d'opérations simultanées
- **Idéal pour** : données indépendantes, opérations répétitives

**Analogie :** 10 000 élèves de primaire qui corrigent chacun une copie simple en même temps.

---

## Architecture GPU : La Hiérarchie CUDA

```
Grid (Grille)
└─ Blocks (Blocs)
   └─ Threads (Fils d'exécution)
      └─ Warps (groupes de 32 threads synchrones)
```

### Composants clés
- **Thread** : unité d'exécution élémentaire (1 ALU = 1 thread)
- **Block** : groupe de threads partageant cache et mémoire
- **Grid** : ensemble de tous les blocs lancés
- **Warp** : 32 threads exécutant strictement la même instruction

### Identification des threads
Chaque thread calcule son identifiant unique :
```c
int id = blockIdx.x * blockDim.x + threadIdx.x;
```

---

## Quand Utiliser le GPU ?

### Problèmes adaptés (Data Parallel)
- **Addition de vecteurs** : chaque élément calculé indépendamment
- **Traitement d'images** : 2 millions de pixels calculés en parallèle
- **Multiplications matricielles** : opérations indépendantes (IA, deep learning)
- **Force brute** : tester millions de combinaisons simultanément
- **Simulations** : particules, cellules atmosphériques

### Problèmes inadaptés (Séquentiels)
- **Suite de Fibonacci** : chaque terme dépend des précédents
- **Graphes avec dépendances** : les étapes doivent s'enchaîner
- **Petits calculs** : l'overhead de transfert annule le gain

---

## Workflow CUDA : Les 4 Étapes Obligatoires

### 1. Allocation (Host → Device)
```c
float *d_A;
cudaMalloc((void**)&d_A, size * sizeof(float));
```
Réserver de la mémoire sur le GPU (VRAM).

### 2. Transfert CPU → GPU
```c
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
```
⚠️ **Goulot d'étranglement** : transfert via bus PCIe lent.

### 3. Exécution du Kernel
```c
addVectors<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
```
Lancement de milliers de threads en parallèle.

### 4. Rapatriement GPU → CPU
```c
cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
cudaFree(d_C);
```
Récupération des résultats et libération mémoire.

---


## Applications Modernes du GPU

### Intelligence Artificielle
- Entraînement de réseaux neuronaux (GPT-3 : 175 milliards de paramètres)
- NVIDIA A100 : **312 TFLOPS** vs 1 TFLOPS pour un CPU

### Cryptomonnaies
- Minage Bitcoin : calcul SHA-256 massivement parallèle
- Milliards de hash testés par seconde

### Science & Ingénierie
- Simulations climatiques, astrophysique
- Rendu 3D (Pixar, studios d'animation)
- Biologie moléculaire (repliement de protéines)

### Traitement Multimédia
- Filtres en temps réel (Instagram, Snapchat)
- Vision par ordinateur (voitures autonomes)
- Compression vidéo H.264/H.265

---

Exemple Pratique : Casseur de Code PIN

Ce tutoriel vous guide pas à pas pour reproduire le casseur de code PIN avec deux approches : CPU et GPU. À la fin, vous lancerez le code et observerez la différence de performance.

## Sommaire
1. [Prérequis](#prérequis)
2. [Introduction](#introduction)
3. [Partie 1 : Kernel GPU](#partie-1--kernel-gpu)
4. [Partie 2 : Comparaison CPU et Configuration GPU](#partie-2--comparaison-cpu-et-configuration-gpu)
5. [Partie 3 : Gestion Mémoire et Lancement GPU](#partie-3--gestion-mémoire-et-lancement-gpu)
6. [Exécution](#exécution)
7. [Analyse des Résultats](#analyse-des-résultats)

---

## Prérequis

- Google Colab avec GPU activé (Runtime → Change runtime type → GPU)

---

## Introduction

Ce tutoriel présente la **programmation parallèle sur GPU avec CUDA** à travers un exemple concret : trouver un code PIN parmi **1 milliard de combinaisons** (0 à 999 999 999).

L'objectif est de comprendre comment le GPU peut résoudre des problèmes massivement parallèles en lançant des milliers de threads simultanément, contrairement au CPU qui traite les données séquentiellement. Nous comparerons les deux approches pour mesurer l'accélération apportée par le GPU.

Le principe : au lieu de tester les codes un par un, nous lançons **1 milliard de threads en parallèle**, chacun testant une combinaison différente simultanément.

---

## Partie 1 : Kernel GPU

```cuda
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void findPinKernel(int* d_resultat, int code_secret) {
    int mon_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (mon_id < 1000000000) {
        if (mon_id == code_secret) {
            *d_resultat = mon_id;
        }
    }
}
```

**Explication :**

Le **kernel** est une fonction qui s'exécute sur le GPU, marquée par le mot-clé `__global__`.

**Formule d'ID thread :** `mon_id = blockIdx.x * blockDim.x + threadIdx.x`
- `blockIdx.x` : numéro du bloc dans la grille
- `blockDim.x` : nombre de threads par bloc (1024)
- `threadIdx.x` : numéro du thread dans son bloc
- Résultat : chaque thread obtient un identifiant unique de 0 à ~1 milliard

**Logique :** Chaque thread teste UNE SEULE combinaison (son ID). Si le thread trouve le code secret, il l'écrit dans la mémoire GPU pointée par `d_resultat`. Tous les threads s'exécutent en parallèle.

---

## Partie 2 : Comparaison CPU et Configuration GPU

```cuda
int main() {
    const int SECRET_PIN = 999999999;
    const int NOMBRE_COMBINAISONS = 1000000000;

    printf("Objectif : Trouver le code PIN secret (%d) parmi %d combinaisons.\n", 
           SECRET_PIN, NOMBRE_COMBINAISONS);
    
    int resultat_cpu = -1;
    int resultat_gpu_host = -1;
    clock_t start, end;
    double temps_cpu, temps_gpu;

    // ========== VERSION CPU (pour comparaison) ==========
    printf("\nLancement de l'attaque CPU (séquentielle)...\n");
    start = clock();
    for (int i = 0; i < NOMBRE_COMBINAISONS; i++) {
        if (i == SECRET_PIN) {
            resultat_cpu = i;
            break;
        }
    }
    end = clock();
    temps_cpu = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CPU - Code trouvé : %d\n", resultat_cpu);
    printf("CPU - Temps : %f secondes (%d étapes)\n", temps_cpu, resultat_cpu + 1);

    // ========== CONFIGURATION GPU ==========
    printf("\nLancement de l'attaque GPU (parallèle)...\n");
    start = clock();
    
    int threadsPerBlock = 1024;
    int numBlocks = (NOMBRE_COMBINAISONS + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Configuration GPU: %d blocs × %d threads = %d threads totaux\n", 
           numBlocks, threadsPerBlock, numBlocks * threadsPerBlock);
```

**Explication :**

**Partie CPU :** On calcule d'abord avec le CPU en testant séquentiellement chaque combinaison dans une boucle `for`. Cette partie sert de **référence** pour mesurer l'accélération du GPU. Le CPU doit parcourir jusqu'à 1 milliard d'itérations.

**Configuration GPU :** 
- `threadsPerBlock = 1024` : nombre maximum de threads par bloc (limite matérielle des GPU)
- `numBlocks` : calculé pour couvrir toutes les combinaisons → 976 563 blocs
- **Architecture CUDA :** Les threads sont organisés en blocs, eux-mêmes organisés en grille
- Total : 976 563 blocs × 1024 threads/bloc = ~1 milliard de threads lancés en parallèle

---

## Partie 3 : Gestion Mémoire et Lancement GPU

```cuda
    // ========== ALLOCATION MÉMOIRE GPU ==========
    int* d_resultat;
    cudaError_t err = cudaMalloc((void**)&d_resultat, sizeof(int));
    if (err != cudaSuccess) {
        printf("Erreur cudaMalloc: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // ========== TRANSFERT CPU → GPU ==========
    err = cudaMemcpy(d_resultat, &resultat_gpu_host, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Erreur cudaMemcpy H2D: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // ========== LANCEMENT DU KERNEL ==========
    findPinKernel<<<numBlocks, threadsPerBlock>>>(d_resultat, SECRET_PIN);
    cudaDeviceSynchronize();

    // ========== TRANSFERT GPU → CPU ==========
    err = cudaMemcpy(&resultat_gpu_host, d_resultat, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Erreur cudaMemcpy D2H: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // ========== LIBÉRATION MÉMOIRE ==========
    cudaFree(d_resultat);

    end = clock();
    temps_gpu = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("GPU - Code trouvé : %d\n", resultat_gpu_host);
    printf("GPU - Temps : %f secondes (toutes combinaisons en parallèle)\n", temps_gpu);

    // ========== RÉSULTATS ==========
    printf("\n--- Conclusion ---\n");
    printf("Temps CPU: %f sec\n", temps_cpu);
    printf("Temps GPU: %f sec (inclut la préparation)\n", temps_gpu);
    
    if (temps_cpu > temps_gpu) {
        printf("ACCÉLÉRATION: Le GPU est %.2fx plus rapide !\n", temps_cpu / temps_gpu);
    }
    
    return 0;
}
```

**Explication :**

**Workflow CUDA en 5 étapes :**

1. **`cudaMalloc`** : Alloue de la mémoire sur le GPU (comme `malloc` mais pour le GPU). Le préfixe `d_` indique "device" (GPU).

2. **`cudaMemcpy(Host→Device)`** : Copie les données du CPU vers le GPU. La mémoire du GPU est séparée de la RAM, il faut transférer explicitement.

3. **`kernel<<<blocs, threads>>>()`** : Syntaxe spéciale CUDA pour lancer le kernel avec la configuration désirée. Les `<<<>>>` définissent combien de threads exécuteront le kernel.

4. **`cudaDeviceSynchronize()`** : Bloque le CPU jusqu'à ce que tous les threads GPU aient terminé leur exécution.

5. **`cudaMemcpy(Device→Host)`** : Récupère le résultat depuis le GPU vers le CPU, puis `cudaFree` libère la mémoire GPU.

**Point clé :** La mémoire GPU et CPU sont physiquement séparées. Toute donnée doit être explicitement transférée entre les deux.

---


## Exécution

**Dans Google Colab :**

Cellule 1 - Créer le fichier :
```python
%%writefile Casseur_de_Code_GPUvsCPU.cu
[Copier tout le code ci-dessus]
```

Cellule 2 - Compiler et exécuter :
```bash
!nvcc -o casseur Casseur_de_Code_GPUvsCPU.cu
!./casseur
```

Le programme affichera les temps d'exécution CPU vs GPU et le facteur d'accélération.


## Analyse des Résultats

### Ordre de grandeur observé
Le GPU est environ **15 à 20 fois plus rapide** que le CPU pour ce problème.

- **CPU** : plusieurs secondes (≈ 2–3 s)  
- **GPU** : une fraction de seconde (≈ 0.1–0.2 s)

### Pourquoi cette accélération ?
- Le **CPU** teste les combinaisons **une par une** → 1 milliard d’itérations séquentielles.  
- Le **GPU** lance **~1 milliard de threads**, chacun testant une combinaison **en parallèle**, dans une seule vague d’exécution.

### Cas limite important
Si l’on cherche un code plus petit (ex. `1000` au lieu de `999 999 999`), l’écart de performance se réduit fortement :

- Le **CPU** trouve le code en quelques millisecondes (arrêt précoce).
- Le **GPU** met toujours ≈ **0.12 s**, car il **exécute tous les threads**, même si la solution est trouvée très tôt.

---

## Points Clés à Retenir

### Avantages du GPU
- **Débit massif** : milliers d'opérations en parallèle
- **Efficacité énergétique** sur gros volumes
- **Accélération 10-100×** sur problèmes adaptés

### Limitations
- **Overhead de transfert** : pénalise les petits calculs
- **Problèmes séquentiels** : GPU moins efficace que CPU
- **Complexité** : programmation plus technique (gestion mémoire, synchronisation)
- **Divergence de branches** : tue les performances dans un warp

### Règle d'Or
Le GPU excelle quand :
1. Les données sont **indépendantes**
2. Le volume de calcul est **massif**
3. L'opération est **répétitive** sur beaucoup de données

---

**Par Elouan Lecorgne et Baptiste Krugler**

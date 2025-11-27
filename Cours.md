# Programmation GPU : La Puissance du Parallélisme au Service de l'Informatique

## Introduction

La programmation d’un GPU (Graphics Processing Unit) consiste à exploiter une carte graphique non plus seulement pour afficher des images, mais pour accélérer des calculs en masse grâce au parallélisme. Elle s’est imposée dans le calcul haute performance (High Performance Computing), l’intelligence artificielle (IA) et la simulation parce qu’elle permet de traiter rapidement des volumes de données et des opérations répétitives que les processeurs centraux (Central Processing Unit) classiques gèrent moins efficacement. L’idée directrice est simple : quand une tâche peut être découpée en milliers de petits calculs indépendants, un processeur graphique peut les exécuter simultanément et faire gagner un ordre de grandeur en temps et en efficacité.

### 1.1 Le pourquoi du GPU

À l'origine dédiée au rendu 3D, la carte graphique est devenue un véritable accélérateur de calcul, capable d'exécuter en parallèle des milliers de threads semblables sur de grandes quantités de données.

**Constat:** Les processeurs (CPU) sont excellents pour les tâches séquentielles et la réactivité, mais de plus en plus d'applications sont massivement parallèles: entraînement de modèles d'IA, traitement d'images et de signaux, simulations numériques, finance, bio-informatique. Miser sur le GPU, c'est profiter d'un meilleur débit de calcul, souvent avec un rapport performance/énergie plus favorable, dès que l'algorithme s'y prête.



### 1.2 Analogie clé

**Le CPU (Central Processing Unit): le PDG.** Le CPU coordonne, prend des décisions rapides et gère des tâches variées avec souplesse. Il excelle quand la logique est complexe, que les étapes dépendent fortement les unes des autres ou que la latence de réponse est critique.

**Le GPU (Graphics Processing Unit): l'armée.** Le GPU mobilise des milliers de "soldats" pour exécuter la même consigne sur des données différentes, en parallèle. Il brille dès qu'il faut "avancer en rang serré" sur des tableaux, des images, des matrices, ou des flux massifs, où le débit global prime sur la micro-optimisation d'un seul fil d'exécution.

---

## I. Le Concept Fondamental : Parallélisme Massif 

Pour comprendre CUDA, il faut cesser de penser comme un humain qui fait une tâche après l'autre (séquentiel) et commencer à penser comme une fourmilière (parallèle).

### 1.1 La Différence Structurelle

La différence entre un CPU et un GPU n'est pas une question de "puissance" brute, mais de philosophie de travail.

**Le CPU (Latency Oriented) :**
Imaginez un professeur de mathématiques brillant, capable de résoudre des équations complexes, de gérer des interruptions (téléphone qui sonne), et de passer d'une tâche à l'autre rapidement. Mais il est seul. S'il doit corriger 10 000 copies simples (additions basiques), il devra les faire une par une pendant des heures.

*Structure :* Peu de cœurs (4 à 32), mais très rapides et polyvalents. Chaque cœur peut exécuter des instructions complexes, gérer des branchements conditionnels, et accéder efficacement à une grande mémoire cache.

**Le GPU (Throughput Oriented) :**
Imaginez une armée de 10 000 élèves de primaire. Individuellement, ils sont lents et ne savent faire que des calculs simples (additions, multiplications). Ils sont incapables de gérer des tâches complexes ou imprévues. Mais ils sont nombreux. Si vous devez corriger 10 000 copies d'additions, chaque élève en prend une, et le travail est fini en quelques secondes.

*Structure :* Des milliers de cœurs simplifiés (2 000 à 10 000+), conçus pour exécuter la même instruction sur des données différentes simultanément. C'est le modèle SIMD (Single Instruction, Multiple Data).

### 1.2 Quand utiliser le GPU ?

Le GPU n'est pas magique et ne convient pas à tous les problèmes. Il faut l'utiliser uniquement quand les données sont **indépendantes** les unes des autres.

**Le problème séquentiel (Mauvais pour le GPU) :**

*Exemple :* La suite de Fibonacci \(U_n = U_{n-1} + U_{n-2}\).

*Pourquoi :* Pour calculer \(U_{10}\), il faut absolument avoir terminé \(U_9\) et \(U_8\). Les 10 000 threads doivent attendre le résultat précédent. Le GPU sera ici plus lent que le CPU car la majorité des threads resteront inactifs.

**Le problème "Data Parallel" (Idéal pour le GPU) :**

*Exemple :* L'addition de deux vecteurs ou le traitement d'image.

*Pourquoi :* Pour additionner l'élément 0 du vecteur A avec l'élément 0 du vecteur B, je n'ai pas besoin de connaître le résultat de l'addition des éléments 1, 2, 3, etc. Tous les threads peuvent travailler simultanément sans communication.

**Exemples concrets :**

- **Rendu d'image / Jeux Vidéo :** Calculer la couleur de 2 millions de pixels (résolution 1080p). Chaque pixel est calculé indépendamment des autres.
- **Entraînement d'IA :** Les réseaux de neurones effectuent des multiplications matricielles massives. Chaque élément de la matrice résultante peut être calculé en parallèle.
- **Casseur de mot de passe (Force Brute) :** Tester le code "0000" n'empêche pas un autre thread de tester "9999" simultanément.
- **Simulations scientifiques :** Calcul de forces entre particules, où chaque particule peut être traitée en parallèle.

### 1.3 Comment ça "Programme" ?

Programmer un GPU demande de gérer deux processeurs séparés qui communiquent. Dans le vocabulaire CUDA, nous utilisons deux termes clés :

- **Host (L'Hôte) :** C'est le CPU et sa mémoire RAM.
- **Device (Le Périphérique) :** C'est le GPU et sa mémoire globale (VRAM).

Le flux de travail est toujours le même, en **4 étapes obligatoires** :

**Étape 1 - Allocation & Préparation (Host) :**
Le CPU prépare les données dans la RAM. Il doit ensuite "réserver" de la place sur la mémoire du GPU avec la commande `cudaMalloc()`, similaire à `malloc()` mais pour le GPU.

float d_A; // Pointeur vers la mémoire GPU
cudaMalloc((void*)&d_A, size * sizeof(float));

text

**Étape 2 - Le Transfert (Le Goulot d'étranglement) :**
Le GPU ne peut pas lire directement la RAM de votre PC. Il faut copier les données du Host vers le Device via le bus PCIe. C'est le rôle de `cudaMemcpy()`.

cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);

text

⚠️ **Attention :** C'est l'étape la plus lente ! Si le calcul est trop court, le temps de transfert annulera complètement le gain de vitesse. C'est comme prendre l'avion pour aller à 10 km : le temps d'embarquement dépasse le temps de vol.

**Étape 3 - L'Exécution (Le Kernel) :**
Le CPU lance l'exécution sur le GPU avec une syntaxe spéciale : `fonction<<<blocs, threads>>>()`. La fonction exécutée sur le GPU s'appelle un **Kernel** et est déclarée avec `__global__`.

addVectors<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, size);

text

**Étape 4 - Le Rapatriement :**
Une fois terminé, les résultats sont bloqués dans la mémoire GPU. Il faut refaire un `cudaMemcpy()` (Device vers Host) pour les récupérer et les utiliser.

cudaMemcpy(h_C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

text

**En résumé :** On charge le camion (données), on l'envoie à l'usine (GPU), l'usine fabrique tout en parallèle, et on ramène le camion au magasin.

---

## II. Architecture CUDA : Grille, Blocs et Threads

### 2.1 La Hiérarchie à Trois Niveaux

CUDA organise les threads selon une hiérarchie à trois niveaux :

Grid (Grille)
└─ Blocks (Blocs)
└─ Threads (Fils d'exécution)

text

**Thread :** L'unité d'exécution la plus petite. C'est lui qui exécute réellement le code du kernel. Chaque thread possède son propre identifiant unique et ses propres registres.

**Block :** Un groupe de threads (typiquement 128, 256 ou 512 threads). Les threads d'un même bloc peuvent communiquer via la **shared memory** et se synchroniser avec `__syncthreads()`.

**Grid :** L'ensemble de tous les blocs lancés. Le grid peut être unidimensionnel, bidimensionnel ou tridimensionnel.

### 2.2 Identifiants Uniques

Chaque thread doit identifier quelle donnée il doit traiter. Pour cela, CUDA fournit des variables intégrées :

threadIdx.x // Index du thread dans son bloc (0 à blockDim.x-1)
blockIdx.x // Index du bloc dans la grille (0 à gridDim.x-1)
blockDim.x // Nombre de threads par bloc
gridDim.x // Nombre de blocs dans la grille

text

**Calcul de l'identifiant global :**

int id = blockIdx.x * blockDim.x + threadIdx.x;

text

**Exemple concret :** Si vous avez 4 blocs de 256 threads chacun :
- Bloc 0, Thread 0 → id = 0
- Bloc 0, Thread 255 → id = 255
- Bloc 1, Thread 0 → id = 256
- Bloc 3, Thread 255 → id = 1023

### 2.3 Les Warps : Le Secret de l'Efficacité

Les threads ne s'exécutent pas isolément. Ils sont regroupés en **warps** de 32 threads qui exécutent la même instruction simultanément.

**Conséquence importante :** Si dans un warp, certains threads prennent la branche `if` et d'autres la branche `else`, le GPU devra exécuter **les deux branches séquentiellement**, en désactivant temporairement certains threads. C'est la **divergence de branche**, qui tue les performances.

**Code à éviter (divergence) :**
if (threadIdx.x % 2 == 0) {
// Calcul lourd
} else {
// Calcul différent
}

---

## III. Applications Modernes

### 3.1 Intelligence Artificielle et Deep Learning

Les GPU sont devenus indispensables pour l'entraînement des réseaux de neurones. Un modèle comme GPT-3 contient 175 milliards de paramètres. L'entraînement nécessite des milliards de multiplications matricielles, opération parfaitement parallélisable. Sans GPU, l'entraînement de ces modèles prendrait des années au lieu de semaines.

**Exemple :** NVIDIA A100 peut effectuer 312 TFLOPS (téraflops) de calcul en précision simple, contre ~1 TFLOPS pour un CPU moderne.

### 3.2 Cryptomonnaies et Blockchain

Le minage de Bitcoin repose sur le calcul intensif de fonctions de hachage SHA-256. Chaque essai est indépendant des autres, ce qui rend le problème parfaitement parallélisable. Les mineurs utilisent aujourd'hui des GPU et des ASIC pour tester des milliards de combinaisons par seconde.

### 3.3 Science et Ingénierie

- **Simulations climatiques :** Calcul de l'évolution de millions de cellules atmosphériques en parallèle.
- **Biologie moléculaire :** Simulation du repliement de protéines (projet Folding@home).
- **Astrophysique :** Simulation de la formation des galaxies avec des milliards de particules.
- **Rendu 3D et Animation :** Pixar et autres studios utilisent des fermes de GPU pour le rendu de films d'animation.

### 3.4 Traitement d'Image et Vidéo

- **Filtres en temps réel :** Instagram, Snapchat appliquent des filtres sur des vidéos 1080p à 30 fps.
- **Vision par ordinateur :** Reconnaissance faciale, détection d'objets pour les voitures autonomes.
- **Compression vidéo :** Encodage H.264/H.265 accéléré par GPU.

---


### 4.1 Le Défi : casser un code à 4 chiffres

On veut retrouver un code PIN secret à 4 chiffres.  
Les codes possibles vont de 0000 à 9999, soit 10 000 combinaisons.

Objectif : comparer deux approches pour retrouver ce code le plus vite possible :

- une approche CPU (le “PDG” qui teste tout seul, dans l’ordre)
- une approche GPU (l’“armée” de milliers de threads qui testent tout en même temps)


### 4.2 La Méthode CPU (Le PDG)

Avec le CPU, on teste les combinaisons **une par une** :

- 0000 → 0001 → 0002 → … → 9999

Le code C correspondant est :

for (int i = 0; i < NOMBRE_COMBINAISONS; i++) {
if (i == SECRET_PIN) {
resultat_cpu = i;
break; // Trouvé ! On arrête.
}
}

text

- Le CPU ne fait qu’une seule tentative à la fois.
- Dans le pire cas : **10 000 tentatives séquentielles**.
- Dans notre exemple, si le code est 8342, il faut 8343 étapes.

On peut voir le CPU comme un PDG très intelligent, mais qui doit signer les 10 000 documents lui‑même, un par un.


### 4.3 La Méthode GPU (L’Armée)

Avec le GPU, on change complètement de stratégie :  
au lieu d’un seul PDG, on envoie **10 000 soldats en parallèle**.

Idée :

- le thread n°0 teste le code 0000
- le thread n°1 teste le code 0001
- …
- le thread n°4732 teste le code 4732
- …
- le thread n°9999 teste le code 9999

Tous les threads travaillent **en même temps**.

Le kernel CUDA ressemble à ceci :

global void findPinKernel(int* d_resultat, int code_secret) {
// Calcul du "numéro de soldat" global
int mon_id = blockIdx.x * blockDim.x + threadIdx.x;

text
// On s'assure qu'on reste dans l'intervalle 0–9999
if (mon_id < 10000) {
    // Si mon ID correspond au code secret, j'écris la réponse
    if (mon_id == code_secret) {
        *d_resultat = mon_id;
    }
}
}

text

Dans le `main`, on lance par exemple :

findPinKernel<<<10, 1000>>>(d_resultat, SECRET_PIN);

text

- 10 blocs × 1000 threads = **10 000 threads**
- Chaque thread teste **exactement une** combinaison

Conceptuellement :

- Tous les codes sont testés dans **une seule vague** de calcul.
- Dès qu’un thread trouve la bonne valeur, il écrit le résultat.

**Temps conceptuel :**

- CPU : jusqu’à 10 000 étapes l’une après l’autre.
- GPU : une seule “étape” où 10 000 essais sont faits en parallèle  
  (plus un petit temps de préparation : allocations, copies mémoire, etc.).

Impact visuel pour la présentation :

- CPU → une barre de progression qui avance petit à petit.
- GPU → 10 000 points qui travaillent en même temps, la réponse apparaît quasi instantanément.

Remarque importante pour les étudiants :

- Pour un petit problème (10 000 codes), le temps de préparation GPU (cudaMalloc, cudaMemcpy, etc.) peut être comparable ou plus lourd que le calcul.
- Mais si le code avait 8 ou 10 chiffres (des millions ou milliards de combinaisons), le CPU deviendrait rapidement trop lent, alors que le GPU pourrait exploiter son parallélisme massif.

---

## Conclusion (30 secondes) **_A définir_**




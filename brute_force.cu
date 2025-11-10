/*
 * LECTURE GPU : LE CASSEUR DE CODES
 * * Ce code trouve un PIN à 4 chiffres (0-9999) en utilisant deux méthodes :
 * 1. CPU (Séquentiel) : Une boucle for classique.
 * 2. GPU (Parallèle) : 10 000 threads CUDA qui testent tout en même temps.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h> // Pour le chronomètre

// ---
// PARTIE 2 : LE KERNEL (CODE POUR LE GPU)
// ---
// Cette fonction sera exécutée par CHAQUE "soldat" (thread) sur le GPU.
// Nous allons lancer 10 000 threads.
__global__ void findPinKernel(int* d_resultat, int code_secret) {
    
    /*
     * Calcul de l'ID global du thread (notre "numéro de soldat")
     * Nous lançons 10 "Blocks" (bataillons) de 1000 "Threads" (soldats)
     * blockIdx.x  = Le numéro de mon bataillon (0-9)
     * blockDim.x  = Le nombre de soldats par bataillon (1000)
     * threadIdx.x = Mon numéro de soldat DANS mon bataillon (0-999)
     *
     * Exemple pour le soldat #5 du bataillon #2 :
     * mon_id = (2 * 1000) + 5 = 2005
     */
    int mon_id = blockIdx.x * blockDim.x + threadIdx.x;

    // On s'assure qu'on ne dépasse pas 9999
    if (mon_id < 10000) {
        
        // L'ORDRE : Si MON ID est le bon, je l'écris dans la case "résultat".
        if (mon_id == code_secret) {
            *d_resultat = mon_id;
        }
    }
}


// ---
// PARTIE 1 : LE CODE HÔTE (CODE POUR LE CPU)
// ---
int main() {
    
    // Le code secret que nous devons trouver
    const int SECRET_PIN = 8342;
    const int NOMBRE_COMBINAISONS = 10000;

    printf("Objectif : Trouver le code PIN secret (%d) parmi %d combinaisons.\n", SECRET_PIN, NOMBRE_COMBINAISONS);
    
    // Variables pour les résultats
    int resultat_cpu = -1;
    int resultat_gpu_host = -1; // "Host" = La variable côté CPU
    
    // Variable pour le chronomètre
    clock_t start, end;
    double temps_cpu, temps_gpu;

    // ==============================================
    // Approche 1: Force Brute avec le CPU (Séquentiel)
    // ==============================================
    
    printf("\nLancement de l'attaque CPU (séquentielle)...\n");
    start = clock();

    for (int i = 0; i < NOMBRE_COMBINAISONS; i++) {
        if (i == SECRET_PIN) {
            resultat_cpu = i;
            break; // Trouvé ! On arrête.
        }
    }

    end = clock();
    temps_cpu = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CPU - Code trouvé : %d\n", resultat_cpu);
    printf("CPU - Temps : %f secondes (%d étapes)\n", temps_cpu, resultat_cpu + 1);


    // ==============================================
    // Approche 2: Force Brute avec le GPU (Parallèle)
    // ==============================================

    printf("\nLancement de l'attaque GPU (parallèle)...\n");
    start = clock();
    
    // --- Etape A : Préparation (Le Général) ---
    
    // Déclarer un pointeur pour la mémoire sur le GPU ("Device")
    int* d_resultat;

    // 1. Allouer 1 "int" d'espace mémoire sur le GPU
    // On vérifie s'il y a une erreur
    cudaError_t err = cudaMalloc((void**)&d_resultat, sizeof(int));
    if (err != cudaSuccess) {
        printf("Erreur cudaMalloc: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // 2. Copier notre valeur initiale (-1) du CPU vers le GPU
    // (cudaMemcpyHostToDevice = CPU -> GPU)
    err = cudaMemcpy(d_resultat, &resultat_gpu_host, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Erreur cudaMemcpy H2D: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // --- Etape B : Lancement (L'Armée) ---

    // 3. Lancer le Kernel !
    // On lance 10 Blocks, et 1000 Threads par Block.
    // Total = 10 * 1000 = 10 000 threads.
    findPinKernel<<<10, 1000>>>(d_resultat, SECRET_PIN);
    
    // Attendre que le GPU ait fini son travail
    cudaDeviceSynchronize(); 

    // --- Etape C : Récupération (Le Rapport) ---

    // 4. Copier le résultat du GPU vers le CPU
    // (cudaMemcpyDeviceToHost = GPU -> CPU)
    err = cudaMemcpy(&resultat_gpu_host, d_resultat, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Erreur cudaMemcpy D2H: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // 5. Libérer la mémoire sur le GPU
    cudaFree(d_resultat);

    // Fin du chrono GPU
    end = clock();
    temps_gpu = ((double) (end - start)) / CLOCKS_PER_SEC;

    // Afficher les résultats
    printf("GPU - Code trouvé : %d\n", resultat_gpu_host);
    printf("GPU - Temps : %f secondes (1 étape conceptuelle)\n", temps_gpu);

    // ==============================================
    // Conclusion
    // ==============================================
    
    printf("\n--- Conclusion ---\n");
    printf("Temps CPU: %f sec\n", temps_cpu);
    printf("Temps GPU: %f sec (inclut la préparation)\n", temps_gpu);
    
    // Note : Le temps GPU peut sembler plus long car le problème est TROP simple.
    // Le "coût de préparation" (cudaMalloc, cudaMemcpy) est plus élevé 
    // que le temps de calcul. Mais si le PIN avait 20 chiffres, 
    // le GPU gagnerait de manière écrasante.
    // L'important est le concept : 1 étape de calcul vs 8343.
    
    return 0;
}
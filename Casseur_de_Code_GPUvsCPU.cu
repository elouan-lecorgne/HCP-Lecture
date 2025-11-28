
%%writefile Casseur_de_Code_GPUvsCPU.cu


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
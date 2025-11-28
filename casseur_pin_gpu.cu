#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define NOMBRE_COMBINAISONS 10000
#define SECRET_PIN 8342

__global__ void findPinKernel(int* d_resultat, int code_secret) {
    int mon_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (mon_id < NOMBRE_COMBINAISONS) {
        if (mon_id == code_secret) {
            *d_resultat = mon_id;
        }
    }
}

int main() {
    int* d_resultat;
    int h_resultat = -1;
    
    // Allocation GPU
    cudaMalloc((void**)&d_resultat, sizeof(int));
    cudaMemset(d_resultat, -1, sizeof(int));
    
    clock_t debut = clock();
    
    // Lancement du kernel
    findPinKernel<<<10, 1000>>>(d_resultat, SECRET_PIN);
    
    // Synchronisation
    cudaDeviceSynchronize();
    
    clock_t fin = clock();
    double temps_gpu = ((double)(fin - debut)) / CLOCKS_PER_SEC;
    
    // Copie du résultat
    cudaMemcpy(&h_resultat, d_resultat, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("PIN trouvé par le GPU : %d\n", h_resultat);
    printf("Temps GPU : %.6f secondes\n", temps_gpu);
    
    // Libération mémoire GPU
    cudaFree(d_resultat);
    
    return 0;
}
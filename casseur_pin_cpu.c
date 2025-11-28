#include <stdio.h>
#include <time.h>

#define NOMBRE_COMBINAISONS 1000000
#define SECRET_PIN 999965564999

int main() {
    int resultat_cpu = -1;
    clock_t debut = clock();
    
    for (int i = 0; i < NOMBRE_COMBINAISONS; i++) {
        if (i == SECRET_PIN) {
            resultat_cpu = i;
            break;
        }
    }
    
    clock_t fin = clock();
    double temps_cpu = ((double)(fin - debut)) / CLOCKS_PER_SEC;
    
    printf("PIN trouvé par le CPU : %d\n", resultat_cpu);
    printf("Temps CPU : %.6f secondes\n", temps_cpu);
    
    return 0;
}
#include <sycl/sycl.hpp>

using namespace sycl;

#define MAX_WINDOW_SIZE 5
#define TILE_SIZE 16
#define WIND_SIZE 3

// Con todas las optimizaciones he conseguido un tiempo de ejcucion en ncu en mi rtx4070 max-q de 18 microsegundos. Aun queda un 50% 
// de posible optimizacion pero no se como mejorar mas el codigo

/*
He investigado algoritmos de ordenado optimizados para arrays cortos y he encontrado esto que funciona muy bien.
*/

// Red de ordenación óptima para 9 elementos (3x3)
void sort_9_elements(float arr[9]) {
    // Red de comparadores pre-optimizada para 9 elementos
    #define SWAP(i, j) if(arr[i] > arr[j]) { float tmp=arr[i]; arr[i]=arr[j]; arr[j]=tmp; }
    
    SWAP(0, 1); SWAP(3, 4); SWAP(6, 7);
    SWAP(1, 2); SWAP(4, 5); SWAP(7, 8);
    SWAP(0, 1); SWAP(3, 4); SWAP(6, 7);
    SWAP(0, 3); SWAP(3, 6); SWAP(1, 4);
    SWAP(4, 7); SWAP(2, 5); SWAP(5, 8);
    SWAP(1, 3); SWAP(2, 6); SWAP(2, 4);
    SWAP(4, 6); SWAP(5, 7); SWAP(2, 3);
    SWAP(4, 5);
    
    #undef SWAP
}

void remove_noise_SYCL(queue Q, float *im, float *image_out,
                      float threshold, int window_size,
                      int height, int width) {
    
    int ws2 = (window_size - 1) / 2;
    size_t rows = height;
    size_t cols = width;
    int tile_halo = TILE_SIZE + 2 * ws2;

    // Configurar rangos de ejecución
    auto global_range = nd_range<2>(
        {(rows + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE,
         (cols + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE},
        {TILE_SIZE, TILE_SIZE}
    );

    Q.submit([&](handler &h) {
        // Memoria local para el bloque con halo
        auto local = local_accessor<float, 1>(tile_halo * tile_halo, h);

        h.parallel_for(global_range, [=](nd_item<2> item) {
            // Coordenadas locales y globales
            const int lid_x = item.get_local_id(0);
            const int lid_y = item.get_local_id(1);
            const int gid_x = item.get_global_id(0);
            const int gid_y = item.get_global_id(1);
            const int lx = lid_x + ws2;
            const int ly = lid_y + ws2;

            //Precomputamos las comparaciones para evitar computarlas varias veces
            const int gx_n = gid_x > 0             ? gid_x-1 : 0;
            const int gx_p = gid_x < height-1      ? gid_x+1 : height-1;
            const int gy_n = gid_y > 0             ? gid_y-1 : 0;
            const int gy_p = gid_y < width-1       ? gid_y+1 : width-1;



            //Copiar los elementos a local
            local[lx * tile_halo + ly] = im[gid_x * width + gid_y];
            
            // Cargar bordes (halo)
            // Borde superior
            if (lid_x == 0)                 local[ly] = im[gx_n * width + gid_y];

            // Borde inferior
            if (lid_x == TILE_SIZE - 1)     local[(tile_halo - 1) * tile_halo + ly] = im[gx_p * width + gid_y];
            
            // Borde izquierdo
            if (lid_y == 0)                 local[lx * tile_halo] = im[gid_x * width + gy_n];
            
            // Borde derecho
            if (lid_y == TILE_SIZE - 1)     local[lx * tile_halo + tile_halo - 1] = im[gid_x * width + gy_p];
            
            // Esquinas (halo diagonal)
            if (lid_x == 0 && lid_y == 0)                           local[0] = im[gx_n * width + gy_n];
            
            if (lid_x == 0 && lid_y == TILE_SIZE - 1)               local[tile_halo - 1] = im[gx_n * width + gy_p];
            
            if (lid_x == TILE_SIZE - 1 && lid_y == 0)               local[(tile_halo - 1) * tile_halo] = im[gx_p * width + gy_n];
            
            if (lid_x == TILE_SIZE - 1 && lid_y == TILE_SIZE - 1)   local[(tile_halo - 1) * tile_halo + tile_halo - 1] = im[gx_p * width + gy_p];
            
            item.barrier(access::fence_space::local_space);

            // Copiamos y cerramos los hilos de los bordes para evitar mas calculos.
            if (gid_x < ws2 || gid_x >= height - ws2 || gid_y < ws2 || gid_y >= width - ws2) {
                image_out[gid_x*width + gid_y] = im[gid_x*width + gid_y];
                return;
            }

            float window[WIND_SIZE * WIND_SIZE];
            // Añadir los elementos a window (array local para procesamiento rapido)
            int idx = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    window[idx++] = local[(lx + dx) * tile_halo + ly + dy];
                }
            }

            // Aqui probe a usar un pragma unroll pero no obtuve ninguna mejora de rendimiento asi que lo he dejado asi.

			// Ordenación óptima para 9 elementos
            sort_9_elements(window);

            // Calcular mediana y aplicar umbral
            const float median = window[(window_size * window_size) / 2];
            const float original = im[gid_x * width + gid_y];
            const float ratio = fabs((median - original) / median);

            image_out[gid_x * width + gid_y] = (ratio <= threshold) ? original : median;
        });
    }).wait();
}
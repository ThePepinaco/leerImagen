#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define KERNEL_SIZE 9
#define KERNEL_RADIUS (KERNEL_SIZE / 2)
#define TAM_BLOQUE 16



int median(const std::vector<int>& v) {
    std::vector<int> sorted = v;
    std::sort(sorted.begin(), sorted.end());
    return sorted[v.size() / 2];
}

void applyMedianFilter(const unsigned char* input, unsigned char* output, int width, int height) {
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::vector<int> neighborhood;
            for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ky++) {
                for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; kx++) {
                    int nx = std::min(std::max(x + kx, 0), width - 1);
                    int ny = std::min(std::max(y + ky, 0), height - 1);
                    neighborhood.push_back(input[ny * width + nx]);
                }
            }

            std::sort(neighborhood.begin(), neighborhood.end());
            output[y * width + x] = neighborhood[neighborhood.size() / 2];
        }
    }
}

__global__ void applyMedianFilterCUDA(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int neighborhood[KERNEL_SIZE * KERNEL_SIZE];

        int startX = fmaxf(x - KERNEL_RADIUS, 0);
        int startY = fmaxf(y - KERNEL_RADIUS, 0);

        for (int ky = 0; ky < KERNEL_SIZE; ky++) {
            for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                int nx = fminf(startX + kx, width - 1);
                int ny = fminf(startY + ky, height - 1);
                neighborhood[ky * KERNEL_SIZE + kx] = input[ny * width + nx];
            }
        }

        for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE - 1; i++) {
            for (int j = 0; j < KERNEL_SIZE * KERNEL_SIZE - i - 1; j++) {
                if (neighborhood[j] > neighborhood[j + 1]) {
                    int temp = neighborhood[j];
                    neighborhood[j] = neighborhood[j + 1];
                    neighborhood[j + 1] = temp;
                }
            }
        }
        
        output[y * width + x] = neighborhood[KERNEL_SIZE * KERNEL_SIZE / 2];
     //   printf("Thread (%d, %d): Input[%d] = %d, Output[%d] = %d\n", x, y, y * width + x, input[y * width + x], y * width + x, output[y * width + x]);

    }
}



int main() {
    int width, height, channels;

    unsigned char* image = stbi_load("1.jpg", &width, &height, &channels, 0);

    if (!image) {
        std::cerr << "No se pudo abrir la imagen." << std::endl;
        return 1;
    }

    unsigned char* grayscaleImage = new unsigned char[width * height];
    // convertir a 1 canal
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = (y * width + x) * channels;
            int r = image[index];
            int g = image[index + 1];
            int b = image[index + 2];
            grayscaleImage[y * width + x] = (unsigned char)((r + g + b) / 3);
        }
    }

    // Definimos los punteros para el GPU
    unsigned char* d_input;
    unsigned char* d_output;

    cudaMalloc(&d_input, width * height * sizeof(unsigned char));
    cudaMalloc(&d_output, width * height * sizeof(unsigned char));

    // Copiamos los datos de la matriz al GPU
    cudaMemcpy(d_input, grayscaleImage, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    

    dim3 blockSize(TAM_BLOQUE, TAM_BLOQUE);
    dim3 gridSize((width + TAM_BLOQUE - 1) / TAM_BLOQUE, (height + TAM_BLOQUE - 1) / TAM_BLOQUE);


    //APLICAMOS EL FILTRO EN GPU
    auto startGPU = std::chrono::high_resolution_clock::now();
    applyMedianFilterCUDA <<<gridSize, blockSize >>> (d_input, d_output, width, height);
    auto endGPU = std::chrono::high_resolution_clock::now();

    unsigned char* MedianGPU = new unsigned char[width * height];

    cudaMemcpy(MedianGPU, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    std::chrono::duration<double> durationGPU = endGPU - startGPU;
    std::cout << "Tiempo de ejecución: " << durationGPU.count() << "ms GPU --- FILTRO MEDIAN BLUR" << std::endl;

    stbi_write_png("FiltroMedianBlurGPU.jpg", width, height, 1, MedianGPU, width);

    unsigned char* filteredImage = new unsigned char[width * height];

    // Aplicar filtro de mediana CPU
    auto start1 = std::chrono::high_resolution_clock::now();
    applyMedianFilter(grayscaleImage, filteredImage, width, height);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration1 = end1 - start1;

    std::cout << "Tiempo de ejecución: " << duration1.count() << "ms  CPU --- FILTRO MEDIAN BLUR" << std::endl;

    // Guardar la imagen filtrada
    stbi_write_png("FiltroMedianBlurCPU.jpg", width, height, 1, filteredImage, width);

    // Liberar memoria
    stbi_image_free(image);
    delete[] grayscaleImage;
    delete[] filteredImage;

    return 0;
}

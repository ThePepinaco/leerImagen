#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define FILTER_SIZE 9

using namespace std;

vector<vector<int>>  crearMascarasSobel(int tamano) {
    int mitad = tamano / 2;
    vector<vector<int>> mascaraX;
    mascaraX.resize(tamano, vector<int>(tamano, 0));

    for (int y = 0; y < tamano; y++) {
        for (int x = 0; x < tamano; x++) {
            mascaraX[y][x] = x - mitad;
        }
    }
    return mascaraX;
}

void aplicarFiltro(const unsigned char* entrada, unsigned char* salida, int width, int height, const std::vector<std::vector<int>>& mascara) {
    int radius = mascara.size() / 2;

    for (int y = radius; y < height - radius; y++) {
        for (int x = radius; x < width - radius; x++) {
            int sum = 0;

            for (int i = -radius; i <= radius; i++) {
                for (int j = -radius; j <= radius; j++) {
                    int pixelX = x + j;
                    int pixelY = y + i;
                    int index = (pixelY * width + pixelX);

                    sum += entrada[index] * mascara[i + radius][j + radius];
                }
            }

            salida[y * width + x] = static_cast<unsigned char>(std::abs(sum)); // Se toma el valor absoluto del resultado
        }
    }
}

__constant__ int d_M[FILTER_SIZE][FILTER_SIZE];

__global__ void sobelFilter(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int sum = 0;
        int radius = FILTER_SIZE / 2;

        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                int pixelX = x + j;
                int pixelY = y + i;
                int index = pixelY * width + pixelX;

                sum += input[index] * d_M[i + radius][j + radius];
            }
        }
        output[y * width + x] = abs(sum); // Se toma el valor absoluto del resultado
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

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Calculate grayscale value
            int index = (y * width + x) * channels;
            int r = image[index];
            int g = image[index + 1];
            int b = image[index + 2];
            grayscaleImage[y * width + x] = (unsigned char)((r + g + b) / 3);
        }
    }

    vector<vector<int>> mascaraX = crearMascarasSobel(FILTER_SIZE);


    vector<int> mascaraGPUX;
    for (int i = 0; i < FILTER_SIZE; ++i) {
        for (int j = 0; j < FILTER_SIZE; ++j) {
            mascaraGPUX.push_back(mascaraX[i][j]);
        }
    }

    cudaMemcpyToSymbol(d_M, &mascaraGPUX[0], FILTER_SIZE * FILTER_SIZE * sizeof(int));


    unsigned char* d_input, * d_output;
    cudaMalloc(&d_input, width * height * sizeof(unsigned char));
    cudaMalloc(&d_output, width * height * sizeof(unsigned char));

    cudaMemcpy(d_input, grayscaleImage, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 numBlocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);


    int mascara_GPU[FILTER_SIZE][FILTER_SIZE];
    cudaMemcpyFromSymbol(mascara_GPU, d_M, FILTER_SIZE * FILTER_SIZE * sizeof(int));


    auto startGPU = std::chrono::high_resolution_clock::now();
    sobelFilter <<<numBlocks, threads >> > (d_input, d_output, width, height);
    cudaDeviceSynchronize();
    auto endGPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationGPU = endGPU - startGPU;
    std::cout << "Tiempo de ejecución: " << durationGPU.count() << "ms  GPU --- FILTRO SOBEL X" << std::endl;

    unsigned char* outputImageGPU = new unsigned char[width * height];
    cudaMemcpy(outputImageGPU, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    stbi_write_png("FiltroSobelXGPU.jpg", width, height, 1, outputImageGPU, width);


    // guardaremos la imagen
    unsigned char* outputImageCPU = new unsigned char[width * height];
    // Aplicar el filtro 
    auto start1 = std::chrono::high_resolution_clock::now();

    aplicarFiltro(grayscaleImage, outputImageCPU, width, height, mascaraX);

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration1 = end1 - start1;
    std::cout << "Tiempo de ejecución: " << duration1.count() << "ms  CPU --- FILTRO SOBEL X" << std::endl;

    stbi_write_png("FiltroSobelXCPU.jpg", width, height, 1, outputImageCPU, width);


    stbi_image_free(image);
    cudaFree(d_input);
    cudaFree(d_output);

    delete[] grayscaleImage;
    delete[] outputImageCPU;

    return 0;
}

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define FILTER_SIZE 5

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

    // Crear el kernel de Sobel
    unsigned char* outputImageCPU = new unsigned char[width * height];
    // Aplicar el filtro de Sobel en la dirección X
    auto start1 = std::chrono::high_resolution_clock::now();

    aplicarFiltro(grayscaleImage, outputImageCPU, width, height, mascaraX);

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration1 = end1 - start1;
    std::cout << "Tiempo de ejecución: " << duration1.count() << "ms  CPU --- FILTRO SOBEL CUSTOM" << std::endl;
    // Guardar la imagen de salida
    stbi_write_png("FiltroSobelCustomCPU.jpg", width, height, 1, outputImageCPU, width);

    // Liberar memoria
    stbi_image_free(image);
    delete[] grayscaleImage;
    delete[] outputImageCPU;

    return 0;
}

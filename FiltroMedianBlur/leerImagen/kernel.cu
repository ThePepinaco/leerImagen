#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define KERNEL_SIZE 3
#define KERNEL_RADIUS (KERNEL_SIZE / 2)

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

int main() {
    int width, height, channels;

    unsigned char* image = stbi_load("images.jpg", &width, &height, &channels, 0);

    if (!image) {
        std::cerr << "No se pudo abrir la imagen." << std::endl;
        return 1;
    }

    unsigned char* grayscaleImage = new unsigned char[width * height];

    // Convertir a escala de grises
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

    unsigned char* filteredImage = new unsigned char[width * height];

    // Aplicar filtro de mediana
    auto start = std::chrono::steady_clock::now();
    applyMedianFilter(grayscaleImage, filteredImage, width, height);
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsedSeconds = end - start;
    std::cout << "Tiempo de procesamiento en la CPU: " << elapsedSeconds.count() << " segundos." << std::endl;

    // Guardar la imagen filtrada
    stbi_write_png("medianBlurCPU.jpg", width, height, 1, filteredImage, width);

    // Liberar memoria
    stbi_image_free(image);
    delete[] grayscaleImage;
    delete[] filteredImage;

    return 0;
}

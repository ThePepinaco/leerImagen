#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"



#define FILTER_SIZE 21
#define FILTER_RADIUS (FILTER_SIZE / 2)
#define SIGMA 100.0f
// Tamaño y sigma del kernel


std::vector<std::vector<float>> createGaussianBlurKernel(int size, float sigma) {
    std::vector<std::vector<float>> kernel(size, std::vector<float>(size, 0.0f));
    float sum = 0.0f;
    int half = size / 2;

    for (int i = -half; i <= half; ++i) {
        for (int j = -half; j <= half; ++j) {
            float value = exp(-(i * i + j * j) / (2 * sigma * sigma));
            kernel[i + half][j + half] = value;
            sum += value;
        }
    }
    // Normalizar el kernel
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            kernel[i][j] /= sum;
        }
    }

    return kernel;
}

void gaussianBlurCPU(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, const std::vector<std::vector<float>>& kernel) {
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            float value = 0.0f;

            for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; ++i) {
                for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; ++j) {
                    int offsetX = x + j;
                    int offsetY = y + i;

                    if (offsetX >= 0 && offsetX < width && offsetY >= 0 && offsetY < height) {
                        float weight = kernel[i + FILTER_RADIUS][j + FILTER_RADIUS];
                        value += weight * inputImage[offsetY * width + offsetX];
                        sum += weight;
                    }
                }
            }

            outputImage[y * width + x] = static_cast<unsigned char>(value / sum);
        }
    }
}

__global__ void gaussianBlurKernelCUDA(const unsigned char* inputImage, unsigned char* outputImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        float value = 0.0f;

        // aplicar el filtro usando las posiciones x e y
        for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; ++i) {
            for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; ++j) {
                int offsetX = x + j;
                int offsetY = y + i;

                // verificar si no estamos en los bordes
                if (offsetX >= 0 && offsetX < width && offsetY >= 0 && offsetY < height) {
                    float weight = expf(-(i * i + j * j) / (2.0f * FILTER_RADIUS * FILTER_RADIUS));
                    sum += weight;
                    value += weight * inputImage[offsetY * width + offsetX];
                }
            }
        }

        outputImage[y * width + x] = static_cast<unsigned char>(value / sum);
    }
}

int main() {
    int width, height, channels;

    unsigned char* image = stbi_load("1.jpg", &width, &height, &channels, 0);

    if (!image) {
        std::cerr << "No se pudo abrir la imagen." << std::endl;
        return 1;
    }
    std::vector<std::vector<float>> gaussianBlurKernel = createGaussianBlurKernel(FILTER_SIZE, SIGMA);
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

    size_t imageSize = width * height * sizeof(unsigned char);





    unsigned char* d_inputImage, * d_outputImage;
    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImage, imageSize);

    cudaMemcpy(d_inputImage, grayscaleImage, imageSize, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    auto start = std::chrono::high_resolution_clock::now();

    gaussianBlurKernelCUDA << <gridDim, blockDim >> > (d_inputImage, d_outputImage, width, height);

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Tiempo de ejecución: " << duration.count() << "ms  GPU --- FILTRO GAUSS BLUR" << std::endl;

    unsigned char* blurredImage = new unsigned char[width * height];
    cudaMemcpy(blurredImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    stbi_write_png("FiltroGaussBlurGPU.jpg", width, height, 1, blurredImage, width);
    






    unsigned char* blurredImage1 = new unsigned char[width * height];

    auto start1 = std::chrono::high_resolution_clock::now();

    // Aplicar el filtro de desenfoque gaussiano en la CPU
    gaussianBlurCPU(grayscaleImage, blurredImage1, width, height, gaussianBlurKernel);

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration1 = end1 - start1;

    std::cout << "Tiempo de ejecución: " << duration1.count() << "ms  CPU --- FILTRO GAUSS BLUR" << std::endl;

    // Guardar la imagen desenfocada
    stbi_write_png("FiltroGaussBlurCPU.jpg", width, height, 1, blurredImage1, width);




    delete[] grayscaleImage;
    delete[] blurredImage;
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    return 0;
}

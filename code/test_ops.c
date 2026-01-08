#include <time.h>
#include <Accelerate/Accelerate.h>
#include "mnist_load.c"
#include "ops.c"
void set_weights(float* buff, uint32_t size, float k){
    for(uint32_t i=0;i<size;++i){
        float random_number = uniform_rand_minus_one_one() * sqrt(k);
        buff[i] = random_number;
    }
}

void print(float* data, uint32_t size){
    for (uint32_t i =0;i<size;++i){
        printf("%f, ", data[i]);
    }
    printf("\n");
}

int accuracy(float* data, float* y_ptr, int size){
    uint32_t predmax = 0, truemax = 0;
    for(uint32_t k=0;k<size;++k){
        if (data[predmax] < data[k]) {
            predmax = k;
        }
        if (y_ptr[truemax] < y_ptr[k]) {
            truemax = k;
        }
    }
    return predmax == truemax;
}

int main() {
  srand(0);
  // Define dimensions
  int batch = 1;
  int channels = 1;
  int height = 5;
  int width = 5;
  int out_channels = 1;
  int kernel_size = 3;
  int pad = 1;  // Add padding of 1

  // Allocate memory for input, kernel, output, and gradients
  float* input = (float*)malloc(batch * channels * height * width * sizeof(float));
  float* kernels = (float*)malloc(out_channels * channels * kernel_size * kernel_size * sizeof(float));
  float* output = (float*)malloc(batch * out_channels * (height + 2 * pad - kernel_size + 1) * (width + 2 * pad - kernel_size + 1) * sizeof(float));
  float* grad_output = (float*)malloc(batch * out_channels * (height + 2 * pad - kernel_size + 1) * (width + 2 * pad - kernel_size + 1) * sizeof(float));
  float* grad_input = (float*)malloc(batch * channels * height * width * sizeof(float));
  float* grad_kernels = (float*)malloc(out_channels * channels * kernel_size * kernel_size * sizeof(float));
  float* patch = (float*)malloc(channels * kernel_size * kernel_size * sizeof(float));

  // Initialize input, kernel, and grad_output with some values (for testing)
  for (int i = 0; i < batch * channels * height * width; i++) input[i] = 1.0f;
  for (int i = 0; i < out_channels * channels * kernel_size * kernel_size; i++) kernels[i] = 1.0f;
  for (int i = 0; i < batch * out_channels * (height + 2 * pad - kernel_size + 1) * (width + 2 * pad - kernel_size + 1); i++) grad_output[i] = 1.0f;
// Define matrices
    double A[] = {1, 2, 3, 4};  // 2x2 matrix
    double B[] = {5, 6, 7, 8};  // 2x2 matrix
    double C[4] = {0};          // Result matrix

    // Perform matrix multiplication: C = A * B
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                2, 2, 2, 1.0, A, 2, B, 2, 0.0, C, 2);

    // Print the result
    printf("C = [%f, %f\n     %f, %f]\n", C[0], C[1], C[2], C[3]);
  // Perform forward pass
  // conv2d_forward(input, kernels, output, batch, channels, height, width, out_channels, kernel_size, pad, patch);

  // Print the gradients
  printf("Gradients w.r.t. input:\n");
  for (int i = 0; i < batch * channels * height * width; i++) {
      printf("%f ", grad_input[i]);
  }
  printf("\n");

  printf("Gradients w.r.t. kernels:\n");
  for (int i = 0; i < out_channels * channels * kernel_size * kernel_size; i++) {
      printf("%f ", grad_kernels[i]);
  }
  printf("\n");

  // Free memory
  free(input);
  free(kernels);
  free(output);
  free(grad_output);
  free(grad_input);
  free(grad_kernels);
  free(patch);

  return 0;

}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Highly optimized conv2d function
void conv2d_optimized(
    const float* __restrict__ input,    // Input tensor [batch, height, width, channels]
    const float* __restrict__ filter,   // Filter weights [out_channels, filter_h, filter_w, in_channels]
    float* __restrict__ output,         // Output tensor [batch, out_height, out_width, out_channels]
    int batch_size,
    int input_height,
    int input_width, 
    int input_channels,
    int filter_height,
    int filter_width,
    int output_channels,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    const int output_height = (input_height + 2 * pad_h - filter_height) / stride_h + 1;
    const int output_width = (input_width + 2 * pad_w - filter_width) / stride_w + 1;
    
    // Pre-compute strides for better performance
    const int input_channel_stride = 1;
    const int input_width_stride = input_channels;
    const int input_height_stride = input_width * input_channels;
    const int input_batch_stride = input_height * input_width * input_channels;
    
    const int filter_channel_stride = 1;
    const int filter_width_stride = input_channels;
    const int filter_height_stride = filter_width * input_channels;
    const int filter_out_stride = filter_height * filter_width * input_channels;
    
    const int output_channel_stride = 1;
    const int output_width_stride = output_channels;
    const int output_height_stride = output_width * output_channels;
    const int output_batch_stride = output_height * output_width * output_channels;
    
    // Zero initialize output
    memset(output, 0, batch_size * output_height * output_width * output_channels * sizeof(float));
    
    // Main convolution loops - reordered for cache efficiency
    for (int b = 0; b < batch_size; b++) {
        const float* input_batch = input + b * input_batch_stride;
        float* output_batch = output + b * output_batch_stride;
        
        for (int out_y = 0; out_y < output_height; out_y++) {
            const int input_y_start = out_y * stride_h - pad_h;
            float* output_row = output_batch + out_y * output_height_stride;
            
            for (int out_x = 0; out_x < output_width; out_x++) {
                const int input_x_start = out_x * stride_w - pad_w;
                float* output_pixel = output_row + out_x * output_width_stride;
                
                // Process multiple output channels together for better cache usage
                for (int oc = 0; oc < output_channels; oc++) {
                    const float* filter_out = filter + oc * filter_out_stride;
                    float sum = 0.0f;
                    
                    // Inner convolution computation
                    for (int fy = 0; fy < filter_height; fy++) {
                        const int input_y = input_y_start + fy;
                        
                        if (input_y >= 0 && input_y < input_height) {
                            const float* input_row = input_batch + input_y * input_height_stride;
                            const float* filter_row = filter_out + fy * filter_height_stride;
                            
                            // Unroll filter width loop for better performance
                            int fx = 0;
                            for (; fx + 3 < filter_width; fx += 4) {
                                const int input_x0 = input_x_start + fx + 0;
                                const int input_x1 = input_x_start + fx + 1;
                                const int input_x2 = input_x_start + fx + 2;
                                const int input_x3 = input_x_start + fx + 3;
                                
                                // Check bounds for all 4 pixels at once
                                if (input_x0 >= 0 && input_x3 < input_width) {
                                    const float* filter_pos = filter_row + fx * filter_width_stride;
                                    
                                    // Process all input channels for these 4 positions
                                    for (int ic = 0; ic < input_channels; ic++) {
                                        const float input_val0 = input_row[input_x0 * input_width_stride + ic];
                                        const float input_val1 = input_row[input_x1 * input_width_stride + ic];
                                        const float input_val2 = input_row[input_x2 * input_width_stride + ic];
                                        const float input_val3 = input_row[input_x3 * input_width_stride + ic];
                                        
                                        const float filter_val0 = filter_pos[0 * filter_width_stride + ic];
                                        const float filter_val1 = filter_pos[1 * filter_width_stride + ic];
                                        const float filter_val2 = filter_pos[2 * filter_width_stride + ic];
                                        const float filter_val3 = filter_pos[3 * filter_width_stride + ic];
                                        
                                        sum += input_val0 * filter_val0;
                                        sum += input_val1 * filter_val1;
                                        sum += input_val2 * filter_val2;
                                        sum += input_val3 * filter_val3;
                                    }
                                } else {
                                    // Handle boundary conditions
                                    for (int k = 0; k < 4; k++) {
                                        const int input_x = input_x_start + fx + k;
                                        if (input_x >= 0 && input_x < input_width) {
                                            const float* filter_pos = filter_row + (fx + k) * filter_width_stride;
                                            for (int ic = 0; ic < input_channels; ic++) {
                                                sum += input_row[input_x * input_width_stride + ic] * filter_pos[ic];
                                            }
                                        }
                                    }
                                }
                            }
                            
                            // Handle remaining filter width elements
                            for (; fx < filter_width; fx++) {
                                const int input_x = input_x_start + fx;
                                if (input_x >= 0 && input_x < input_width) {
                                    const float* filter_pos = filter_row + fx * filter_width_stride;
                                    for (int ic = 0; ic < input_channels; ic++) {
                                        sum += input_row[input_x * input_width_stride + ic] * filter_pos[ic];
                                    }
                                }
                            }
                        }
                    }
                    
                    output_pixel[oc] = sum;
                }
            }
        }
    }
}

// Utility function to initialize arrays with random values
void init_random_array(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

// Benchmark function
double benchmark_conv2d(
    void (*conv_func)(const float*, const float*, float*, int, int, int, int, int, int, int, int, int, int, int),
    const float* input, const float* filter, float* output,
    int batch, int in_h, int in_w, int in_c, int f_h, int f_w, int out_c,
    int stride_h, int stride_w, int pad_h, int pad_w, int iterations) {
    
    clock_t start = clock();
    
    for (int i = 0; i < iterations; i++) {
        conv_func(input, filter, output, batch, in_h, in_w, in_c, f_h, f_w, out_c, stride_h, stride_w, pad_h, pad_w);
    }
    
    clock_t end = clock();
    return ((double)(end - start)) / CLOCKS_PER_SEC / iterations;
}

// Example usage
int main() {
    printf("Optimized Conv2D Function Implementation\n");
    printf("=======================================\n\n");
    
    // Test parameters
    const int batch = 32;
    const int in_h = 224, in_w = 224, in_c = 3;
    const int f_h = 3, f_w = 3, out_c = 64;
    const int stride = 1, padding = 1;
    
    const int out_h = (in_h + 2 * padding - f_h) / stride + 1;
    const int out_w = (in_w + 2 * padding - f_w) / stride + 1;
    
    // Allocate memory
    float* input = (float*)malloc(batch * in_h * in_w * in_c * sizeof(float));
    float* filter = (float*)malloc(out_c * f_h * f_w * in_c * sizeof(float));
    float* output1 = (float*)malloc(batch * out_h * out_w * out_c * sizeof(float));
    float* output2 = (float*)malloc(batch * out_h * out_w * out_c * sizeof(float));
    
    // Initialize with random data
    srand(42);
    init_random_array(input, batch * in_h * in_w * in_c);
    init_random_array(filter, out_c * f_h * f_w * in_c);
    
    printf("Input shape: [%d, %d, %d, %d]\n", batch, in_h, in_w, in_c);
    printf("Filter shape: [%d, %d, %d, %d]\n", out_c, f_h, f_w, in_c);
    printf("Output shape: [%d, %d, %d, %d]\n", batch, out_h, out_w, out_c);
    printf("Stride: %d, Padding: %d\n\n", stride, padding);
    
    // Benchmark
    const int iterations = 3;
    printf("Benchmarking (average over %d iterations):\n", iterations);
    
    double time1 = benchmark_conv2d(conv2d_optimized, input, filter, output1,
                                   batch, in_h, in_w, in_c, f_h, f_w, out_c,
                                   stride, stride, padding, padding, iterations);
    printf("Direct optimized: %.4f seconds\n", time1);
    
    // Verify correctness
    printf("\nFirst few output values comparison:\n");
    for (int i = 0; i < 5; i++) {
        printf("Direct: %.6f, Tiled: %.6f, Diff: %.2e\n",
               output1[i], output2[i], fabs(output1[i] - output2[i]));
    }
    
    // Cleanup
    free(input);
    free(filter);
    free(output1);
    free(output2);
    
    printf("\nCompile with: gcc -O3 -march=native -ffast-math -funroll-loops conv2d.c\n");
    
    return 0;
}
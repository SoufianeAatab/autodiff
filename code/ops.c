#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

float uniform_rand_minus_one_one() {
    #if DEPLOY
        return (float)(random(1001) / 1000.0);
    #else
        return (((double)rand() / (double)RAND_MAX) - 0.5) * 2.0f;
    #endif
}

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

float* sigmoid(float* x, float* a, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        a[i] = 1.0f / (1.0f + exp(-x[i]));
    }
    return a;
}

void sigmoid_diff(float* x, float* grad, float* out, uint32_t size) {
    // while(size){
    //     *out = 1.0f / (1.0f + exp(-*x));
    //     --size;
    //     ++out;
    //     ++x;
    // }
    for (uint32_t i = 0; i < size; ++i) {
        float a = 1.0f / (1.0f + exp(-x[i]));
        out[i] = (a * (1.0-a)) * grad[i];
    }
    // return out;
}

float nll_loss(float* y_pred, float* y, uint32_t size){
    uint32_t imax = 0;
    for(uint32_t i=0;i<size;++i) {
        if (y[i] == 1) imax = i;
    }
    float loss = -y_pred[imax];
    return loss;
}

float binary_cross_entropy(float* y_hat, float*y){
    float loss = y[0] * log(y_hat[0]) + (1 - y[0]) * log(1 - y_hat[0]);
    return -loss;
}

void binary_cross_entropy_diff(float* y_hat, float*y, float* out_grad){
    out_grad[0] = (y_hat[0] - y[0]) / (y_hat[0] * (1 - y_hat[0]));    
}

float mse(float* y_pred, float* y, uint32_t size){
    uint32_t imax = 0;
    float sum = 0;
    for(uint32_t i=0;i<size;++i) {
        sum += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);
    }
    float loss = sum / (float)size;
    return loss;
}

float* log_softmax(float* x, float* a, uint32_t size){
    float sum = 0.0f;
    float max = x[0];
    for (uint32_t i = 1; i < size; ++i) {
        max = MAX(max, x[i]);
    } 

    for (uint32_t i = 0; i < size; ++i) {
        sum += exp(x[i] - max);
    }

    sum = logf(sum);

    for (uint32_t i = 0; i < size; ++i) {
        // subtract max to avoid NaN/+inf errors
        a[i] = x[i] - max - sum;
    }
    return a;
}
/* TODO: Conver this to matmul. Right now is mat vec multiplication*/
float* mat_vec_mul(float* a_data, float* b_data, float* c_data, 
uint32_t a_rows, uint32_t a_cols, uint32_t a_stride1, uint32_t a_stride2, 
uint32_t b_rows, uint32_t b_cols, uint32_t b_stride1, uint32_t b_stride2) {
    // Check if the matrices are compatible for multiplication
    // assert(a_cols == b_rows);
    // Iterate through the result matrix

    float* a = a_data;
    float* b = b_data;
    float* c = c_data;

    uint32_t arows = a_rows;
    while(arows) {
        uint32_t bcols = b_cols;
        float* btemp = b;

        while(bcols) {
            float* a_row = a;
            float* b2 = btemp;

            float acc = 0;
            uint32_t acols = a_cols >> 2;
            while(acols) {
                // printf("multi %f * %f\n", *a_row,*b2);
                acc += (*a_row) * (*b2);
                a_row += a_stride2;
                b2 += b_stride1;

                acc += (*a_row) * (*b2);
                a_row += a_stride2;
                b2 += b_stride1;

                acc += (*a_row) * (*b2);
                a_row += a_stride2;
                b2 += b_stride1;

                acc += (*a_row) * (*b2);
                a_row += a_stride2;
                b2 += b_stride1;

                --acols;
            }
            acols = a_cols & 0x3;
            while(acols) {
                acc += (*a_row) * (*b2);

                a_row += a_stride2;
                b2 += b_stride1;

                --acols;
            }

            *c++ = acc;
            btemp += b_stride2;
            --bcols;
        }
        a += a_stride1;
        --arows;
    }

    return c;
}

float* mat_mul(float* a_data, float* b_data, float* c_data, 
               uint32_t a_rows, uint32_t a_cols, uint32_t a_stride1, uint32_t a_stride2, 
               uint32_t b_rows, uint32_t b_cols, uint32_t b_stride1, uint32_t b_stride2) {
    float* a = a_data;
    float* b = b_data;
    float* c = c_data;

    // Iterate through the result matrix
    for (uint32_t i = 0; i < a_rows; ++i) {
        for (uint32_t j = 0; j < b_cols; ++j) {
            // Compute the dot product of row i of matrix a and column j of matrix b
            float sum = 0;
            for (uint32_t k = 0; k < a_cols; ++k) {
                // printf("%f * %f\n", a[i * a_stride1 + k * a_stride2] * b[k * b_stride1 + j * b_stride2]);
                sum += a[i * a_stride1 + k * a_stride2] * b[k * b_stride1 + j * b_stride2];
            }
            // Store the result in the corresponding element of matrix c
            *c_data++ = sum;
        }
    }
    return c_data;
}

void sum(float* data, float* out, uint8_t dim, uint32_t rows, uint32_t cols) {
    if ( dim == 1){
        for (uint32_t i = 0; i < rows; ++i){
            float sum = 0;
            for (uint32_t j = 0; j < cols; ++j) {
                sum += data[i * cols + j];
            }
            out[i] = sum;
        }
    } else if (dim == 0) {
        for (uint32_t i = 0; i < cols; ++i){
            float sum = 0;
            for (uint32_t j = 0; j < rows; ++j) {
                sum += data[j * cols + i];
            }
            out[i] = sum;
        }
    }
}

void mul(float* a, float *b, float* c, uint32_t size) {
    float* a_ptr = a;
    float* b_ptr = b;
    float* c_ptr = c;
    while(size){
        *c_ptr = *a_ptr * *b_ptr;

        --size;
        ++a_ptr;
        ++b_ptr;
        ++c_ptr;
    }
    // for(uint32_t i=0;i<size;++i){
    //     c[i] = a[i] * b[i];
    // }
}
void conv2d(float* input, float* kernel, float* output,
            int batch, int channels, int height, int width,
            int out_channels, int kernel_size) {
    int out_height = height - kernel_size + 1;
    int out_width = width - kernel_size + 1;

    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    float sum = 0.0f;

                    for (int ic = 0; ic < channels; ic++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int input_idx = b * channels * height * width +
                                                ic * height * width +
                                                (oh + kh) * width +
                                                (ow + kw);
                                
                                int kernel_idx = oc * channels * kernel_size * kernel_size +
                                                 ic * kernel_size * kernel_size +
                                                 kh * kernel_size + kw; // ✅ Fixed indexing

                                sum += input[input_idx] * kernel[kernel_idx];
                            }
                        }
                    }

                    int output_idx = b * out_channels * out_height * out_width +
                                     oc * out_height * out_width +
                                     oh * out_width + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
}
#include <stdio.h>
#include <stdlib.h>

// Matrix multiplication for a single patch (out_channels × patch_size)
void matmul_patch(float* kernels, float* patch, float* output, 
                  int out_channels, int patch_size, int base_index, int out_height, int out_width) {
    for (int oc = 0; oc < out_channels; oc++) {
        float sum = 0.0;
        for (int i = 0; i < patch_size; i++) {
            sum += kernels[oc * patch_size + i] * patch[i];
        }
        // Correct output indexing to match 4D tensor layout
        int output_idx = base_index + oc * out_height * out_width;
        output[output_idx] = sum;
    }
}

// Optimized im2col + immediate matrix multiplication with padding
void im2col_matmul(float* x, float* kernels, float* output, 
                   int batch, int channels, int height, int width, 
                   int out_channels, int kernel_size, int pad, float* patch) {
    
    int out_H = height + 2 * pad - kernel_size + 1;
    int out_W = width + 2 * pad - kernel_size + 1;
    int patch_size = channels * kernel_size * kernel_size;

    // Precompute frequently used values
    int input_stride = channels * height * width;
    int kernel_stride = channels * kernel_size * kernel_size;
    int output_stride = out_channels * out_H * out_W;

    // Iterate over batches
    for (int b = 0; b < batch; b++) {
        float* input_batch = &x[b * input_stride];  // Pointer to the current batch

        // Iterate over output spatial dimensions
        for (int oh = 0; oh < out_H; oh++) {
            for (int ow = 0; ow < out_W; ow++) {
                float* patch_ptr = patch;

                // Extract the patch
                for (int ic = 0; ic < channels; ic++) {
                    float* input_channel = &input_batch[ic * height * width];  // Pointer to the current channel
                    for (int kh = 0; kh < kernel_size; kh++) {
                        int h_idx = oh + kh - pad;  // Adjust for padding
                        if (h_idx >= 0 && h_idx < height) {
                            float* input_row = &input_channel[h_idx * width];  // Pointer to the current row
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int w_idx = ow + kw - pad;
                                *patch_ptr++ = (w_idx >= 0 && w_idx < width) ? input_row[w_idx] : 0.0f;
                            }
                        } else {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                *patch_ptr++ = 0.0f;
                            }
                        }
                    }
                }

                // Perform matmul for this patch
                int base_index = b * output_stride + oh * out_W + ow;
                matmul_patch(kernels, patch, output, out_channels, patch_size, base_index, out_H, out_W);
            }
        }
    }
}

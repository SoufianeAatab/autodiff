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
typedef struct
{
    int32_t min; /**< Min value used to clamp the result */
    int32_t max; /**< Max value used to clamp the result */
} cmsis_nn_activation;

typedef struct
{
    int32_t w; /**< Width */
    int32_t h; /**< Height */
} tile;

/** CMSIS-NN object used for the function context. */
typedef struct
{
    void *buf;    /**< Pointer to a buffer needed for the optimization */
    int32_t size; /**< Buffer size */
} cmsis_nn_context;

typedef struct conv_params_t {
    int32_t input_offset;  /**< The negative of the zero value for the input tensor */
    int32_t output_offset; /**< The negative of the zero value for the output tensor */
    tile stride;
    tile padding;
    tile dilation;
    cmsis_nn_activation activation;
};

typedef struct
{
    int32_t n; /**< Generic dimension to contain either the batch size or output channels.
                     Please refer to the function documentation for more information */
    int32_t h; /**< Height */
    int32_t w; /**< Width */
    int32_t c; /**< Input channels */
} conv_dims;

typedef enum {
    ARM_CMSIS_NN_SUCCESS = 0,
    ARM_CMSIS_NN_ARG_ERROR = -1
} arm_cmsis_nn_status;

float* sigmoid(float* x, float* a, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        a[i] = 1.0f / (1.0f + exp(-x[i]));
    }
    return a;
}

void sigmoid_diff(float* x, float* grad, float* out, uint32_t size) {
    while(size){
        *out = 1.0f / (1.0f + exp(-*x));
        --size;
        ++out;
        ++x;
    }
    // for (uint32_t i = 0; i < size; ++i) {
    //     float a = 1.0f / (1.0f + exp(-x[i]));
    //     out[i] = (a * (1.0-a)) * grad[i];
    // }
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

void arm_add_f32(
  float * pSrcA,
  float * pSrcB,
  float * pDst,
  uint32_t blockSize)
{
  uint32_t blkCnt;                               /* loop counter */

#ifndef ARM_MATH_CM0_FAMILY

/* Run the below code for Cortex-M4 and Cortex-M3 */
  float inA1, inA2, inA3, inA4;              /* temporary input variabels */
  float inB1, inB2, inB3, inB4;              /* temporary input variables */

  /*loop Unrolling */
  blkCnt = blockSize >> 2u;

  /* First part of the processing with loop unrolling.  Compute 4 outputs at a time.        
   ** a second loop below computes the remaining 1 to 3 samples. */
  while(blkCnt > 0u)
  {
    /* C = A + B */
    /* Add and then store the results in the destination buffer. */

    /* read four inputs from sourceA and four inputs from sourceB */
    inA1 = *pSrcA;
    inB1 = *pSrcB;
    inA2 = *(pSrcA + 1);
    inB2 = *(pSrcB + 1);
    inA3 = *(pSrcA + 2);
    inB3 = *(pSrcB + 2);
    inA4 = *(pSrcA + 3);
    inB4 = *(pSrcB + 3);

    /* C = A + B */
    /* add and store result to destination */
    *pDst = inA1 + inB1;
    *(pDst + 1) = inA2 + inB2;
    *(pDst + 2) = inA3 + inB3;
    *(pDst + 3) = inA4 + inB4;

    /* update pointers to process next samples */
    pSrcA += 4u;
    pSrcB += 4u;
    pDst += 4u;


    /* Decrement the loop counter */
    blkCnt--;
  }

  /* If the blockSize is not a multiple of 4, compute any remaining output samples here.        
   ** No loop unrolling is used. */
  blkCnt = blockSize % 0x4u;

#else

  /* Run the below code for Cortex-M0 */

  /* Initialize blkCnt with number of samples */
  blkCnt = blockSize;

#endif /* #ifndef ARM_MATH_CM0_FAMILY */

  while(blkCnt > 0u)
  {
    /* C = A + B */
    /* Add and then store the results in the destination buffer. */
    *pDst++ = (*pSrcA++) + (*pSrcB++);

    /* Decrement the loop counter */
    blkCnt--;
  }
}

void arm_mult_f32(
  float * pSrcA,
  float * pSrcB,
  float * pDst,
  uint32_t blockSize)
{
  uint32_t blkCnt;                               /* loop counters */
#ifndef ARM_MATH_CM0_FAMILY

  /* Run the below code for Cortex-M4 and Cortex-M3 */
  float inA1, inA2, inA3, inA4;              /* temporary input variables */
  float inB1, inB2, inB3, inB4;              /* temporary input variables */
  float out1, out2, out3, out4;              /* temporary output variables */

  /* loop Unrolling */
  blkCnt = blockSize >> 2u;

  /* First part of the processing with loop unrolling.  Compute 4 outputs at a time.        
   ** a second loop below computes the remaining 1 to 3 samples. */
  while(blkCnt > 0u)
  {
    /* C = A * B */
    /* Multiply the inputs and store the results in output buffer */
    /* read sample from sourceA */
    inA1 = *pSrcA;
    /* read sample from sourceB */
    inB1 = *pSrcB;
    /* read sample from sourceA */
    inA2 = *(pSrcA + 1);
    /* read sample from sourceB */
    inB2 = *(pSrcB + 1);

    /* out = sourceA * sourceB */
    out1 = inA1 * inB1;

    /* read sample from sourceA */
    inA3 = *(pSrcA + 2);
    /* read sample from sourceB */
    inB3 = *(pSrcB + 2);

    /* out = sourceA * sourceB */
    out2 = inA2 * inB2;

    /* read sample from sourceA */
    inA4 = *(pSrcA + 3);

    /* store result to destination buffer */
    *pDst = out1;

    /* read sample from sourceB */
    inB4 = *(pSrcB + 3);

    /* out = sourceA * sourceB */
    out3 = inA3 * inB3;

    /* store result to destination buffer */
    *(pDst + 1) = out2;

    /* out = sourceA * sourceB */
    out4 = inA4 * inB4;
    /* store result to destination buffer */
    *(pDst + 2) = out3;
    /* store result to destination buffer */
    *(pDst + 3) = out4;


    /* update pointers to process next samples */
    pSrcA += 4u;
    pSrcB += 4u;
    pDst += 4u;

    /* Decrement the blockSize loop counter */
    blkCnt--;
  }

  /* If the blockSize is not a multiple of 4, compute any remaining output samples here.        
   ** No loop unrolling is used. */
  blkCnt = blockSize % 0x4u;

#else

  /* Run the below code for Cortex-M0 */

  /* Initialize blkCnt with number of samples */
  blkCnt = blockSize;

#endif /* #ifndef ARM_MATH_CM0_FAMILY */

  while(blkCnt > 0u)
  {
    /* C = A * B */
    /* Multiply the inputs and store the results in output buffer */
    *pDst++ = (*pSrcA++) * (*pSrcB++);

    /* Decrement the blockSize loop counter */
    blkCnt--;
  }
}


void add(float* a, float *b, float* c, uint32_t size) {
    float* a_ptr = a;
    float* b_ptr = b;
    float* c_ptr = c;
    while(size){
        *c_ptr = *a_ptr + *b_ptr;

        --size;
        ++a_ptr;
        ++b_ptr;
        ++c_ptr;
    }
}

void sub(float* a, float *b, float* c, uint32_t size) {
    float* a_ptr = a;
    float* b_ptr = b;
    float* c_ptr = c;
    while(size){
        *c_ptr = *a_ptr - *b_ptr;

        --size;
        ++a_ptr;
        ++b_ptr;
        ++c_ptr;
    }
}

void exp(float* a, float *b, uint32_t size) {
    float* a_ptr = a;
    float* b_ptr = b;
    while(size){
        *b_ptr = expf(*a_ptr);

        --size;
        ++a_ptr;
        ++b_ptr;
    }
}

#if __arm__
__STATIC_FORCEINLINE 
#else
inline
#endif
void arm_memcpy(int8_t *
#if __arm__
__RESTRICT 
#endif

dst, const int8_t *
#if __arm__
__RESTRICT 
#endif
src, uint32_t block_size)
{
#if defined(ARM_MATH_MVEI)
    __asm volatile("   wlstp.8                 lr, %[cnt], 1f             \n"
                   "2:                                                    \n"
                   "   vldrb.8                 q0, [%[in]], #16            \n"
                   "   vstrb.8                 q0, [%[out]], #16           \n"
                   "   letp                    lr, 2b                     \n"
                   "1:                                                    \n"
                   : [in] "+r"(src), [out] "+r"(dst)
                   : [cnt] "r"(block_size)
                   : "q0", "memory", "r14");
#else
    memcpy(dst, src, block_size);
#endif
}

static void compare_and_replace_if_larger(float *base, const float *target, int32_t length)
{
    while(length){
        if (*target > *base)
        {
            *base = *target;
        }

        --length;
    }
}

arm_cmsis_nn_status arm_max_pool_s16(int32_t stride_x, int32_t stride_y, int32_t pad_x, int pad_y,
                                    int16_t act_min, int16_t act_max, 
                                    int32_t batch_cnt, int32_t input_x, int32_t input_y, int32_t channel_in,
                                    int32_t output_x, int32_t output_y, int32_t kernel_x, int32_t kernel_y,
                                     const float *src,
                                     float *dst)
{
    const int32_t batch_size = input_x * input_y * channel_in;
    while (batch_cnt)
    {
        float *dst_base = dst;
        for (int i_y = 0, base_idx_y = -pad_y; i_y < output_y; base_idx_y += stride_y, i_y++)
        {
            for (int i_x = 0, base_idx_x = -pad_x; i_x < output_x; base_idx_x += stride_x, i_x++)
            {
                /* Condition for kernel start dimension: (base_idx_<x,y> + kernel_<x,y>_start) >= 0 */
                const int32_t ker_y_start = MAX(0, -base_idx_y);
                const int32_t ker_x_start = MAX(0, -base_idx_x);

                /* Condition for kernel end dimension: (base_idx_<x,y> + kernel_<x,y>_end) < dim_src_<width,height> */
                const int32_t kernel_y_end = MIN(kernel_y, input_y - base_idx_y);
                const int32_t kernel_x_end = MIN(kernel_x, input_x - base_idx_x);

                int count = 0;

                for (int k_y = ker_y_start; k_y < kernel_y_end; k_y++)
                {
                    for (int k_x = ker_x_start; k_x < kernel_x_end; k_x++)
                    {
                        const float *start = src + channel_in * (k_x + base_idx_x + (k_y + base_idx_y) * input_x);

                        if (count == 0)
                        {
                            memcpy(dst, start, channel_in * sizeof(float));
                            count++;
                        }
                        else
                        {
                            compare_and_replace_if_larger(dst, start, channel_in);
                        }
                    }
                }
                /* 'count' is expected to be non-zero here. */
                dst += channel_in;
            }
        }

        // clamp_output(dst_base, output_x * output_y * channel_in, act_min, act_max);

        src += batch_size;
        batch_cnt--;
    }

    return ARM_CMSIS_NN_SUCCESS;
}

void max_pool_backward(const float *grad_output, const float *input, float *grad_input, 
                       size_t input_x, size_t input_y, size_t channel_in,
                       size_t output_x, size_t output_y, size_t kernel_x, size_t kernel_y,
                       size_t stride_x, size_t stride_y, size_t pad_x, size_t pad_y) {
        for (size_t c = 0; c < channel_in; c++) {
        for (size_t b = 0; b < 1; b++) {
            for (size_t i_y = 0; i_y < output_y; i_y++) {
                for (size_t i_x = 0; i_x < output_x; i_x++) {
                    // Find indices of the corresponding pooling window in the input
                    size_t base_idx_y = i_y * stride_y - pad_y;
                    size_t base_idx_x = i_x * stride_x - pad_x;

                    /* Condition for kernel end dimension: (base_idx_<x,y> + kernel_<x,y>_end) < dim_src_<width,height> */
                    const int32_t kernel_y_end = MIN(kernel_y, input_y - base_idx_y);
                    const int32_t kernel_x_end = MIN(kernel_x, input_x - base_idx_x);

                    // Initialize gradient for this pooling window
                    float grad = grad_output[b * output_x * output_y * channel_in + i_y * output_x * channel_in + i_x * channel_in + c];
                    
                    // Find index of the maximum value in the corresponding forward pass
                    size_t max_idx_y = base_idx_y;
                    size_t max_idx_x = base_idx_x;
                    float max_val = input[b * input_x * input_y * channel_in + max_idx_y * input_x * channel_in + max_idx_x * channel_in + c];

                    // Iterate over elements in the pooling window to find the maximum value
                    for (size_t k_y = base_idx_y; k_y < base_idx_y+kernel_y_end; k_y++) {
                        for (size_t k_x = base_idx_x; k_x < base_idx_x+kernel_x_end; k_x++) {
                            // Find index in the input array
                            size_t input_idx = b * input_x * input_y * channel_in + k_y * input_x * channel_in + k_x * channel_in + c;
                            // Check if the current input value is greater than the current max value
                            if (input[input_idx] > max_val) {
                                // Update max value and its indices
                                max_val = input[input_idx];
                                max_idx_y = k_y;
                                max_idx_x = k_x;
                            }
                        }
                    }
                    // Assign the gradient to the position of the maximum value in the input
                    grad_input[b * input_x * input_y * channel_in + max_idx_y * input_x * channel_in + max_idx_x * channel_in + c] += grad;
                }
            }
        }
    }
}

arm_cmsis_nn_status arm_convolve_NHWC( float* ctx_buf,
                                           uint32_t pad_x, uint32_t pad_y, uint32_t stride_x, uint32_t stride_y,
                                           uint32_t out_activation_min, uint32_t out_activation_max,
                                           uint32_t input_batches, uint32_t input_x, uint32_t input_y, uint32_t input_ch, uint32_t kernel_x, uint32_t kernel_y, uint32_t rhs_cols,
                                           float *input_data,
                                           float *filter_data,
                                           uint32_t output_x, uint32_t output_y, uint32_t output_ch,
                                          float *output_data)
{
    float *buffer_a = ctx_buf;
    for (int i_batch = 0; i_batch < input_batches; i_batch++)
    {
        /* Generate two columns from the input tensor a GEMM computation */
        float *two_column_buf = buffer_a;
        float *out = output_data;
        /* This part implements the im2col function */
        for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++)
        {
            for (int32_t i_out_x = 0; i_out_x < output_x; i_out_x++)
            {
                for (int32_t i_ker_y = i_out_y * stride_y - pad_y; i_ker_y < i_out_y * stride_y - pad_y + kernel_y;
                     i_ker_y++)
                {

                    for (int32_t i_ker_x = i_out_x * stride_x - pad_x; i_ker_x < i_out_x * stride_x - pad_x + kernel_x;
                         i_ker_x++)
                    {
                        if (i_ker_y < 0 || i_ker_y >= input_y || i_ker_x < 0 || i_ker_x >= input_x) {
                            /* Filling 0 for out-of-bound paddings */
                            memset((int8_t *)two_column_buf, 0, sizeof(float) * input_ch);
                        } else {
                            arm_memcpy((int8_t *)two_column_buf, (const int8_t *)(input_data + (i_ker_y * input_x + i_ker_x) * input_ch), input_ch * sizeof(float) * 2);
                        }
                        two_column_buf += input_ch;
                    }
                }
                
                /* Computation is filed for every 1 columns */
                if (two_column_buf == buffer_a + 2 * rhs_cols)
                {
                    out = mat_mul(filter_data, buffer_a, out, 
                    output_ch, rhs_cols, rhs_cols, 1,
                    rhs_cols, 2, 2, 1);
                    /* Counter reset */
                    two_column_buf = buffer_a;
                }
            }
        }

        /* Left-over because odd number of output pixels */
        if (two_column_buf != buffer_a)
        {
            const float *ker_a = filter_data;
            int i;

            for (i = 0; i < output_ch; i++)
            {
                /* Init the accumulator*/
                float sum = 0;

                /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
                const float *ip_as_col = buffer_a;

                /* 4 multiply and accumulates are done in one loop. */
                int32_t col_count = rhs_cols >> 2;

                while (col_count)
                {
                    float ker_a1, ker_a2, ker_a3, ker_a4;
                    float ip_b1, ip_b2, ip_b3, ip_b4;

                    ker_a1 = *ker_a++;
                    ip_b1 = *ip_as_col++;
                    sum += ker_a1 * ip_b1;

                    ker_a2 = *ker_a++;
                    ip_b2 = *ip_as_col++;
                    sum += ker_a2 * ip_b2;

                    ker_a3 = *ker_a++;
                    ip_b3 = *ip_as_col++;
                    sum += ker_a3 * ip_b3;

                    ker_a4 = *ker_a++;
                    ip_b4 = *ip_as_col++;
                    sum += ker_a4 * ip_b4;

                    col_count--;
                }
                /* Handle left over mac */
                col_count = rhs_cols & 0x3;
                while (col_count)
                {
                    float ker_a1 = *ker_a++;
                    float ip_b1 = *ip_as_col++;
                    sum += ker_a1 * ip_b1;

                    col_count--;
                }
                sum = MAX(sum, out_activation_min);
                sum = MIN(sum, out_activation_max);
                *out++ = (float)sum;
            }
        }

        /* Advance to the next batch */
        input_data += (input_x * input_y * input_ch);
        output_data += (output_x * output_y * output_ch);
    }

    /* Return to application */
    return ARM_CMSIS_NN_SUCCESS;
}

// void bp_convolve2d(float* ctx_buf, uint32_t pad_x, uint32_t pad_y, uint32_t stride_x, uint32_t stride_y,
//                                            uint32_t out_activation_min, uint32_t out_activation_max,
//                                            uint32_t input_batches, uint32_t input_x, uint32_t input_y, uint32_t input_ch, uint32_t kernel_x, uint32_t kernel_y, uint32_t rhs_cols,
//                                            float *input_data,
//                                            float *filter_data,
//                                            uint32_t output_x, uint32_t output_y, uint32_t output_ch,
//                                           float *output_data,
//                                           float *dx
//                     ){

//     // input_batches = out_ch

//     for (int out_ch = 0; out_ch < output_ch; out_ch++) {
//         float *out = output_data;
//         for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++) {
//             for (int32_t i_out_x = 0; i_out_x < output_x; i_out_x++) {
//                 for (int32_t i_ker_y = i_out_y * stride_y - pad_y; i_ker_y < i_out_y * stride_y - pad_y + kernel_y;
//                      i_ker_y++) {
//                     for (int32_t i_ker_x = i_out_x * stride_x - pad_x; i_ker_x < i_out_x * stride_x - pad_x + kernel_x;
//                          i_ker_x++) {
//                         if (i_ker_y < 0 || i_ker_y >= input_y || i_ker_x < 0 || i_ker_x >= input_x) {
//                             /* Filling 0 for out-of-bound paddings */
//                             memset((int8_t *)two_column_buf, 0, sizeof(float) * input_ch);
//                         } else {
//                             arm_memcpy((int8_t *)two_column_buf, (const int8_t *)(input_data + (i_ker_y * input_x + i_ker_x) * input_ch), input_ch * sizeof(float));
//                         }

//                         const uint32_t input_index = input_h * input_w;

//                         if(dx) {
//                             dx[input_index] += grads[grad_index] * filter_data[kernel_index];
//                         }
//                     }
//                 }
//                 /* Computation is filed for every 1 columns */
//                 if (two_column_buf == buffer_a + 1 * rhs_cols)
//                 {
//                     out = mat_vec_mul(filter_data, buffer_a, out, 
//                     output_ch, rhs_cols, rhs_cols, 1,
//                     rhs_cols, 1, 1, 1);
//                     /* Counter reset */
//                     two_column_buf = buffer_a;
//                 }
//             }
//         }
//     }

//     // for (uint32_t out = 0; out < out_ch; ++out){
//     //     float dbias = 0.0f;
//     //     for (uint32_t c = 0; c < in_ch; ++c) {
//     //         for (uint32_t h = 0; h < output_h; ++h) {
//     //             for (uint32_t w = 0; w < output_w; ++w) {
//     //                 dbias += grads[out * output_h * output_w + h * output_w + w];
//     //                 for (uint32_t k_h = 0; k_h < 3; ++k_h) {
//     //                     for (uint32_t k_w = 0; k_w < 3; ++k_w) {

//     //                         // kernels OCHW
//     //                         const uint32_t kernel_index =  out * (in_ch * kernel_x * kernel_y) + 
//     //                                     c * (kernel_x * kernel_y) + 
//     //                                     k_h * kernel_x + 
//     //                                     k_w;
                            
//     //                         // const uint32_t out_grad_index = c * (input_h * input_w) + (h * stride + k_h) * input_w + (w*stride + k_w);
//     //                         const uint32_t input_index = c * (input_h * input_w) + (h * stride + k_h) * input_w + (w*stride + k_w);
//     //                         const uint32_t grad_index = out * (output_h * output_w) + h * output_w + w;
//     //                         dw[kernel_index] += grads[grad_index] * input[input_index];
//     //                         if(dx && (h*stride+k_h)<input_h && (w*stride+k_w) < input_w){
//     //                             dx[input_index] += grads[grad_index] * kernels[kernel_index];
//     //                         }
//     //                     }
//     //                 }
//     //             }
//     //         }

//     //     }
//     //     db[out] += dbias;
//     // }
// }

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

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

float* mat_mul(float* a_data, float* b_data, float* c_data, 
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

arm_cmsis_nn_status arm_convolve_NHWC( cmsis_nn_context *ctx,
                                           conv_params_t *conv_params,
                                           conv_dims *input_dims,
                                           float *input_data,
                                           conv_dims *filter_dims,
                                           float *filter_data,
                                           conv_dims *output_dims,
                                          float *output_data)
{
    //(void)bias_dims;
    if (filter_dims->w * filter_dims->h * input_dims->c >= 512)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // if (ctx->buf == NULL && arm_convolve_s8_get_buffer_size(input_dims, filter_dims) > 0)
    // {
    //     return ARM_CMSIS_NN_ARG_ERROR;
    // }
    float *buffer_a = (float *)ctx->buf;

    const int32_t input_batches = input_dims->n;
    const int32_t input_x = input_dims->w;
    const int32_t input_y = input_dims->h;
    const int32_t input_ch = input_dims->c;
    const int32_t kernel_x = filter_dims->w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t output_ch = output_dims->c;
    const int32_t rhs_cols = input_ch * kernel_y * kernel_x;

    const int32_t pad_x = conv_params->padding.w;
    const int32_t pad_y = conv_params->padding.h;
    const int32_t stride_x = conv_params->stride.w;
    const int32_t stride_y = conv_params->stride.h;

    const int16_t out_activation_min = conv_params->activation.min;
    const int16_t out_activation_max = conv_params->activation.max;

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
                            arm_memcpy((int8_t *)two_column_buf, (const int8_t *)(input_data + (i_ker_y * input_x + i_ker_x) * input_ch), input_ch * sizeof(float));
                        }
                        two_column_buf += input_ch;
                    }
                }
                
                /* Computation is filed for every 1 columns */
                if (two_column_buf == buffer_a + 1 * rhs_cols)
                {
                    out = mat_mul(filter_data, buffer_a, out, 
                    output_ch, rhs_cols, rhs_cols, 1,
                    rhs_cols, 1, 1, 1);
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
                    //printf("%f * %f = %f \n", ker_a1 , ip_b1, ker_a1 * ip_b1);

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

int main() {

    cmsis_nn_context ctx = {};
    ctx.buf = malloc(1024*32);
    float filter_data[] = {0.4963, 0.6323, 0.1610, 0.7682, 0.3489, 0.2823, 0.0885, 0.4017, 0.6816,
        0.1320, 0.0223, 0.9152, 0.3074, 0.1689, 0.3971, 0.6341, 0.2939, 0.8742,
        0.4901, 0.5185, 0.4194, 0.8964, 0.6977, 0.5529, 0.4556, 0.8000, 0.9527,
        0.0362, 0.2081, 0.2422, 0.1852, 0.9298, 0.8155, 0.3734, 0.7231, 0.7932,
        0.3051, 0.7423, 0.2783, 0.9320, 0.5263, 0.4820, 0.1759, 0.2437, 0.8198,
        0.2698, 0.5846, 0.9971, 0.1507, 0.0332, 0.6984, 0.0317, 0.1387, 0.5675,
        0.8352, 0.6511, 0.6965, 0.2056, 0.7745, 0.9143, 0.5932, 0.4369, 0.9351,
        0.1123, 0.5191, 0.9412, 0.1535, 0.6159, 0.5995, 0.2417, 0.8102, 0.0652,
        0.7262, 0.9801, 0.5460, 0.7011, 0.1147, 0.1872, 0.2038, 0.3168, 0.0340,
        0.9442, 0.6833, 0.2121, 0.8802, 0.7529, 0.9704, 0.0012, 0.8579, 0.8369,
        0.5936, 0.6870, 0.2820, 0.4158, 0.0051, 0.3742, 0.4177, 0.1757, 0.0237,
        0.2711, 0.7497, 0.4910, 0.6923, 0.6047, 0.1235, 0.2038, 0.1100, 0.1143,
        0.4725, 0.5226, 0.3748, 0.5751, 0.5730, 0.2564, 0.2952, 0.6186, 0.3251,
        0.7967, 0.6962, 0.0902, 0.1957, 0.5300, 0.3936, 0.9537, 0.2560, 0.6069,
        0.8426, 0.7366, 0.1743, 0.0784, 0.0204, 0.4743, 0.3756, 0.2036, 0.8579};

    float input_data[] = {0.4963, 0.9442, 0.0513, 0.7682, 0.8802, 0.0683, 0.0885, 0.0012, 0.4218,
        0.1320, 0.5936, 0.5065, 0.3074, 0.4158, 0.2729, 0.6341, 0.4177, 0.6883,
        0.4901, 0.2711, 0.0500, 0.8964, 0.6923, 0.4663, 0.4556, 0.2038, 0.9397,
        0.6323, 0.6833, 0.2961, 0.3489, 0.7529, 0.9515, 0.4017, 0.8579, 0.6811,
        0.0223, 0.6870, 0.0488, 0.1689, 0.0051, 0.8163, 0.2939, 0.1757, 0.4423,
        0.5185, 0.7497, 0.2768, 0.6977, 0.6047, 0.8998, 0.8000, 0.1100, 0.0960,
        0.1610, 0.2121, 0.5537, 0.2823, 0.9704, 0.3953, 0.6816, 0.8369, 0.8571,
        0.9152, 0.2820, 0.6396, 0.3971, 0.3742, 0.7403, 0.8742, 0.0237, 0.6766,
        0.4194, 0.4910, 0.3798, 0.5529, 0.1235, 0.3948, 0.9527, 0.1143, 0.0880,
        0.0362, 0.4725, 0.7709, 0.1852, 0.5751, 0.8970, 0.3734, 0.2952, 0.8421,
        0.3051, 0.7967, 0.1473, 0.9320, 0.1957, 0.5223, 0.1759, 0.9537, 0.1475,
        0.2698, 0.8426, 0.2248, 0.1507, 0.0784, 0.2086, 0.0317, 0.3756, 0.6709,
        0.2081, 0.5226, 0.2020, 0.9298, 0.5730, 0.4891, 0.7231, 0.6186, 0.5210,
        0.7423, 0.6962, 0.8223, 0.5263, 0.5300, 0.1220, 0.2437, 0.2560, 0.1567,
        0.5846, 0.7366, 0.2097, 0.0332, 0.0204, 0.8500, 0.1387, 0.2036, 0.3203,
        0.2422, 0.3748, 0.9217, 0.8155, 0.2564, 0.6808, 0.7932, 0.3251, 0.5633,
        0.2783, 0.0902, 0.4963, 0.4820, 0.3936, 0.4012, 0.8198, 0.6069, 0.5627,
        0.9971, 0.1743, 0.3858, 0.6984, 0.4743, 0.4965, 0.5675, 0.8579, 0.5638,
        0.8352, 0.4486, 0.1089, 0.2056, 0.5139, 0.2379, 0.5932, 0.4569, 0.9037,
        0.1123, 0.6012, 0.0942, 0.1535, 0.8179, 0.4641, 0.2417, 0.9736, 0.9946,
        0.7262, 0.8175, 0.6806, 0.7011, 0.9747, 0.5142, 0.2038, 0.4638, 0.0667,
        0.6511, 0.0508, 0.7477, 0.7745, 0.2630, 0.1439, 0.4369, 0.8405, 0.3581,
        0.5191, 0.4968, 0.3322, 0.6159, 0.2515, 0.4260, 0.8102, 0.1168, 0.5055,
        0.9801, 0.0321, 0.9124, 0.1147, 0.0780, 0.5624, 0.3168, 0.3986, 0.9478,
        0.6965, 0.7742, 0.8059, 0.9143, 0.7703, 0.1839, 0.9351, 0.0178, 0.7243,
        0.9412, 0.8119, 0.1466, 0.5995, 0.1087, 0.2881, 0.0652, 0.3943, 0.6471,
        0.5460, 0.2973, 0.6651, 0.1872, 0.4037, 0.8751, 0.0340, 0.4018, 0.3390};

    conv_dims input_dims = {};
    input_dims.n = 1;
    input_dims.c = 3;
    input_dims.h = 9;
    input_dims.w = 9;

    conv_dims filter_dims = {};
    filter_dims.n = 5;
    filter_dims.c = 3;
    filter_dims.h = 3;
    filter_dims.w = 3;

    conv_params_t conv_params = {};
    conv_params.stride.h = 1;
    conv_params.stride.w = 1;
    conv_params.padding.h = 0;
    conv_params.padding.w = 0;
    conv_params.dilation.h = 1;
    conv_params.dilation.w = 1;
    conv_params.activation.min = 0;
    conv_params.activation.max = 6;

    conv_dims output_dims = {};
    output_dims.n = 1;
    output_dims.c = 5;
    output_dims.h = 7;
    output_dims.w = 7;

    float* output_data = (float*)malloc(5*7*7*sizeof(float));

    arm_convolve_NHWC(&ctx,
                                &conv_params,
                                &input_dims,
                                input_data,
                                &filter_dims,
                                filter_data,
                                &output_dims,
                                output_data);

    for(int i=0;i<5*7*7;++i){
        printf("%f, ", output_data[i]);
    }
    printf("\n");

}
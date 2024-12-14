#include <stdint.h>

void sum(
    float * __restrict__ data, 
    float * __restrict__ out, 
    uint8_t dim, uint32_t rows, uint32_t cols
) {
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
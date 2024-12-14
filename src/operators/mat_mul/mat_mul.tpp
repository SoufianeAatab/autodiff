#include <stdint.h>
#include <memory>

/**
 * B[K][N] = transpose(B_t[N][K])
 * C[M][N] = A[M][K] * B[K][N]
 */
template<typename T, unsigned int M, unsigned int K, unsigned int N>
void mat_mul(
    const T * __restrict__ A,
    const T * __restrict__ B_t,
    T * __restrict__ C
) {
    A = (T*) __builtin_assume_aligned ((void*)A, 16);
    B_t = (T*) __builtin_assume_aligned ((void*)B_t, 16);
    C = (T*) __builtin_assume_aligned ((void*)C, 16);

    // Iterate through the result matrix
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            // Compute the dot product of row i of matrix a and column j of matrix b
            float sum = 0;
            for (uint32_t k = 0; k < K; ++k) {
                // printf("%f * %f\n", a[i * a_stride1 + k * a_stride2] * b[k * b_stride1 + j * b_stride2]);
                sum += A[i*K + k] * B_t[j*K + k];
            }
            // Store the result in the corresponding element of matrix c
            *C++ = sum;
        }
    }
}

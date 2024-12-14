#include <stdint.h>
#include <src/operators/mat_mul/mat_mul.tpp>
#include <random>
#include "../common.h"

#define M 2052
#define K 2052
#define N 2052

float* base_mat_mul(
    float* a_data, float* b_data, float* c_data, 
    uint32_t a_rows, uint32_t a_cols, uint32_t a_stride1, uint32_t a_stride2, 
    uint32_t b_rows, uint32_t b_cols, uint32_t b_stride1, uint32_t b_stride2
    ) {

    float* a = a_data;
    float* b = b_data;

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



int main() {
    float *a, *b, *b_t, *g, *c;

    CHECK(posix_memalign((void**)&a,16,sizeof(float[M][K])));
    CHECK(posix_memalign((void**)&b,16,sizeof(float[K][N])));
    CHECK(posix_memalign((void**)&b_t,16,sizeof(float[N][K])));
    CHECK(posix_memalign((void**)&g,16,sizeof(float[M][N])));
    CHECK(posix_memalign((void**)&c,16,sizeof(float[M][N])));

    srand (time(0));
    random_fill(a, M*K);
    random_fill(b, K*N);
    for(int k=0; k < K; ++k) {
        for(int n=0; n < N; ++n) {
            b_t[n*K + k] = b[k*N + n];
        }
    }

    printf("Benchmark base data\n");
    auto base_time = TIME({
        // base_mat_mul(a, b, g, M/4, K/4, K/4, 1, K/4, N/4, N/4, 1);
        // base_mat_mul(a, b, g, M/2, K/2, K/2, 1, K/2, N/2, N/2, 1);
        base_mat_mul(a, b, g, M, K, K, 1, K, N, N, 1);
    });
    printf("Time: %ld.\nBenchmark operator data\n", base_time);
    auto operator_time = TIME({
        // (mat_mul<float, M/4, K/4, N/4>(a, b_t, c));
        // (mat_mul<float,M/2,K/2,N/2>(a, b_t, c));
        (mat_mul<float,M,K,N>(a, b_t, c));
    });
    printf("Time: %ld.\nVerify data.\n", operator_time);
    cmp_data(g,c, M*N);

    printf("Speed up: %f.\n", (double)base_time / (double)operator_time);
    exit(0);
}
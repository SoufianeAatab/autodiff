#include <stdint.h>
#include <src/operators/mat_mul/mat_mul.tpp>
#include <random>
#include <chrono>

#define M 2052
#define K 2052
#define N 2052

float* base_mat_mul(float* a_data, float* b_data, float* c_data, 
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

#define CHECK(_VALUE) \
    { \
        uint32_t error = _VALUE; \
        if (error != 0) { \
            fprintf(stderr, "Error: %d.\n", error); \
            exit(error); \
        } \
    }

int main() {
    float *a, *b, *g, *c;

    CHECK(posix_memalign((void**)&a,16,sizeof(float[M][K])));
    CHECK(posix_memalign((void**)&b,16,sizeof(float[K][N])));
    CHECK(posix_memalign((void**)&g,16,sizeof(float[M][N])));
    CHECK(posix_memalign((void**)&c,16,sizeof(float[M][N])));

    srand (time(0));

    for(int i = 0; i < M*K; ++i) {
        a[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/10));
    }

    for(int i = 0; i < K*N; ++i) {
        b[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/10)); 
    }

    printf("Benchmark data\n");
    auto begin = std::chrono::high_resolution_clock::now();
    base_mat_mul(a, b, g, M/4, K/4, K/4, 1, K/4, N/4, N/4, 1);
    base_mat_mul(a, b, g, M/2, K/2, K/2, 1, K/2, N/2, N/2, 1);
    base_mat_mul(a, b, g, M, K, K, 1, K, N, N, 1);
    auto half = std::chrono::high_resolution_clock::now();
    mat_mul<float,M/4,K/4,N/4>(a, b, c);
    mat_mul<float,M/2,K/2,N/2>(a, b, c);
    mat_mul<float,M,K,N>(a, b, c);
    auto end = std::chrono::high_resolution_clock::now();

    printf("Check data\n");
    for(int i = 0; i < M*N; ++i) {
        if (g[i] != c[i]) {
            fprintf(stderr, "Gold data do not match with operator data in %i: %f != %f.\n", i, g[i], c[i]);
            break;
        }
    }

    auto base_time = std::chrono::duration_cast<std::chrono::microseconds>(half-begin).count();
    auto operator_time = std::chrono::duration_cast<std::chrono::microseconds>(end-half).count();

    printf("Time: %ld, %ld. Speed up: %f.\n", base_time, operator_time, (double)base_time / (double)operator_time);
    exit(0);
}
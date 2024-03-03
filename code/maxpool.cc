#include "conv2d_cmsis.cc"
#include "sockets.c"
#include <time.h>

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

    uint32_t size = 0;
    float* input_ptr = get_data(&size);
    float* y_ptr = get_data(&size);
    float* buf = (float*)calloc(175254, sizeof(float));
    float* ctx = (float*)calloc(32*32*32, sizeof(float));

    set_weights(&buf[1114], 54080, 0.0001849112426035503); // (10, 5408)
    set_weights(&buf[0], 288, 0.1111111111111111); // (32, 3, 3, 1)
    set_weights(&buf[288], 32, 0.03125); // (1, 32)


    
    float lr = 0.01;
    for(uint32_t n=0;n<100;++n){
        float loss = 0.0f;
        float correct = 0;
        clock_t start_time = clock();
        for(uint32_t l=0;l<60000;++l){
            arm_convolve_NHWC( ctx, 0, 0, 1, 1,-6, 6, 1, 28, 28, 1, 3, 3, 9, &input_ptr[l*784], &buf[0], 26, 26, 32,&buf[55194]); // [1, 26, 26, 32] 5
            arm_max_pool_s16(2, 2, 0, 0, -6,  6, 1, 26, 26, 32,13, 13, 2, 2, &buf[55194],&buf[76826]); // (1, 13, 13, 32) 6
            sigmoid(&buf[76826] /* (1, 13, 13, 32)*/ , &buf[82234] /*(1, 13, 13, 32)*/, 5408); // (1, 13, 13, 32) 7
            mat_mul(&buf[82234] /* (1, 5408) */, &buf[1114] /* (5408, 10) */, &buf[87642] /* (1, 10) */, 1, 5408, 5408, 1, 5408, 10, 1, 5408); // (1, 10) 10
            log_softmax(&buf[87642], &buf[87652], 10); // (1, 10) 11
            if(accuracy(&buf[87652], &y_ptr[l*10], 10)) correct += 1;
            buf[87662] = nll_loss(&buf[87652], &y_ptr[l*10], 10); // (1, 10) 12
            for(uint32_t k=0;k<10;++k){
                buf[87672 + k] = 1.0f;
            }
            for(uint32_t k=0;k<10;++k){
                buf[87682 + k] = -1;
            }
            mul(&buf[87672], &buf[87682], &buf[87692], 10); // (1, 10) 18
            mul(&buf[87692], &y_ptr[l*10], &buf[87702], 10); // (1, 10) 19
            exp(&buf[87652], &buf[87712], 10); // (1, 10) 20
            add(&buf[87712], &buf[87702], &buf[87722], 10); // (1, 10) 21
            mat_mul(&buf[87722] /* (10, 1) */, &buf[82234] /* (1, 5408) */, &buf[87732] /* (10, 5408) */, 10, 1, 1, 10, 1, 5408, 5408, 1); // (10, 5408) 24
            mat_mul(&buf[87722] /* (1, 10) */, &buf[1114] /* (10, 5408) */, &buf[141812] /* (1, 5408) */, 1, 10, 10, 1, 10, 5408, 5408, 1); // (1, 5408) 25
            sigmoid_diff(&buf[76826], &buf[141812], &buf[147220], 5408); // (1, 13, 13, 32) 27
            max_pool_backward(&buf[147220], &buf[55194], &buf[152628],  13,  13,  32, 26, 26, 2,  2, 2, 2, 0, 0); // (1, 26, 26, 32) 28
            sum(&buf[152628], &buf[174260], 1, 32, 676); // (676,) 32
            for(uint32_t k=0;k<10;++k){
                buf[174936 + k] = -1;
            }
            mul(&buf[87672], &buf[174936], &buf[174946], 10); // (1, 10) 15
            mul(&buf[174946], &buf[87652], &buf[174956], 10); // (1, 10) 16
            arm_convolve_NHWC( ctx, 0, 0, 1, 1,-6, 6, 1, 28, 28, 1, 26, 26, 676, &input_ptr[l*784], &buf[152628], 3, 3, 32,&buf[174966]); // (1, 3, 3, 32) 33
            for (int i = 0; i < 32; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        for (int l = 0; l < 1; l++) {
                            // Compute linear indices for both arrays
                            int index1 = (i * 3 * 3 * 1) + (j * 3 * 1) + (k * 1) + l;
                            int index2 = (j * 32 * 3 * 3) + (i * 3 * 32) + (k * 32) + l;

                            // Add corresponding elements and store the result
                            buf[0 + index1] -= lr * buf[174966 + index2];
                        }
                    }
                }
            }
            for (uint32_t k=0;k<54080;++k){
                buf[1114 + k] -= buf[87732 + k] * lr;
            }

            for (uint32_t k=0;k<676;++k){
                buf[288 + k] -= buf[174260 + k] * lr;
            }

            loss += buf[87662];
    }
        clock_t end_time = clock();
        double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        printf("Epoch %d loss = %f, acc = %f, elapsed time: %.6f seconds\n", n, loss / 60000.0f, correct/60000.0f, elapsed_time);

    }
} 
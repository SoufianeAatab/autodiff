#include "conv2d_cmsis.cc"
#include "sockets.c"
#include <time.h>
#include "mnist_load.c"

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
    mnist_dataset_t ds = mnist_get_train();
    uint32_t size = 0;
    float* input_ptr = ds.images;
    float* y_ptr = ds.labels;
    float* buf = (float*)calloc(521366, sizeof(float));
    float* ctx = (float*)calloc(32*32*32, sizeof(float));

    set_weights(&buf[874], 54080, 0.0001849112426035503); // (10, 5408)
    set_weights(&buf[0], 72, 0.1111111111111111); // (8, 3, 3, 1)
    set_weights(&buf[72], 8, 0.125); // (1, 8)
    
    float lr = 0.1;

    for(uint32_t n=0;n<100;++n){
        float loss = 0.0f;
        float correct = 0;
        for(uint32_t l=0;l<60000;++l){
            arm_convolve_NHWC( ctx, 0, 0, 1, 1,-6, 6, 1, 28, 28, 1, 3, 3, 9, &input_ptr[l*784], &buf[0], 26, 26, 8,&buf[54954]); // (1, 8, 26, 26) 5
            sigmoid(&buf[54954] /* (1, 8, 26, 26)*/ , &buf[60362] /*(1, 8, 26, 26)*/, 5408); // (1, 8, 26, 26) 6
            mat_mul(&buf[60362] /* (1, 5408) */, &buf[874] /* (5408, 10) */, &buf[65770] /* (1, 10) */, 1, 5408, 5408, 1, 5408, 10, 1, 5408); // (1, 10) 9
            log_softmax(&buf[65770], &buf[65780], 10); // (1, 10) 10
            if(accuracy(&buf[65780], &y_ptr[l*10], 10)) correct +=1;
            buf[65790] = nll_loss(&buf[65780], &y_ptr[l*10], 10); // (1, 10) 11
            for(uint32_t k=0;k<10;++k){
                buf[65800 + k] = 1.0f;
            }
            for(uint32_t k=0;k<10;++k){
                buf[65810 + k] = -1;
            }
            mul(&buf[65800], &buf[65810], &buf[65820], 10); // (1, 10) 14
            mul(&buf[65820], &buf[65780], &buf[65830], 10); // (1, 10) 15
            for(uint32_t k=0;k<10;++k){
                buf[65840 + k] = -1;
            }
            mul(&buf[65800], &buf[65840], &buf[65850], 10); // (1, 10) 17
            mul(&buf[65850], &y_ptr[l*10], &buf[65860], 10); // (1, 10) 18
            exp(&buf[65780], &buf[65870], 10); // (1, 10) 19
            add(&buf[65870], &buf[65860], &buf[65880], 10); // (1, 10) 20
            mat_mul(&buf[65880] /* (1, 10) */, &buf[874] /* (10, 5408) */, &buf[65890] /* (1, 5408) */, 1, 10, 10, 1, 10, 5408, 5408, 1); // (1, 5408) 24
            sigmoid_diff(&buf[54954], &buf[65890], &buf[71298], 5408); // (1, 8, 26, 26) 26
            arm_convolve_NHWC( ctx, 0, 0, 1, 1,-6, 6, 1, 28, 28, 1, 26, 26, 676, &input_ptr[l*784], &buf[71298], 3, 3, 8,&buf[76706]); // (1, 8, 3, 3) 31
            mat_mul(&buf[65880] /* (10, 1) */, &buf[60362] /* (1, 5408) */, &buf[76778] /* (10, 5408) */, 10, 1, 1, 10, 1, 5408, 5408, 1); // (10, 5408) 23
            sum(&buf[71298], &buf[130858], 1, 8, 676); // (676,) 30

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        for (int l = 0; l < 1; l++) {
                            // Compute linear indices for both arrays
                            // index1 = (i*k_h*k_w*c) + (j*k_w*c) + (k*c) + l
                            // index2 = (i*n*k_w*c) + (j*k_w*c) + (k*c) + l
                            int index1 = (i * 3 * 3 * 1) + (j * 3 * 1) + (k * 1) + l;
                            int index2 = (j * 8 * 3 * 1) + (i * 3 * 1) + (k * 1) + l;

                            // Add corresponding elements and store the result
                            buf[0 + index1] -= lr * buf[76706 + index2];
                        }
                    }
                }
            }

            for (uint32_t k=0;k<54080;++k){
                buf[874 + k] -= buf[76778 + k] * lr;
            }
            for (uint32_t k=0;k<676;++k){
                buf[72 + k] -= buf[130858 + k] * lr;
            }

        loss += buf[65790];
    }
        printf("Loss: %f, acc: %f \n", loss / 60000.0f, correct / 60000.0f);

    }
} 
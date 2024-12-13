#include "conv2d_cmsis.cc"
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
    float* buf = (float*)calloc(51828, sizeof(float));
  
    set_weights(&buf[0], 25088, 0.0012755102040816326); // (32, 784)
    set_weights(&buf[25088], 320, 0.03125); // (10, 32)


    float lr = 0.01;
    for (uint32_t j=0;j<100;++j){
        float loss = 0.0f;
        clock_t start_time = clock();
        uint32_t correct = 0;
        for(uint32_t l=0;l<60000;++l){
            for(uint32_t k=0;k<10;++k){
                buf[26202 + k] = -1;
            }
            mat_mul(&input_ptr[l*784] /* (1, 784) */, &buf[0] /* (784, 32) */, &buf[26212] /* (1, 32) */, 1, 784, 784, 1, 784, 32, 1, 784); // (1, 32) 5
            sigmoid(&buf[26212] /* (1, 32)*/ , &buf[26244] /*(1, 32)*/, 32); // (1, 32) 6
            mat_mul(&buf[26244] /* (1, 32) */, &buf[25088] /* (32, 10) */, &buf[26276] /* (1, 10) */, 1, 32, 32, 1, 32, 10, 1, 32); // (1, 10) 8
            log_softmax(&buf[26276], &buf[26286], 10); // (1, 10) 9
            if(accuracy(&buf[26286], &y_ptr[l*10], 10)) correct +=1;

            buf[26296] = nll_loss(&buf[26286], &y_ptr[l*10], 10); // (1, 10) 10
            for(uint32_t k=0;k<10;++k){
                buf[26306 + k] = 1.0f;
            }
            mul(&buf[26306], &buf[26202], &buf[26316], 10); // (1, 10) 16
            mul(&buf[26316], &y_ptr[l*10], &buf[26326], 10); // (1, 10) 17
            exp(&buf[26286], &buf[26336], 10); // (1, 10) 18
            add(&buf[26336], &buf[26326], &buf[26346], 10); // (1, 10) 19
            mat_mul(&buf[26346] /* (1, 10) */, &buf[25088] /* (10, 32) */, &buf[26356] /* (1, 32) */, 1, 10, 10, 1, 10, 32, 32, 1); // (1, 32) 23
            sigmoid_diff(&buf[26212], &buf[26356], &buf[26388], 32); // (1, 32) 24
            mat_mul(&buf[26388] /* (32, 1) */, &input_ptr[l*784] /* (1, 784) */, &buf[26420] /* (32, 784) */, 32, 1, 1, 32, 1, 784, 784, 1); // (32, 784) 27
            mat_mul(&buf[26346] /* (10, 1) */, &buf[26244] /* (1, 32) */, &buf[51508] /* (10, 32) */, 10, 1, 1, 10, 1, 32, 32, 1); // (10, 32) 22
            // sgd for 27
            for (uint32_t k=0;k<25088;++k){
                buf[0 + k] -= buf[26420 + k] * lr;
            }
            // sgd for 22
            for (uint32_t k=0;k<320;++k){
                buf[25088 + k] -= buf[51508 + k] * lr;
            }

            loss += buf[26296];
        }
        clock_t end_time = clock();
        double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        printf("Epoch %d loss = %f, acc = %f, elapsed time: %.6f seconds\n", j, loss / 60000.0f, correct/60000.0f, elapsed_time);
    }
}
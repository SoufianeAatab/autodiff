#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "conv2d_cmsis.cc"
#include "mnist_load.c"

bool check(float* a, float* b, int size) {
    for (int i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > 0.001) return false;
    }
    return true;
}

void init_weights(float* w, int size, float k){
    for(int i=0;i<size;++i){
        w[i] = (-1 +  2* ((float)rand() / (float)RAND_MAX)) * k;
    }
}

void print(float* a, int size) {
    for (int i = 0; i < size; ++i) {
        printf("%f, ", a[i]);
    }
    printf("\n");
}


void train(float* inputs, float* outputs, size_t input_size, size_t output_size, size_t data_size, size_t epochs){//===================================================
float lr = 0.01;
float* buf = (float*)calloc(51828, sizeof(float));
init_weights(&buf[794], 25088, 0.0012755102040816326); // (32, 784) 1
init_weights(&buf[25882], 320, 0.03125); // (10, 32) 2
//===================================================

 
for (size_t i = 0; i < epochs; ++i) {
    float loss = 0;
    int correct = 0;
    for (size_t j = 0; j < data_size; ++j) {    // buf[0] = Note: this ptr is where the input data should be.
    //buf[784] = output; // Note: this ptr is where output data should be.
    mat_mul(&buf[0] /* (1, 784) */, &buf[794] /* (784, 32) */, &buf[26202] /* (1, 32) */, 1, 784, 784, 1, 784, 32, 1, 784); // (1, 32) 5
    sigmoid(&buf[26202] /* (1, 32)*/ , &buf[26234] /*(1, 32)*/, 32); // (1, 32) 6
    mat_mul(&buf[26234] /* (1, 32) */, &buf[25882] /* (32, 10) */, &buf[26266] /* (1, 10) */, 1, 32, 32, 1, 32, 10, 1, 32); // (1, 10) 8
    log_softmax(&buf[26266], &buf[26276], 10); // (1, 10) 9
    nll_loss(&buf[26276], &buf[784], &buf[26286], 10);
    loss+=buf[26286]; // (1, 10) 10
    for(uint32_t k=0;k<10;++k){
    	buf[26296 + k] = 1.0f;}
    for(uint32_t k=0;k<10;++k){
    	buf[26306 + k] = -1;
    }
    mul(&buf[26296], &buf[26306], &buf[26316], 10); // (1, 10) 13
    mul(&buf[26316], &buf[784], &buf[26326], 10); // (1, 10) 14
    exp_(&buf[26276], &buf[26336], 10); // (1, 10) 15
    add(&buf[26336], &buf[26326], &buf[26346], 10); // (1, 10) 16
    mat_mul(&buf[26346] /* (1, 10) */, &buf[25882] /* (10, 32) */, &buf[26356] /* (1, 32) */, 1, 10, 10, 1, 10, 32, 32, 1); // (1, 32) 20
    sigmoid_diff(&buf[26202], &buf[26356], &buf[26388], 32); // (1, 32) 21
    mat_mul(&buf[26388] /* (32, 1) */, &buf[0] /* (1, 784) */, &buf[26420] /* (32, 784) */, 32, 1, 1, 32, 1, 784, 784, 1); // (32, 784) 24
    mat_mul(&buf[26346] /* (10, 1) */, &buf[26234] /* (1, 32) */, &buf[51508] /* (10, 32) */, 10, 1, 1, 10, 1, 32, 32, 1); // (10, 32) 19
    // sgd for 24
    for (uint32_t k=0;k<25088;++k){
    	buf[794 + k] -= buf[26420 + k] * lr;
    }
    // sgd for 19
    for (uint32_t k=0;k<320;++k){
    	buf[25882 + k] -= buf[51508 + k] * lr;
    }
    
    correct += accuracy(&buf[26286], &outputs[j*output_size], output_size);
    }
    printf("Loss: %f, acc:%f\n", loss / data_size, (float)correct / (float)data_size);
}
}

int main() {
    mnist_dataset_t ds = mnist_get_train();

    train(ds.images, ds.labels, 784, 10, 60000, 10);
    return 0;
}

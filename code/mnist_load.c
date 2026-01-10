#include <stdlib.h>
#include <stdio.h>
#include <zlib.h>
#include <time.h>
#include "conv2d_cmsis.cc"
#define XTRAIN_PATH "../data/train-images-idx3-ubyte.gz"
#define YTRAIN_PATH "../data/train-labels-idx1-ubyte.gz"

typedef struct {
    float* images;    // Flattened images
    float* labels;    // One-hot encoded labels
} mnist_dataset_t;

mnist_dataset_t mnist_get_train() {
    mnist_dataset_t ds = {};

    // Allocate memory for images as floats
    ds.images = (float*)calloc(784 * 60000, sizeof(float));
    if (!ds.images) {
        printf("Memory allocation for images failed.\n");
        exit(EXIT_FAILURE);
    }

    // Allocate memory for one-hot encoded labels
    ds.labels = (float*)calloc(10 * 60000, sizeof(float));
    if (!ds.labels) {
        printf("Memory allocation for labels failed.\n");
        free(ds.images);
        exit(EXIT_FAILURE);
    }

    // Open and read XTRAIN_PATH (images)
    gzFile x_train_file = gzopen(XTRAIN_PATH, "rb");
    if (!x_train_file) {
        printf("The XTRAIN file could not be opened. Exiting.\n");
        free(ds.images);
        free(ds.labels);
        exit(EXIT_FAILURE);
    }

    // Skip the 16-byte header of the images file
    if (gzseek(x_train_file, 16, SEEK_SET) != 16) {
        printf("Error seeking in XTRAIN file.\n");
        free(ds.images);
        free(ds.labels);
        gzclose(x_train_file);
        exit(EXIT_FAILURE);
    }

    // Allocate buffer for image data
    unsigned char* image_buffer = (unsigned char*)malloc(784 * 60000 * sizeof(unsigned char));
    if (!image_buffer) {
        printf("Memory allocation for image buffer failed.\n");
        free(ds.images);
        free(ds.labels);
        gzclose(x_train_file);
        exit(EXIT_FAILURE);
    }

    if (gzread(x_train_file, image_buffer, 784 * 60000) != 784 * 60000) {
        printf("Error reading XTRAIN file.\n");
        free(image_buffer);
        free(ds.images);
        free(ds.labels);
        gzclose(x_train_file);
        exit(EXIT_FAILURE);
    }
    gzclose(x_train_file);

    // Convert unsigned char images to floats and normalize to [0, 1]
    for (size_t i = 0; i < 784 * 60000; i++) {
        ds.images[i] = (float)image_buffer[i] / 255.0f;
    }
    free(image_buffer);

    // Open and read YTRAIN_PATH (labels)
    gzFile y_train_file = gzopen(YTRAIN_PATH, "rb");
    if (!y_train_file) {
        printf("The YTRAIN file could not be opened. Exiting.\n");
        free(ds.images);
        free(ds.labels);
        exit(EXIT_FAILURE);
    }

    // Skip the 8-byte header of the labels file
    if (gzseek(y_train_file, 8, SEEK_SET) != 8) {
        printf("Error seeking in YTRAIN file.\n");
        free(ds.images);
        free(ds.labels);
        gzclose(y_train_file);
        exit(EXIT_FAILURE);
    }

    // Allocate buffer for label data
    unsigned char* label_buffer = (unsigned char*)malloc(60000 * sizeof(unsigned char));
    if (!label_buffer) {
        printf("Memory allocation for label buffer failed.\n");
        free(ds.images);
        free(ds.labels);
        gzclose(y_train_file);
        exit(EXIT_FAILURE);
    }

    if (gzread(y_train_file, label_buffer, 60000) != 60000) {
        printf("Error reading YTRAIN file.\n");
        free(label_buffer);
        free(ds.images);
        free(ds.labels);
        gzclose(y_train_file);
        exit(EXIT_FAILURE);
    }
    gzclose(y_train_file);

    // Convert unsigned char labels to one-hot encoding
    for (size_t i = 0; i < 60000; i++) {
        unsigned char label = label_buffer[i];
        if (label >= 0 && label < 10) { // Ensure valid label range
            ds.labels[i * 10 + label] = 1.0f;
        }
    }
    free(label_buffer);

    return ds;
}

void init_weights(float* buff, uint32_t size, float k){
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

bool check(float* a, float* b, int size){
  for(int i=0;i<size;++i){
    if (abs(a[i] - b[i]) > 0.0000001) return false;
  }
  return true;
}

int main() {

  // mnist_dataset_t data = mnist_get_train();
  // float* input_ptr = data.images;
  // float* y_ptr = data.labels;
//===================================================
float lr = 0.01;
float* buf = (float*)calloc(134, sizeof(float));
//===================================================

//buf[0] = input;
//buf[4] = output;
float temp_2[16] = {2, 3, 0, 0, 2, 1, 2, 2, 2, 2, 3, 0, 3, 3, 3, 2, };
memcpy(&buf[8], temp_2, sizeof(float) * 16 );
float temp_0[4] = {2, 3, 0, 2, };
memcpy(&buf[0], temp_0, sizeof(float) * 4 );
mat_mul(&buf[0] /* (1, 4) */, &buf[8] /* (4, 4) */, &buf[92] /* (1, 4) */, 1, 4, 4, 1, 4, 4, 1, 4); // (1, 4) 5
sigmoid(&buf[92] /* (1, 4)*/ , &buf[96] /*(1, 4)*/, 4); // (1, 4) 6
float temp_3[68] = {1, 0, 1, 3, 3, 1, 1, 1, 3, 3, 0, 0, 3, 1, 1, 0, 3, 0, 0, 2, 2, 2, 1, 3, 3, 3, 3, 2, 1, 1, 2, 1, 2, 3, 2, 3, 3, 0, 2, 0, 2, 2, 0, 0, 2, 1, 3, 0, 3, 1, 1, 1,
 0, 1, 0, 1, 3, 3, 2, 3, 2, 3, 0, 3, 2, 2, 1, 0, };
memcpy(&buf[24], temp_3, sizeof(float) * 68 );
mat_mul(&buf[96] /* (1, 4) */, &buf[24] /* (4, 17) */, &buf[100] /* (1, 17) */, 1, 4, 4, 1, 4, 17, 1, 4); // (1, 17) 8
sigmoid(&buf[100] /* (1, 17)*/ , &buf[117] /*(1, 17)*/, 17); // (1, 17) 9

float ground[17] = 
{0.99330683212215, 0.9975272069299129, 0.9975272365314249, 0.9933066911402603, 0.9933071039207763, 0.9996646219302764, 0.9999832953530161, 0.9933064193249839, 0.9999545955285666, 0.993306500348989, 0.9820131201941611, 0.9975269885457263, 0.9975272069299129, 0.8807953238363039, 0.999983296111248, 0.9996646315510697, 0.9933065951278168, 
};
bool test = check(&buf[117], ground, 17);

if(test) printf("OK!");else printf("NO OK!");
    return 0;

}

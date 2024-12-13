#include <stdlib.h>
#include <stdio.h>
#include <zlib.h>

#define XTRAIN_PATH "../data/train-images-idx3-ubyte.gz"
#define YTRAIN_PATH "../data/train-labels-idx1-ubyte.gz"

struct mnist_dataset_t {
    float* images;    // Flattened images
    float* labels;    // One-hot encoded labels
};

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

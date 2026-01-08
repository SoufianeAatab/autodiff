#include <Metal/Metal.h> // Objective-C header for Metal
#include <time.h>
#include "mnist_load.c"
#include "conv2d_cmsis.cc"

#import <Foundation/Foundation.h> // For NSString

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
const char* mat_mul_kernel = R"(
#include <metal_stdlib>
using namespace metal;

kernel void matrix_multiply(const device float *X [[ buffer(0) ]],
                             const device float *W [[ buffer(1) ]],
                             device float *C [[ buffer(2) ]],
                             constant uint &N_COLS [[ buffer(3) ]],
                             uint2 id [[ thread_position_in_grid ]]) {
    uint col = id.y; 
    uint row = id.x;
    // if (row >= N_COLS || col >= N_COLS) return;

    float sum = 0.0;
    for (uint i = 0; i < 784; i++) {
        sum += X[row * 784 + i] * W[col * 784 + i]; 
    }
    if (row < 500 && col < 32) {
        C[row * 32 + col] = sum;
    }
}
)";

const char* mat_mul_kernel = R"(
#include <metal_stdlib>
using namespace metal;

kernel void sigmoid(const device float *X [[ buffer(0) ]],
                             const device float *W [[ buffer(1) ]],
                             device float *C [[ buffer(2) ]],
                             constant uint &N_COLS [[ buffer(3) ]],
                             uint2 id [[ thread_position_in_grid ]]) {
    uint col = id.y; 
    uint row = id.x;
    // if (row >= N_COLS || col >= N_COLS) return;

    float sum = 0.0;
    for (uint i = 0; i < 784; i++) {
        sum += X[row * 784 + i] * W[col * 784 + i]; 
    }
    if (row < 500 && col < 32) {
        C[row * 32 + col] = sum;
    }
}
)";

float* mat_mul_gpu(float* A, float* B, const unsigned int R, const unsigned int C, const unsigned int C2,  id<MTLBuffer> bufferA, id<MTLBuffer> bufferB, id<MTLBuffer> bufferC, id<MTLBuffer> bufferD, id<MTLComputePipelineState> pipelineState, id<MTLCommandQueue> commandQueue){
    memcpy([bufferA contents], A, sizeof(float) * R * C);
    memcpy([bufferB contents], B, sizeof(float) * C * C2);
    memcpy([bufferD contents], &C, sizeof(unsigned int));

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:pipelineState];
    [computeEncoder setBuffer:bufferA offset:0 atIndex:0];
    [computeEncoder setBuffer:bufferB offset:0 atIndex:1];
    [computeEncoder setBuffer:bufferC offset:0 atIndex:2];
    [computeEncoder setBuffer:bufferD offset:0 atIndex:3];

    MTLSize gridSize = MTLSizeMake(R, C2, 1);
    MTLSize threadGroupSize = MTLSizeMake(C2, C2, 1);
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    float* result = (float*)[bufferC contents];
    return result;
}

int main() {
    srand(0);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        printf("Metal is not supported on this device!\n");
        return -1;
    }

    NSString *shaderNSString = [NSString stringWithCString:mat_mul_kernel encoding:NSUTF8StringEncoding];
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:shaderNSString options:nil error:&error];
    if (error) {
        printf("Error loading shader: %s \n" , error);
        return -1;
    }

    id<MTLFunction> function = [library newFunctionWithName:@"matrix_multiply"];
    if (function == nil)
    {
        printf("Failed to find the adder function.\n");
        return -1;
    }
    
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
    if (error) {
        printf("Error creating pipeline state: %s\n",error);
        return -1;
    }
    mnist_dataset_t ds = mnist_get_train();
    id<MTLBuffer> bufferA = [device newBufferWithLength:sizeof(float) * 500 * 784 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [device newBufferWithLength:sizeof(float) * 784 * 32 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferC = [device newBufferWithLength:sizeof(float) * 500 * 32 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferD = [device newBufferWithLength:sizeof(unsigned int) options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];

    float* buf = (float*)calloc(51828, sizeof(float));
    set_weights(&buf[0], 25088, 0.0012755102040816326); // (32, 784)
    set_weights(&buf[25088], 320, 0.03125); // (10, 32)

    // For batch computation we need to accumulate weights, these are temporary buffers
    float* dw1 = (float*)calloc(784*32,sizeof(float));
    float* dw2 = (float*)calloc(10*32,sizeof(float));
    printf("Starting training\n");
    float lr = 0.1;
    // Number of repetitions for the matrix multiplication
    for (uint32_t j=0;j<50;++j){
        float loss = 0.0f;
        clock_t start_time = clock();
        uint32_t correct = 0;
        for(size_t p=0;p<60000;p+=500){
            float* result = mat_mul_gpu(&ds.images[p*784], &buf[0], 500, 784, 32, bufferA, bufferB, bufferC, bufferD, pipelineState, commandQueue);
            
            for(uint32_t l=0;l<500;++l){
                for(uint32_t k=0;k<10;++k){
                    buf[26202 + k] = -1;
                }
                //mat_mul(&ds.images[l*784] /* (1, 784) */, &buf[0] /* (784, 32) */, &result[0] /* (1, 32) */, 1, 784, 784, 1, 784, 32, 1, 784); // (1, 32) 5
                sigmoid(&result[l*32] /* (1, 32)*/ , &buf[26244] /*(1, 32)*/, 32); // (1, 32) 6
                mat_mul(&buf[26244] /* (1, 32) */, &buf[25088] /* (32, 10) */, &buf[26276] /* (1, 10) */, 1, 32, 32, 1, 32, 10, 1, 32); // (1, 10) 8
                log_softmax(&buf[26276], &buf[26286], 10); // (1, 10) 9
                if(accuracy(&buf[26286], &ds.labels[(p+l)*10], 10)) correct +=1;

                buf[26296] = nll_loss(&buf[26286], &ds.labels[(p+l)*10], 10); // (1, 10) 10
                for(uint32_t k=0;k<10;++k){
                    buf[26306 + k] = 1.0f;
                }
                mul(&buf[26306], &buf[26202], &buf[26316], 10); // (1, 10) 16
                mul(&buf[26316], &ds.labels[(p+l)*10], &buf[26326], 10); // (1, 10) 17
                exp(&buf[26286], &buf[26336], 10); // (1, 10) 18
                add(&buf[26336], &buf[26326], &buf[26346], 10); // (1, 10) 19
                mat_mul(&buf[26346] /* (1, 10) */, &buf[25088] /* (10, 32) */, &buf[26356] /* (1, 32) */, 1, 10, 10, 1, 10, 32, 32, 1); // (1, 32) 23
                sigmoid_diff(&result[l*32], &buf[26356], &buf[26388], 32); // (1, 32) 24
                
                mat_mul(&buf[26388] /* (32, 1) */, &ds.images[(p + l)*784] /* (1, 784) */, &dw1[0] /* (32, 784) */, 32, 1, 1, 32, 1, 784, 784, 1); // (32, 784) 27
                mat_mul(&buf[26346] /* (10, 1) */, &buf[26244] /* (1, 32) */, &dw2[0] /* (10, 32) */, 10, 1, 1, 10, 1, 32, 32, 1); // (10, 32) 22
                for(size_t k=0;k<25088;++k){
                    buf[26420+k] += dw1[0+k];
                }
                for(size_t k=0;k<320;++k){
                    buf[51508+k] += dw2[0+k];
                }
                loss += buf[26296];
            }
            // sgd for 27
            for (uint32_t k=0;k<25088;++k){
                buf[0 + k] -= (buf[26420 + k] / 500.0f) * lr;
                buf[26420 + k] = 0;
            }
            // sgd for 22
            for (uint32_t k=0;k<320;++k){
                buf[25088 + k] -= (buf[51508 + k] / 500.0f) * lr;
                buf[51508 + k] = 0;
            }
        }
        clock_t end_time = clock();
        double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        printf("Epoch %d loss = %f, acc = %f, elapsed time: %.6f seconds\n", j, loss / 60000.0f, correct/60000.0f, elapsed_time);
    }
    bufferA = nullptr;  // Set to nullptr to release
    bufferB = nullptr;  // Set to nullptr to release
    bufferC = nullptr;  // Set to nullptr to release
    commandQueue = nullptr;
    free(dw1);
    free(dw2);
    free(buf);
    free(ds.images);
    free(ds.labels);
    device = nullptr; // Allow the device to be deallocated

    return 0;
}

int main(int argc, char** argv){
    // mat_mul(&input_ptr[0] /* (32, 784) */, &buf[0] /* (784, 32) */, &e_muls[0] /* (32, 32) */, 32, 784, 784, 1, 784, 32, 1, 784); // (32, 32) 5
    // float* gpu_muls = mat_mul_gpu(&input_ptr[0], &buf[0], bufferA, bufferB, bufferC, pipelineState, commandQueue);
    // for(size_t i=0;i<32*32;++i){
    //     if(abs(gpu_muls[i] - e_muls[i]) > 0.1){
    //         printf("i=%zu: %f != %f\n", i, gpu_muls[i], e_muls[i]);
    //         break;
    //     }
    // }
}
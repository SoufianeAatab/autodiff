clang -o test test_ops.c -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -framework Accelerate -lz && ./test

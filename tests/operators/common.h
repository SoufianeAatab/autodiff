#include <time.h>
#include <chrono>

#define CHECK(_VALUE) \
    { \
        uint32_t error = _VALUE; \
        if (error != 0) { \
            fprintf(stderr, "Error: %d.\n", error); \
            exit(error); \
        } \
    }

template<typename T>
void random_fill(T* p, size_t sz) {
    T* end = p + sz;
    while (p != end) {
        *p++ = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX));
    }
}

template<typename T>
void cmp_data(T* gold, T* data, size_t sz) {
    T* end = gold + sz;
    while (gold != end) {
        if (*gold != *data) {
            fprintf(stderr, "Gold data do not match with operator data in %i: %f != %f.\n", uint32_t((uint64_t)end-(uint64_t)gold), *gold, *data);
            break;
        }
        ++gold;
        ++data;
    }
}

#define TIME(_SCOPE) \
    ({ \
        auto begin = std::chrono::high_resolution_clock::now(); \
        _SCOPE; \
        auto end = std::chrono::high_resolution_clock::now(); \
        std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count(); \
    })

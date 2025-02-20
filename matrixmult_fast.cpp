#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <bits/stdc++.h>
#include <omp.h>
#include <immintrin.h>
#include <chrono>
#include <cstdio>

/**
 * Techniques I used in the end:
 * - AVX-512 instructions
 * - Three-level cache blocking
 * - 12 x 16 kernel for calculations
 * - Manual loop unrolling
 * - Branch-less programming for processing input/output
 *
 * References:
 * - https://en.algorithmica.org/hpc/algorithms/matmul/
 * - https://en.algorithmica.org/hpc/pipelining/branchless/
 */

typedef int vec __attribute__((vector_size(64))); // not sure why 64 works here

// Trial and error to get these -- worked better than my calculated values..
static const int OFFSET_1 = 512; // 512
static const int OFFSET_2 = 2048; // 2048
static const int OFFSET_3 = 48; // 48
static const int KERNEL_HEIGHT = 12;
static const int KERNEL_WIDTH = 16;

static __attribute__((always_inline)) void *alloc64(std::size_t n_bytes) {
    return std::aligned_alloc(64, n_bytes);
}

__attribute__((optimize("unroll-loops")))
__attribute__((hot))
__attribute__((target("avx512f")))
inline __attribute__((always_inline)) void kernel_dyn(const int *__restrict__ a,
                                                      const vec *__restrict__ b,
                                                      vec *__restrict__ c,
                                                      int x, int y,
                                                      int start, int end,
                                                      int stride) {
    int rowStart[KERNEL_HEIGHT];
    for (int i = 0; i < KERNEL_HEIGHT; i++) {
        rowStart[i] = stride * (x + i);
    }

    vec v00 = (vec){};
    vec v01 = (vec){};
    vec v02 = (vec){};
    vec v03 = (vec){};
    vec v04 = (vec){};
    vec v05 = (vec){};
    vec v06 = (vec){};
    vec v07 = (vec){};
    vec v08 = (vec){};
    vec v09 = (vec){};
    vec v10 = (vec){};
    vec v11 = (vec){};

    for (int k = start; k < end; k++) {
        int bIndex = (k * stride + y) / 16;
        vec bVector = b[bIndex];

        vec aVector0 = (vec){} + a[rowStart[0] + k];
        vec aVector1 = (vec){} + a[rowStart[1] + k];
        vec aVector2 = (vec){} + a[rowStart[2] + k];
        vec aVector3 = (vec){} + a[rowStart[3] + k];
        vec aVector4 = (vec){} + a[rowStart[4] + k];
        vec aVector5 = (vec){} + a[rowStart[5] + k];
        vec aVector6 = (vec){} + a[rowStart[6] + k];
        vec aVector7 = (vec){} + a[rowStart[7] + k];
        vec aVector8 = (vec){} + a[rowStart[8] + k];
        vec aVector9 = (vec){} + a[rowStart[9] + k];
        vec aVector10 = (vec){} + a[rowStart[10] + k];
        vec aVector11 = (vec){} + a[rowStart[11] + k];

        v00 += aVector0 * bVector;
        v01 += aVector1 * bVector;
        v02 += aVector2 * bVector;
        v03 += aVector3 * bVector;
        v04 += aVector4 * bVector;
        v05 += aVector5 * bVector;
        v06 += aVector6 * bVector;
        v07 += aVector7 * bVector;
        v08 += aVector8 * bVector;
        v09 += aVector9 * bVector;
        v10 += aVector10 * bVector;
        v11 += aVector11 * bVector;
    }

    c[(rowStart[0] + y) / KERNEL_WIDTH] += v00;
    c[(rowStart[1] + y) / KERNEL_WIDTH] += v01;
    c[(rowStart[2] + y) / KERNEL_WIDTH] += v02;
    c[(rowStart[3] + y) / KERNEL_WIDTH] += v03;
    c[(rowStart[4] + y) / KERNEL_WIDTH] += v04;
    c[(rowStart[5] + y) / KERNEL_WIDTH] += v05;
    c[(rowStart[6] + y) / KERNEL_WIDTH] += v06;
    c[(rowStart[7] + y) / KERNEL_WIDTH] += v07;
    c[(rowStart[8] + y) / KERNEL_WIDTH] += v08;
    c[(rowStart[9] + y) / KERNEL_WIDTH] += v09;
    c[(rowStart[10] + y) / KERNEL_WIDTH] += v10;
    c[(rowStart[11] + y) / KERNEL_WIDTH] += v11;
}


__attribute__((optimize("unroll-loops")))
__attribute__((target("avx512f")))
inline __attribute__((always_inline)) void kernel_tail(const int *__restrict__ a,
                                                       const vec *__restrict__ bVector,
                                                       vec *__restrict__ cVector,
                                                       int x, int y,
                                                       int height, int width,
                                                       int start, int end,
                                                       int stride) {
    const int *b = reinterpret_cast<const int *>(bVector);
    int *c = reinterpret_cast<int *>(cVector);

    for (int i = 0; i < height; i++) {
        int aRowStart = (x + i) * stride;
        int cRowStart = (x + i) * stride;

        for (int k = start; k < end; k++) {
            int aValue = a[aRowStart + k];
            int bRowStart = k * stride;

            for (int j = 0; j < width; j++) {
                c[cRowStart + (y + j)] += aValue * b[bRowStart + (y + j)];
            }
        }
    }
}

__attribute__((optimize("unroll-loops")))
__attribute__((target("avx512f")))
inline __attribute__((always_inline)) void matmul6(const int *A, const int *B, int *C, int n, int stride) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int outsideStart = 0; outsideStart < n; outsideStart += OFFSET_3) {
        for (int middleStart = 0; middleStart < n; middleStart += OFFSET_2) {
            for (int innerStart = 0; innerStart < n; innerStart += OFFSET_1) {
                const int outsideEnd = (outsideStart + OFFSET_3 < n) ? (outsideStart + OFFSET_3) : n;
                const int middleEnd = (middleStart + OFFSET_2 < n) ? (middleStart + OFFSET_2) : n;
                const int innerEnd = (innerStart + OFFSET_1 < n) ? (innerStart + OFFSET_1) : n;

                for (int x = middleStart; x < middleEnd; x += KERNEL_HEIGHT) {
                    for (int y = outsideStart; y < outsideEnd; y += KERNEL_WIDTH) {
                        const int height = (x + KERNEL_HEIGHT <= middleEnd) ? KERNEL_HEIGHT : (middleEnd - x);
                        const int width = (y + KERNEL_WIDTH <= outsideEnd) ? KERNEL_WIDTH : (outsideEnd - y);

                        if (__builtin_expect(height == KERNEL_HEIGHT && width == KERNEL_WIDTH, 1)) {
                            kernel_dyn(A, (const vec *) B, (vec *) C,
                                       x, y,
                                       innerStart, innerEnd,
                                       stride);
                        } else {
                            kernel_tail(A, (const vec *) B, (vec *) C,
                                        x, y,
                                        height, width,
                                        innerStart, innerEnd,
                                        stride);
                        }
                    }
                }
            }
        }
    }
}

inline __attribute__((always_inline)) void matmul_naive(const int *A, const int *B, int *C, int n, int stride) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            int aValue = A[i * stride + k];
            for (int j = 0; j < n; j++) {
                C[i * stride + j] += aValue * B[k * stride + j];
            }
        }
    }
}

inline __attribute__((always_inline)) void matmul_naive_parallel(const int *A, const int *B, int *C, int n, int stride) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            int aValue = A[i * stride + k];
            for (int j = 0; j < n; j++) {
                C[i * stride + j] += aValue * B[k * stride + j];
            }
        }
    }
}

static const int BUFFER_SIZE = 1024 * (1 << 20); // 1 GB
static char inputBuffer[BUFFER_SIZE];
static int inputPos = 0, inputLen = 0;
static char *outputPtr = (char *) &inputBuffer;

inline __attribute__((always_inline)) void initialRead() {
    inputLen = (int) fread(inputBuffer, 1, BUFFER_SIZE, stdin);
    inputPos = 0;
}

inline __attribute__((always_inline)) int getChar() {
    return inputBuffer[inputPos++];
}

inline __attribute__((always_inline)) void readInt(int &number) {
    int &position = inputPos;
    char *buffer = inputBuffer;

    char c = buffer[position++];
    while (__builtin_expect(c == ' ' || c == '\n', 1)) {
        c = buffer[position++];
    }

    int is_negative = (c == '-');
    c = buffer[position - 1 + is_negative];
    position += is_negative;

    int sign = 1 - (is_negative << 1);
    number = 0;

    // Manual unrolling
    if ((unsigned) (c - '0') <= 9) {
        number = number * 10 + (c - '0');
        c = buffer[position++];
        if ((unsigned) (c - '0') <= 9) {
            number = number * 10 + (c - '0');
            c = buffer[position++];
            if ((unsigned) (c - '0') <= 9) {
                number = number * 10 + (c - '0');
                c = buffer[position++];
                if ((unsigned) (c - '0') <= 9) {
                    number = number * 10 + (c - '0');
                    c = buffer[position++];
                }
            }
        }
    }

    while (__builtin_expect((unsigned) (c - '0') <= 9, 0)) {
        number = number * 10 + (c - '0');
        c = buffer[position++];
    }

    number *= sign;
}


inline __attribute__((always_inline)) void putChar(char c) {
    *(outputPtr++) = c;
}

inline __attribute__((always_inline)) void flushOutput() {
    fwrite(inputBuffer, 1, (size_t) (outputPtr - inputBuffer), stdout);
}

inline __attribute__((always_inline)) void writeInt(int number) {
    static const int divide_lookup[100] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        9, 9, 9, 9, 9, 9, 9, 9, 9, 9
    };

    static const int modulo_lookup[100] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    if (number < 0) {
        putChar('-');
        number = -number;
    }

    char buf[12];
    int len = 0;

    if (__builtin_expect(number < 100, 1)) {
        if (number >= 10) {
            buf[len++] = char('0' + modulo_lookup[number]);
            number = divide_lookup[number];
        }
        buf[len++] = char('0' + number);
    } else {
        do {
            buf[len++] = char('0' + (number % 10));
            number /= 10;
        } while (number);
    }

    for (int i = len - 1; i >= 0; i--) {
        putChar(buf[i]);
    }

    putChar(' ');
}


inline void printMatrix(int *C, int n, int stride) {
    const char *start = "The resulting matrix C = A x B is:\n";
    for (const char *p = start; *p; p++) putChar(*p);

    for (int i = 0; i < n; i++) {
        int base = i * stride;
        for (int j = 0; j < n; j++) {
            writeInt(C[base + j]);
        }
        putChar('\n');
    }

    flushOutput();
}

#define TIMING 0

__attribute__((optimize("unroll-loops")))
__attribute__((target("avx512f")))
int main() {
#if TIMING == 1
    auto start = std::chrono::high_resolution_clock::now();
#endif
    int n;
    std::ios::sync_with_stdio(0);
    std::cin.tie(0);
    std::cout.tie(0);
#if TIMING == 1
    std::cerr << std::fixed << std::setprecision(3);
#endif

    initialRead();
    readInt(n);
#if TIMING == 1
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cerr << "n: " << elapsed.count() << " seconds";
#endif

#if TIMING == 1
    start = std::chrono::high_resolution_clock::now();
#endif
    // TODO: perform matrix multiplication A x B and write into C: C = A x B
    int n_round = (n + KERNEL_HEIGHT - 1) / KERNEL_HEIGHT * KERNEL_HEIGHT;
    if (n_round % KERNEL_WIDTH != 0) {
        n_round = ((n_round + KERNEL_WIDTH - 1) / KERNEL_WIDTH) * KERNEL_WIDTH;
    }

    size_t mem_size = n_round * n_round * sizeof(int);
    int *aPtr = static_cast<int *>(alloc64(mem_size));
    int *bPtr = static_cast<int *>(alloc64(mem_size));
    int *C = static_cast<int *>(alloc64(mem_size));
    std::memset(C, 0, mem_size);

#if TIMING == 1
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cerr << ", alloc: " << elapsed.count() << " seconds";
#endif

#if TIMING == 1
    start = std::chrono::high_resolution_clock::now();
#endif

    for (int i = 0; i < n; i++) {
        int base = i * n_round;
        for (int j = 0; j < n; j++) {
            readInt(aPtr[base + j]);
        }
    }

    for (int i = 0; i < n; i++) {
        int base = i * n_round;
        for (int j = 0; j < n; j++) {
            readInt(bPtr[base + j]);
        }
    }

#if TIMING == 1
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cerr << ", read_in: " << elapsed.count() << " seconds";
#endif

#if TIMING == 1
    start = std::chrono::high_resolution_clock::now();
#endif
    matmul6(aPtr, bPtr, C, n, n_round);
#if TIMING == 1
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cerr << ", mult: " << elapsed.count() << " seconds";
#endif

#if TIMING == 1
    start = std::chrono::high_resolution_clock::now();
#endif
    printMatrix(C, n, n_round);

#if TIMING == 1
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cerr << ", output: " << elapsed.count() << " seconds";
#endif

    return 0;
}

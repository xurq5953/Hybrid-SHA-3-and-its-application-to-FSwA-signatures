#ifndef DILITHIUM_AVX2_NTT_H
#define DILITHIUM_AVX2_NTT_H

#include <immintrin.h>
#include <stdint.h>
#include "../params.h"


void shuffle(int32_t  *c);

void pointwise_avx(int32_t c[N],
                   const int32_t a[N],
                   const int32_t b[N],
                   const int32_t *qdata);

void pointwise_acc_avx(int32_t c[N],
                       const int32_t *a,
                       const int32_t *b,
                       const int32_t *qdata);

void ntt_bo_avx512(int32_t a[N]);

void ntt_so_avx512(int32_t a[N]);

void intt_bo_avx512(int32_t a[N]);

void intt_so_avx512(int32_t a[N]);

void pointwise_avx512(int32_t c[N],
                      const int32_t a[N],
                      const int32_t b[N]);


#endif

#ifndef HAETAE_SAMPLER_H
#define HAETAE_SAMPLER_H

#include "fixpoint.h"
#include "params.h"
#include "reduce.h"
#include <stdint.h>
#include "align.h"
#include "symmetric.h"


// #define GAUSS_RAND (72 + 16 + 48)
// #define GAUSS_RAND_BYTES ((GAUSS_RAND + 7) / 8)
// #define NUM_GAUSSIANS 278 // ca 80% require up to 272 Gaussian samples
// #define POLY_HYPERBALL_BUFLEN_4X (GAUSS_RAND_BYTES * NUM_GAUSSIANS + N / 8)
// #define POLY_HYPERBALL_NBLOCKS_4X ((POLY_HYPERBALL_BUFLEN_4X + STREAM256_BLOCKBYTES - 1) / STREAM256_BLOCKBYTES)

#define rej_uniform HAETAE_NAMESPACE(rej_uniform)
unsigned int rej_uniform(int32_t *a, unsigned int len, const uint8_t *buf,
                         unsigned int buflen);

unsigned int rej_eta(int32_t *a, unsigned int len, const uint8_t *buf,
                     unsigned int buflen);

#define sample_gauss_N HAETAE_NAMESPACE(sample_gauss_N)
void sample_gauss_N(uint64_t *r, uint8_t *signs, fp96_76 *sqsum,
                    const uint8_t seed[CRHBYTES], const uint16_t nonce,
                    const size_t len);
int sample_gauss(uint64_t *r, fp96_76 *sqsum, const uint8_t *buf, const size_t buflen, const size_t len, const int dont_write_last);

void sample_gauss_N_4x(uint64_t *r0, uint64_t *r1, uint64_t *r2, uint64_t *r3, 
    uint8_t *signs0, uint8_t *signs1, uint8_t *signs2, uint8_t *signs3,
    fp96_76 *sqsum,  const uint8_t seed[CRHBYTES], 
    const uint16_t nonce0, const uint16_t nonce1, const uint16_t nonce2, const uint16_t nonce3,
    const size_t len0, const size_t len1, const size_t len2, const size_t len3);


unsigned int rej_eta_avx2(int32_t *a, unsigned int len, const uint8_t *buf,
                     unsigned int buflen);

#endif

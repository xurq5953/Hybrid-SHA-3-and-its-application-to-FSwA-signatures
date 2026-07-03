#ifndef REJSAMPLE_H
#define REJSAMPLE_H

#include <stdint.h>
#include "params.h"
#include "polyvec.h"

unsigned int rej_uniform_avx(int32_t *restrict r, const uint8_t *buf, unsigned int num);


unsigned int rej_eta_avx(int32_t *r,
                         unsigned int len,
                         const uint8_t *buf,
                         unsigned int buflen);

unsigned int rej_gamma1m1_avx(int32_t *r,
                              unsigned int len,
                              const uint8_t *buf,
                              unsigned int buflen);


#define REJ_UNIFORM_NBLOCKS ((768+STREAM128_BLOCKBYTES-1)/STREAM128_BLOCKBYTES) // 5
#define REJ_UNIFORM_BUFLEN (REJ_UNIFORM_NBLOCKS*STREAM128_BLOCKBYTES) // 840

#if ETA == 2
#define REJ_UNIFORM_ETA_NBLOCKS ((136+STREAM256_BLOCKBYTES-1)/STREAM256_BLOCKBYTES) // 1
#elif ETA == 4
#define REJ_UNIFORM_ETA_NBLOCKS ((227+STREAM256_BLOCKBYTES-1)/STREAM256_BLOCKBYTES)
#endif
#define REJ_UNIFORM_ETA_BUFLEN (REJ_UNIFORM_ETA_NBLOCKS*STREAM256_BLOCKBYTES) // 136

extern const uint8_t idxlut[256][8];

unsigned int XURQ_AVX2_rej_uniform_avx_s1s3(int32_t *restrict r, const uint8_t *buf, unsigned int num);

unsigned int XURQ_AVX2_rej_uniform_avx_s1s3_final(int32_t *restrict r, const uint8_t *buf, unsigned int num);

unsigned int XURQ_AVX2_rej_eta_avx_with_pack(int32_t *restrict r, uint8_t *pipe, const uint8_t buf[REJ_UNIFORM_ETA_BUFLEN]);

void poly_uniform_4x_op13(poly *a0, poly *a1, poly *a2, poly *a3, const uint8_t seed[32], uint16_t nonce0,
                          uint16_t nonce1, uint16_t nonce2, uint16_t nonce3);

void ExpandA_shuffled(polyvecl mat[K], const uint8_t rho[32]);

void ExpandA(polyvecl mat[K], const uint8_t *rho);

void ExpandA_row(polyvecl **row, polyvecl buf[2], const uint8_t *rho, unsigned int i);

void pack_eta(uint8_t *r, const uint8_t *pipe);

void ExpandS_with_pack(polyvecl *s1,
                       polyveck *s2,
                       uint8_t *r,
                       const uint64_t seed[4]);

int rej_uniform_avx512(int32_t *r, uint8_t *buf);

int rej_uniform_avx512_SHAKE128RATE(int32_t *r, uint8_t *buf, uint32_t ctr);

void poly_uniform_8x(poly *restrict a0, poly *restrict a1, poly *restrict a2, poly *restrict a3,
                                 poly *restrict a4, poly *restrict a5, poly *restrict a6, poly *restrict a7,
                                 const uint64_t *restrict seed,
                                 uint16_t nonce0, uint16_t nonce1, uint16_t nonce2, uint16_t nonce3,
                                 uint16_t nonce4, uint16_t nonce5, uint16_t nonce6, uint16_t nonce7);

#endif

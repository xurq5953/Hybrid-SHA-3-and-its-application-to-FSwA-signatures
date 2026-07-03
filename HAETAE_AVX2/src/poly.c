#include "poly.h"
#include "consts.h"
#include "decompose.h"
#include "fips202x4.h"
#include "ntt.h"
#include "params.h"
#include "reduce.h"
#include "symmetric.h"
#include <stdint.h>
#include <string.h>

#include "hybrid.h"

/*************************************************
 * Name:        poly_add
 *
 * Description: Add polynomials. No modular reduction is performed.
 *
 * Arguments:   - poly *c: pointer to output polynomial
 *              - const poly *a: pointer to first summand
 *              - const poly *b: pointer to second summand
 **************************************************/
void poly_add(poly *c, const poly *a, const poly *b) {
    unsigned int i;
    __m256i f, g;

    for (i = 0; i < N / 8; i++) {
        f = _mm256_load_si256(&a->vec[i]);
        g = _mm256_load_si256(&b->vec[i]);
        f = _mm256_add_epi32(f, g);
        _mm256_store_si256(&c->vec[i], f);
    }
}

/*************************************************
 * Name:        poly_sub
 *
 * Description: Subtract polynomials. No modular reduction is
 *              performed.
 *
 * Arguments:   - poly *c: pointer to output polynomial
 *              - const poly *a: pointer to first input polynomial
 *              - const poly *b: pointer to second input polynomial to be
 *                               subtraced from first input polynomial
 **************************************************/
void poly_sub(poly *c, const poly *a, const poly *b) {
    unsigned int i;
    __m256i f, g;

    for (i = 0; i < N / 8; i++) {
        f = _mm256_load_si256(&a->vec[i]);
        g = _mm256_load_si256(&b->vec[i]);
        f = _mm256_sub_epi32(f, g);
        _mm256_store_si256(&c->vec[i], f);
    }
}

/*************************************************
 * Name:        poly_ntt
 *
 * Description: Inplace forward NTT. Coefficients can grow by up to
 *              8*Q in absolute value.
 *
 * Arguments:   - poly *a: pointer to input/output polynomial
 **************************************************/
void poly_ntt(poly *a) { ntt_avx(a->vec, qdata.vec); }

/*************************************************
 * Name:        poly_invntt_tomont
 *
 * Description: Inplace inverse NTT and multiplication by 2^{32}.
 *              Input coefficients need to be less than Q in absolute
 *              value and output coefficients are again bounded by Q.
 *
 * Arguments:   - poly *a: pointer to input/output polynomial
 **************************************************/
void poly_invntt_tomont(poly *a) { invntt_avx(a->vec, qdata.vec); }

void poly_nttunpack(poly *a) { nttunpack_avx(a->vec); }

/*************************************************
 * Name:        poly_pointwise_montgomery
 *
 * Description: Pointwise multiplication of polynomials in NTT domain
 *              representation and multiplication of resulting polynomial
 *              by 2^{-32}.
 *
 * Arguments:   - poly *c: pointer to output polynomial
 *              - const poly *a: pointer to first input polynomial
 *              - const poly *b: pointer to second input polynomial
 **************************************************/
void poly_pointwise_montgomery(poly *c, const poly *a, const poly *b) {
    pointwise_avx(c->vec, a->vec, b->vec, qdata.vec);
}

/*************************************************
 * Name:        poly_reduce2q
 *
 * Description: Inplace reduction of all coefficients of polynomial to 2q
 *
 * Arguments:   - poly *a: pointer to input/output polynomial
 **************************************************/
static const union {
  __m256i vec[7];
  int64_t arr[7*4];
} freeze_dat_avx = {.arr = {
  DQREC, DQREC, DQREC, DQREC,
  DQ, DQ, DQ, DQ, 
  DQ*2, DQ*2, DQ*2, DQ*2, 
  (1ULL<<32)-1, (1ULL<<32)-1, (1ULL<<32)-1, (1ULL<<32)-1, 
  -1, -1, -1, -1, 
  0,0,0,0,
  Q,Q,Q,Q,
}};
void poly_reduce2q(poly *a) {
    unsigned int i;
    __m256i tmp[6];
/*
    int64_t t = (int64_t)a * DQREC;
    t >>= 32;
    t = a - t * DQ;              // -4Q <  t < 4Q
    t += (t >> 31) & (DQ * 2);   //   0 <= t < 4Q
    t -= ~((t - DQ) >> 31) & DQ; //   0 <= t < Q
    t -= ~((t - Q) >> 31) & DQ;  // centered representation
    return (int32_t)t;
*/
    for (i = 0; i < N/8; ++i)
    {
      tmp[0] = _mm256_and_si256(a->vec[i], freeze_dat_avx.vec[3]);
      tmp[1] = _mm256_srli_epi64(a->vec[i], 32);

      // int64_t t = (int64_t)a * DQREC;
      tmp[2] = _mm256_mul_epi32(a->vec[i], freeze_dat_avx.vec[0]);
      tmp[3] = _mm256_mul_epi32(tmp[1], freeze_dat_avx.vec[0]);

      // t >>= 32;
      tmp[2] = _mm256_srli_epi64(tmp[2], 32);
      tmp[3] = _mm256_srli_epi64(tmp[3], 32);

      // t = a - t * DQ;              // -4Q <  t < 4Q
      tmp[4] = _mm256_mul_epi32(tmp[2], freeze_dat_avx.vec[1]);
      tmp[5] = _mm256_mul_epi32(tmp[3], freeze_dat_avx.vec[1]);

      tmp[2] = _mm256_sub_epi64(tmp[0], tmp[4]);
      tmp[3] = _mm256_sub_epi64(tmp[1], tmp[5]);


      // t += (t >> 31) & (DQ * 2);   //   0 <= t < 4Q
      tmp[4] = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(freeze_dat_avx.vec[5]), _mm256_castsi256_ps(freeze_dat_avx.vec[2]), _mm256_castsi256_ps(tmp[2])));
      tmp[5] = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(freeze_dat_avx.vec[5]), _mm256_castsi256_ps(freeze_dat_avx.vec[2]), _mm256_castsi256_ps(tmp[3])));

      tmp[2] = _mm256_add_epi64(tmp[2], tmp[4]);
      tmp[3] = _mm256_add_epi64(tmp[3], tmp[5]);


      // t -= ~((t - DQ) >> 31) & DQ; //   0 <= t < Q
      tmp[4] = _mm256_sub_epi32(tmp[2], freeze_dat_avx.vec[1]);
      tmp[5] = _mm256_sub_epi32(tmp[3], freeze_dat_avx.vec[1]);

      tmp[4] = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(freeze_dat_avx.vec[1]), _mm256_castsi256_ps(freeze_dat_avx.vec[5]), _mm256_castsi256_ps(tmp[4])));
      tmp[5] = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(freeze_dat_avx.vec[1]), _mm256_castsi256_ps(freeze_dat_avx.vec[5]), _mm256_castsi256_ps(tmp[5])));

      tmp[2] = _mm256_sub_epi64(tmp[2], tmp[4]);
      tmp[3] = _mm256_sub_epi64(tmp[3], tmp[5]);


      // t -= ~((t - Q) >> 31) & DQ;  // centered representation
      tmp[4] = _mm256_sub_epi32(tmp[2], freeze_dat_avx.vec[6]);
      tmp[5] = _mm256_sub_epi32(tmp[3], freeze_dat_avx.vec[6]);

      tmp[4] = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(freeze_dat_avx.vec[1]), _mm256_castsi256_ps(freeze_dat_avx.vec[5]), _mm256_castsi256_ps(tmp[4])));
      tmp[5] = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(freeze_dat_avx.vec[1]), _mm256_castsi256_ps(freeze_dat_avx.vec[5]), _mm256_castsi256_ps(tmp[5])));

      tmp[2] = _mm256_sub_epi32(tmp[2], tmp[4]);
      tmp[3] = _mm256_sub_epi32(tmp[3], tmp[5]);
      tmp[3] = _mm256_slli_epi64(tmp[3], 32);

      a->vec[i] = _mm256_blend_epi32(tmp[2], tmp[3], 0xaa);
    }
}

/*************************************************
 * Name:        poly_freeze2q
 *
 * Description: For all coefficients of in/out polynomial compute standard
 *              representative r = a mod^+ 2Q
 *
 * Arguments:   - poly *a: pointer to input/output polynomial
 **************************************************/
void poly_freeze2q(poly *a) {
    unsigned int i;
    __m256i tmp[6];
/*
    int64_t t = (int64_t)a * DQREC;
    t >>= 32;
    t = a - t * DQ;              // -4Q <  t < 4Q
    t += (t >> 31) & (DQ * 2);   //   0 <= t < 4Q
    t -= ~((t - DQ) >> 31) & DQ; //   0 <= t < Q
    return (int32_t)t;
*/

    for (i = 0; i < N/8; ++i)
    {
      tmp[0] = _mm256_srli_epi64(a->vec[i], 32);
      tmp[3] = _mm256_and_si256(a->vec[i], freeze_dat_avx.vec[3]);

      // int64_t t = (int64_t)a * DQREC;
      tmp[1] = _mm256_mul_epi32(a->vec[i], freeze_dat_avx.vec[0]);
      tmp[2] = _mm256_mul_epi32(tmp[0], freeze_dat_avx.vec[0]);

      // t >>= 32
      tmp[1] = _mm256_srli_epi64(tmp[1], 32);
      tmp[2] = _mm256_srli_epi64(tmp[2], 32);


      // t = a - t * DQ;              // -4Q <  t < 4Q
      tmp[1] = _mm256_mul_epi32(tmp[1], freeze_dat_avx.vec[1]);
      tmp[2] = _mm256_mul_epi32(tmp[2], freeze_dat_avx.vec[1]);

      tmp[1] = _mm256_sub_epi64(tmp[3], tmp[1]);
      tmp[2] = _mm256_sub_epi64(tmp[0], tmp[2]);


      // t += (t >> 31) & (DQ * 2);   //   0 <= t < 4Q
      tmp[4] = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(freeze_dat_avx.vec[5]), _mm256_castsi256_ps(freeze_dat_avx.vec[2]), _mm256_castsi256_ps(tmp[1])));
      tmp[5] = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(freeze_dat_avx.vec[5]), _mm256_castsi256_ps(freeze_dat_avx.vec[2]), _mm256_castsi256_ps(tmp[2])));

      tmp[1] = _mm256_add_epi32(tmp[1], tmp[4]);
      tmp[2] = _mm256_add_epi32(tmp[2], tmp[5]);

      // t -= ~((t - DQ) >> 31) & DQ; //   0 <= t < Q
      tmp[4] = _mm256_sub_epi32(tmp[1], freeze_dat_avx.vec[1]);
      tmp[5] = _mm256_sub_epi32(tmp[2], freeze_dat_avx.vec[1]);
   
      tmp[4] = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(freeze_dat_avx.vec[1]), _mm256_castsi256_ps(freeze_dat_avx.vec[5]), _mm256_castsi256_ps(tmp[4])));
      tmp[5] = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(freeze_dat_avx.vec[1]), _mm256_castsi256_ps(freeze_dat_avx.vec[5]), _mm256_castsi256_ps(tmp[5])));

      tmp[1] = _mm256_sub_epi32(tmp[1], tmp[4]);
      tmp[2] = _mm256_sub_epi32(tmp[2], tmp[5]);
      tmp[2] = _mm256_slli_epi64(tmp[2], 32);

      a->vec[i] = _mm256_blend_epi32(tmp[1], tmp[2], 0xaa);
    }
}

/*************************************************
 * Name:        poly_freeze
 *
 * Description: For all coefficients of in/out polynomial compute standard
 *              representative r = a mod^+ Q
 *
 * Arguments:   - poly *a: pointer to input/output polynomial
 **************************************************/
void poly_freeze(poly *a) {
    unsigned int i;

    for (i = 0; i < N; ++i)
        a->coeffs[i] = freeze(a->coeffs[i]);
}

/*************************************************
 * Name:        poly_highbits
 *
 * Description: Compute HighBits of polynomial
 *
 * Arguments:   - poly *a2: pointer to output polynomial
 *              - const poly *a: pointer to input polynomial
 **************************************************/
 static const union {
  __m256i vec[4];
  int32_t arr[8*4];
 } decomp_data_avx = {.arr = {
  255, 255, 255, 255, 255, 255, 255, 255, 
  128, 128, 128, 128, 128, 128, 128, 128, 
  1,1,1,1,1,1,1,1,
  256, 256, 256, 256, 256, 256, 256, 256
 }};
void poly_highbits(poly *a2, const poly *a) {
    unsigned int i;

    for (i = 0; i < N/8; ++i)
    {
      //decompose_z1(&a2->coeffs[i], &a1tmp, a->coeffs[i]);
      a2->vec[i] = _mm256_add_epi32(a->vec[i], decomp_data_avx.vec[1]);
      a2->vec[i] = _mm256_srai_epi32(a2->vec[i], 8); // TODO magic number!
    }
}

/*************************************************
 * Name:        poly_lowbits
 *
 * Description: Compute LowBits of polynomial
 *
 * Arguments:   - poly *a1: pointer to output polynomial
 *              - const poly *a: pointer to input polynomial
 **************************************************/
void poly_lowbits(poly *a1, const poly *a) {
    unsigned int i = 0;
    __m256i lb, center;

    for (i = 0; i < N/8; ++i)
    {
      lb = _mm256_and_si256(a->vec[i], decomp_data_avx.vec[0]);
      center = _mm256_add_epi32(lb, decomp_data_avx.vec[2]); // lb+1
      center = _mm256_sub_epi32(decomp_data_avx.vec[1], center); // (alpha >> 1) - (lb + 1)
      center = _mm256_srai_epi32(center, 31);
      center = _mm256_and_si256(center, decomp_data_avx.vec[3]);
      a1->vec[i] = _mm256_sub_epi32(lb, center);
    }
}

/*************************************************
 * Name:        poly_compose
 *
 * Description: Compose HighBits and LowBits to recreate the polynomial
 *
 * Arguments:   - poly *a3: pointer to output polynomial
 *              - const poly *ha: pointer to HighBits polynomial
 *              - const poly *la: pointer to HighBits polynomial
 **************************************************/
void poly_compose(poly *a, const poly *ha, const poly *la) {
    unsigned int i = 0;
    __m256i tmp;

    for (i = 0; i < N/8; ++i)
    {
        tmp = _mm256_slli_epi32(ha->vec[i], 8);
        a->vec[i] = _mm256_add_epi32(tmp, la->vec[i]);
    }
}

/*************************************************
 * Name:        poly_lsb
 *
 * Description: Compute least significant bits of polynomial
 *
 * Arguments:   - poly *a0: pointer to output polynomial
 *              - const poly *a: pointer to input polynomial
 **************************************************/
 static const union {
  __m256i vec;
  int32_t arr[8];
 } lsb_data_avx = {.arr = {
  1,1,1,1,1,1,1,1
 }};
void poly_lsb(poly *a0, const poly *a) {
    unsigned int i;

    for (i = 0; i < N/8; ++i)
    {
      a0->vec[i] = _mm256_and_si256(a->vec[i], lsb_data_avx.vec);
    }
}

/*************************************************
 * Name:        poly_uniform
 *
 * Description: Sample polynomial with uniformly random coefficients
 *              in [0,Q-1] by performing rejection sampling on the
 *              output stream of SHAKE256(seed|nonce)
 *
 * Arguments:   - poly *a: pointer to output polynomial
 *              - const uint8_t seed[]: byte array with seed of length SEEDBYTES
 *              - uint16_t nonce: 2-byte nonce
 **************************************************/
#define POLY_UNIFORM_NBLOCKS                                                   \
    ((512 + STREAM128_BLOCKBYTES - 1) / STREAM128_BLOCKBYTES)
// N * 2(random bytes for [0, Q - 1])

void poly_uniform(poly *a, const uint8_t seed[SEEDBYTES], uint16_t nonce) {
    unsigned int i, ctr, off;
    unsigned int buflen = POLY_UNIFORM_NBLOCKS * STREAM128_BLOCKBYTES;
    uint8_t buf[POLY_UNIFORM_NBLOCKS * STREAM128_BLOCKBYTES + 1];
    stream128_state state;

    stream128_init(&state, seed, nonce);
    stream128_squeezeblocks(buf, POLY_UNIFORM_NBLOCKS, &state);

    ctr = rej_uniform(a->coeffs, N, buf, buflen);

    while (ctr < N) {
        off = buflen % 2;
        for (i = 0; i < off; ++i)
            buf[i] = buf[buflen - off + i];

        stream128_squeezeblocks(buf + off, 1, &state);
        buflen = STREAM128_BLOCKBYTES + off;
        ctr += rej_uniform(a->coeffs + ctr, N - ctr, buf, buflen);
    }
}

#ifndef HAETAE_USE_AES
#define REJ_UNIFORM_NBLOCKS                                                    \
    ((512 + STREAM128_BLOCKBYTES - 1) / STREAM128_BLOCKBYTES)
#define REJ_UNIFORM_BUFLEN (REJ_UNIFORM_NBLOCKS * STREAM128_BLOCKBYTES)
void poly_uniform_4x(poly *a0, poly *a1, poly *a2, poly *a3,
                     const uint8_t seed[32], uint16_t nonce0, uint16_t nonce1,
                     uint16_t nonce2, uint16_t nonce3) {
    unsigned int ctr0, ctr1, ctr2, ctr3;
    ALIGNED_UINT8(REJ_UNIFORM_BUFLEN + 8) buf[4];
    keccakx4_state state;
    __m256i f;

    f = _mm256_loadu_si256((__m256i *)seed);
    _mm256_store_si256(buf[0].vec, f);
    _mm256_store_si256(buf[1].vec, f);
    _mm256_store_si256(buf[2].vec, f);
    _mm256_store_si256(buf[3].vec, f);

    buf[0].coeffs[SEEDBYTES + 0] = nonce0;
    buf[0].coeffs[SEEDBYTES + 1] = nonce0 >> 8;
    buf[1].coeffs[SEEDBYTES + 0] = nonce1;
    buf[1].coeffs[SEEDBYTES + 1] = nonce1 >> 8;
    buf[2].coeffs[SEEDBYTES + 0] = nonce2;
    buf[2].coeffs[SEEDBYTES + 1] = nonce2 >> 8;
    buf[3].coeffs[SEEDBYTES + 0] = nonce3;
    buf[3].coeffs[SEEDBYTES + 1] = nonce3 >> 8;

    shake128x4_absorb_once(&state, buf[0].coeffs, buf[1].coeffs, buf[2].coeffs,
                           buf[3].coeffs, SEEDBYTES + 2);
    ST_AVX2_shake128x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs,
                             buf[3].coeffs, REJ_UNIFORM_NBLOCKS, &state);

    ctr0 = rej_uniform(a0->coeffs, N, buf[0].coeffs, REJ_UNIFORM_BUFLEN);
    ctr1 = rej_uniform(a1->coeffs, N, buf[1].coeffs, REJ_UNIFORM_BUFLEN);
    ctr2 = rej_uniform(a2->coeffs, N, buf[2].coeffs, REJ_UNIFORM_BUFLEN);
    ctr3 = rej_uniform(a3->coeffs, N, buf[3].coeffs, REJ_UNIFORM_BUFLEN);

    while (ctr0 < N || ctr1 < N || ctr2 < N || ctr3 < N) {
        ST_AVX2_shake128x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs,
                                 buf[3].coeffs, 1, &state);

        ctr0 += rej_uniform(a0->coeffs + ctr0, N - ctr0, buf[0].coeffs,
                            SHAKE128_RATE);
        ctr1 += rej_uniform(a1->coeffs + ctr1, N - ctr1, buf[1].coeffs,
                            SHAKE128_RATE);
        ctr2 += rej_uniform(a2->coeffs + ctr2, N - ctr2, buf[2].coeffs,
                            SHAKE128_RATE);
        ctr3 += rej_uniform(a3->coeffs + ctr3, N - ctr3, buf[3].coeffs,
                            SHAKE128_RATE);
    }
}

void poly_uniform_4x_state(poly *a0, poly *a1, poly *a2, poly *a3,
                     const keccakx4_state32 *pre_state, uint16_t nonce0, uint16_t nonce1,
                     uint16_t nonce2, uint16_t nonce3) {
    unsigned int ctr0, ctr1, ctr2, ctr3;
    ALIGNED_UINT8(REJ_UNIFORM_BUFLEN + 8) buf[4];
    keccakx4_state state;
    __m256i f;

    init_state_32(&state, pre_state, nonce0,nonce1,nonce2,nonce3);
    ST_AVX2_shake128x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs,
                             buf[3].coeffs, REJ_UNIFORM_NBLOCKS, &state);

    ctr0 = rej_uniform(a0->coeffs, N, buf[0].coeffs, REJ_UNIFORM_BUFLEN);
    ctr1 = rej_uniform(a1->coeffs, N, buf[1].coeffs, REJ_UNIFORM_BUFLEN);
    ctr2 = rej_uniform(a2->coeffs, N, buf[2].coeffs, REJ_UNIFORM_BUFLEN);
    ctr3 = rej_uniform(a3->coeffs, N, buf[3].coeffs, REJ_UNIFORM_BUFLEN);

    while (ctr0 < N || ctr1 < N || ctr2 < N || ctr3 < N) {
        ST_AVX2_shake128x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs,
                                 buf[3].coeffs, 1, &state);

        ctr0 += rej_uniform(a0->coeffs + ctr0, N - ctr0, buf[0].coeffs,
                            SHAKE128_RATE);
        ctr1 += rej_uniform(a1->coeffs + ctr1, N - ctr1, buf[1].coeffs,
                            SHAKE128_RATE);
        ctr2 += rej_uniform(a2->coeffs + ctr2, N - ctr2, buf[2].coeffs,
                            SHAKE128_RATE);
        ctr3 += rej_uniform(a3->coeffs + ctr3, N - ctr3, buf[3].coeffs,
                            SHAKE128_RATE);
    }
}

static uint64_t load64(const uint8_t *x) {
    uint64_t r;
    memcpy(&r, x, sizeof(uint64_t));
    return r;
}

static void refresh_states_x2(keccakx4_state *state, const keccakx4_state64 *S_state, uint16_t nonce) {
    const __m256i mask = _mm256_set_epi64x(0, 0, UINT64_MAX, UINT64_MAX);
    const __m256i f9 = _mm256_set_epi64x(0x1ULL << 63, 0x1ULL << 63, 0, 0);
    __m256i f;

    state->s[0] = _mm256_permute2x128_si256(state->s[0], S_state->s[0], 0x20);
    state->s[1] = _mm256_permute2x128_si256(state->s[1], S_state->s[1], 0x20);
    state->s[2] = _mm256_permute2x128_si256(state->s[2], S_state->s[2], 0x20);
    state->s[3] = _mm256_permute2x128_si256(state->s[3], S_state->s[3], 0x20);
    state->s[4] = _mm256_permute2x128_si256(state->s[4], S_state->s[4], 0x20);
    state->s[5] = _mm256_permute2x128_si256(state->s[5], S_state->s[5], 0x20);
    state->s[6] = _mm256_permute2x128_si256(state->s[6], S_state->s[6], 0x20);
    state->s[7] = _mm256_permute2x128_si256(state->s[7], S_state->s[7], 0x20);
    f = _mm256_set_epi64x((0x1f << 16) ^ (nonce + 1), (0x1f << 16) ^ nonce, (0x1f << 16) ^ (nonce + 1), (0x1f << 16) ^ nonce);
    state->s[8] = _mm256_permute2x128_si256(state->s[8], f, 0x20);

    for (int i = 9; i < 25; ++i) {
        state->s[i] = _mm256_and_si256(state->s[i], mask);
    }
    state->s[16] = _mm256_or_si256(state->s[16], f9);
}

int hybrid_squeezeblocks(uint8_t *out0, uint8_t *out1, loop_queue_kg *loop, const keccakx4_state64 *S_state,
                         uint16_t Snonce, int nblocks, keccakx4_state *state) {
    __m256i t0, t1, t2, t3, t4, t5, t6, t7;
    __m256i f0, f1, f2, f3, f4, f5, f6, f7;
    __m128i t;
    uint8_t *out2, *out3;

    for (int i = 0; i < nblocks; ++i) {
        Snonce += 2;
        out2 = loop_kg_next(loop);
        out3 = loop_kg_next(loop);
        f1600x4(state->s, KeccakF_RoundConstants);
        // print_loop_kg_state(loop);

        t0 = _mm256_unpacklo_epi64(state->s[0], state->s[1]);
        t1 = _mm256_unpackhi_epi64(state->s[0], state->s[1]);
        t2 = _mm256_unpacklo_epi64(state->s[2], state->s[3]);
        t3 = _mm256_unpackhi_epi64(state->s[2], state->s[3]);

        t4 = _mm256_unpacklo_epi64(state->s[4], state->s[5]);
        t5 = _mm256_unpackhi_epi64(state->s[4], state->s[5]);
        t6 = _mm256_unpacklo_epi64(state->s[6], state->s[7]);
        t7 = _mm256_unpackhi_epi64(state->s[6], state->s[7]);

        f0 = _mm256_permute2x128_si256(t0, t2, 0x20);
        f1 = _mm256_permute2x128_si256(t1, t3, 0x20);
        f2 = _mm256_permute2x128_si256(t0, t2, 0x31);
        f3 = _mm256_permute2x128_si256(t1, t3, 0x31);

        f4 = _mm256_permute2x128_si256(t4, t6, 0x20);
        f5 = _mm256_permute2x128_si256(t5, t7, 0x20);
        f6 = _mm256_permute2x128_si256(t4, t6, 0x31);
        f7 = _mm256_permute2x128_si256(t5, t7, 0x31);


        _mm256_storeu_si256((__m256i *) out0, f0);
        _mm256_storeu_si256((__m256i *) out1, f1);
        _mm256_storeu_si256((__m256i *) out2, f2);
        _mm256_storeu_si256((__m256i *) out3, f3);
        _mm256_storeu_si256((__m256i *) (out0 + 32), f4);
        _mm256_storeu_si256((__m256i *) (out1 + 32), f5);
        _mm256_storeu_si256((__m256i *) (out2 + 32), f6);
        _mm256_storeu_si256((__m256i *) (out3 + 32), f7);

        out0 += 64;
        out1 += 64;
        out2 += 64;
        out3 += 64;

        t0 = _mm256_unpacklo_epi64(state->s[8], state->s[9]);
        t1 = _mm256_unpackhi_epi64(state->s[8], state->s[9]);
        t2 = _mm256_unpacklo_epi64(state->s[10], state->s[11]);
        t3 = _mm256_unpackhi_epi64(state->s[10], state->s[11]);

        t4 = _mm256_unpacklo_epi64(state->s[12], state->s[13]);
        t5 = _mm256_unpackhi_epi64(state->s[12], state->s[13]);
        t6 = _mm256_unpacklo_epi64(state->s[14], state->s[15]);
        t7 = _mm256_unpackhi_epi64(state->s[14], state->s[15]);

        f0 = _mm256_permute2x128_si256(t0, t2, 0x20);
        f1 = _mm256_permute2x128_si256(t1, t3, 0x20);
        f2 = _mm256_permute2x128_si256(t0, t2, 0x31);
        f3 = _mm256_permute2x128_si256(t1, t3, 0x31);

        f4 = _mm256_permute2x128_si256(t4, t6, 0x20);
        f5 = _mm256_permute2x128_si256(t5, t7, 0x20);
        f6 = _mm256_permute2x128_si256(t4, t6, 0x31);
        f7 = _mm256_permute2x128_si256(t5, t7, 0x31);

        _mm256_storeu_si256((__m256i *) out0, f0);
        _mm256_storeu_si256((__m256i *) out1, f1);
        _mm256_storeu_si256((__m256i *) out2, f2);
        _mm256_storeu_si256((__m256i *) out3, f3);
        _mm256_storeu_si256((__m256i *) (out0 + 32), f4);
        _mm256_storeu_si256((__m256i *) (out1 + 32), f5);
        _mm256_storeu_si256((__m256i *) (out2 + 32), f6);
        _mm256_storeu_si256((__m256i *) (out3 + 32), f7);

        out0 += 64;
        out1 += 64;
        out2 += 64;
        out3 += 64;

        t0 = _mm256_unpacklo_epi64(state->s[16], state->s[17]);
        t1 = _mm256_unpackhi_epi64(state->s[16], state->s[17]);
        t2 = _mm256_unpacklo_epi64(state->s[18], state->s[19]);
        t3 = _mm256_unpackhi_epi64(state->s[18], state->s[19]);

        f0 = _mm256_permute2x128_si256(t0, t2, 0x20);
        f1 = _mm256_permute2x128_si256(t1, t3, 0x20);
        f2 = _mm256_permute2x128_si256(t0, t2, 0x31);
        f3 = _mm256_permute2x128_si256(t1, t3, 0x31);

        _mm256_storeu_si256((__m256i *) out0, f0);
        _mm256_storeu_si256((__m256i *) out1, f1);
        _mm256_storeu_si256((__m256i *) out2, f2);
        _mm256_storeu_si256((__m256i *) out3, f3);

        out0 += 32;
        out1 += 32;
        out2 += 32;
        out3 += 32;

        t = _mm256_castsi256_si128(state->s[20]);
        _mm_storeu_si64(out0, t);
        _mm_storeu_si64(out1, _mm_bsrli_si128(t, 8));
        t = _mm256_extracti128_si256(state->s[20], 1);
        _mm_storeu_si64(out2, t);
        _mm_storeu_si64(out3, _mm_bsrli_si128(t, 8));

        out0 += 8;
        out1 += 8;
        out2 += 8;
        out3 += 8;

        //////////////
        t0 = _mm256_unpacklo_epi64(state->s[17], state->s[18]);
        t1 = _mm256_unpackhi_epi64(state->s[17], state->s[18]);
        t2 = _mm256_unpacklo_epi64(state->s[19], state->s[20]);
        t3 = _mm256_unpackhi_epi64(state->s[19], state->s[20]);

        t4 = _mm256_unpacklo_epi64(state->s[21], state->s[22]);
        t5 = _mm256_unpackhi_epi64(state->s[21], state->s[22]);
        t6 = _mm256_unpacklo_epi64(state->s[23], state->s[24]);
        t7 = _mm256_unpackhi_epi64(state->s[23], state->s[24]);

        f2 = _mm256_permute2x128_si256(t0, t2, 0x31);
        f3 = _mm256_permute2x128_si256(t1, t3, 0x31);

        f6 = _mm256_permute2x128_si256(t4, t6, 0x31);
        f7 = _mm256_permute2x128_si256(t5, t7, 0x31);

        _mm256_storeu_si256((__m256i *) out2, f2);
        _mm256_storeu_si256((__m256i *) out3, f3);
        _mm256_storeu_si256((__m256i *) (out2 + 32), f6);
        _mm256_storeu_si256((__m256i *) (out3 + 32), f7);

        refresh_states_x2(state, S_state, Snonce);

    }
    return Snonce;
}

int hybrid_poly_uniform_2x_eta_2x(poly *a0, poly *a1, loop_queue_kg *loop,
                     const keccakx4_state32 *A_state, uint16_t Anonce0, uint16_t Anonce1,
                     const keccakx4_state64 *S_state, uint16_t Snonce) {
    unsigned int ctr0, ctr1;
    ALIGNED_UINT8(REJ_UNIFORM_BUFLEN + 8) buf[4];
    keccakx4_state state;

    init_state_32(&state, A_state, Anonce0, Anonce1,0,0);
    refresh_states_x2(&state,S_state,Snonce);

    Snonce = hybrid_squeezeblocks(buf[0].coeffs, buf[1].coeffs, loop, S_state, Snonce,
                         REJ_UNIFORM_NBLOCKS, &state);

    ctr0 = rej_uniform(a0->coeffs, N, buf[0].coeffs, REJ_UNIFORM_BUFLEN);
    ctr1 = rej_uniform(a1->coeffs, N, buf[1].coeffs, REJ_UNIFORM_BUFLEN);

    while (ctr0 < N || ctr1 < N ) {
        shake128x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs,
                                 buf[3].coeffs, 1, &state);

        ctr0 += rej_uniform(a0->coeffs + ctr0, N - ctr0, buf[0].coeffs,
                            SHAKE128_RATE);
        ctr1 += rej_uniform(a1->coeffs + ctr1, N - ctr1, buf[1].coeffs,
                            SHAKE128_RATE);
    }
    return Snonce;
}
#endif

/*************************************************
 * Name:        poly_uniform_eta
 *
 * Description: Sample polynomial with uniformly random coefficients
 *              in [-ETA,ETA] by performing rejection sampling on the
 *              output stream from SHAKE256(seed|nonce)
 *
 * Arguments:   - poly *a: pointer to output polynomial
 *              - const uint8_t seed[]: byte array with seed of length CRHBYTES
 *              - uint16_t nonce: 2-byte nonce
 **************************************************/
#if ETA == 1
#define POLY_UNIFORM_ETA_NBLOCKS 1
#endif

void poly_uniform_eta(poly *a, const uint8_t seed[CRHBYTES], uint16_t nonce) {
    unsigned int ctr;
    unsigned int buflen = POLY_UNIFORM_ETA_NBLOCKS * STREAM256_BLOCKBYTES;
    uint8_t buf[POLY_UNIFORM_ETA_NBLOCKS * STREAM256_BLOCKBYTES];
    stream256_state state;

    stream256_init(&state, seed, nonce);
    stream256_squeezeblocks(buf, POLY_UNIFORM_ETA_NBLOCKS, &state);

    ctr = rej_eta(a->coeffs, N, buf, buflen);

    while (ctr < N) {
        stream256_squeezeblocks(buf, 1, &state);
        ctr += rej_eta(a->coeffs + ctr, N - ctr, buf, STREAM256_BLOCKBYTES);
    }
}


void poly_uniform_eta_4x(poly *a0, poly *a1, poly *a2, poly *a3,
                         const uint8_t seed[CRHBYTES], uint16_t nonce0,
                         uint16_t nonce1, uint16_t nonce2, uint16_t nonce3) {
    unsigned int ctr0, ctr1, ctr2, ctr3;
    ALIGNED_UINT8(POLY_UNIFORM_ETA_NBLOCKS * STREAM256_BLOCKBYTES + 8) buf[4];
    keccakx4_state state;
    __m256i f;

    f = _mm256_loadu_si256((__m256i *)seed);
    _mm256_store_si256(buf[0].vec, f);
    _mm256_store_si256(buf[1].vec, f);
    _mm256_store_si256(buf[2].vec, f);
    _mm256_store_si256(buf[3].vec, f);
    f = _mm256_loadu_si256((__m256i *)&seed[32]);
    _mm256_store_si256(&buf[0].vec[1], f);
    _mm256_store_si256(&buf[1].vec[1], f);
    _mm256_store_si256(&buf[2].vec[1], f);
    _mm256_store_si256(&buf[3].vec[1], f);

    buf[0].coeffs[CRHBYTES + 0] = nonce0;
    buf[0].coeffs[CRHBYTES + 1] = nonce0 >> 8;
    buf[1].coeffs[CRHBYTES + 0] = nonce1;
    buf[1].coeffs[CRHBYTES + 1] = nonce1 >> 8;
    buf[2].coeffs[CRHBYTES + 0] = nonce2;
    buf[2].coeffs[CRHBYTES + 1] = nonce2 >> 8;
    buf[3].coeffs[CRHBYTES + 0] = nonce3;
    buf[3].coeffs[CRHBYTES + 1] = nonce3 >> 8;

    shake256x4_absorb_once(&state, buf[0].coeffs, buf[1].coeffs, buf[2].coeffs,
                           buf[3].coeffs, CRHBYTES + 2);
    shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs,
                             buf[3].coeffs, POLY_UNIFORM_ETA_NBLOCKS, &state);

    ctr0 = rej_eta(a0->coeffs, N, buf[0].coeffs,
                   POLY_UNIFORM_ETA_NBLOCKS * SHAKE256_RATE);
    ctr1 = rej_eta(a1->coeffs, N, buf[1].coeffs,
                   POLY_UNIFORM_ETA_NBLOCKS * SHAKE256_RATE);
    ctr2 = rej_eta(a2->coeffs, N, buf[2].coeffs,
                   POLY_UNIFORM_ETA_NBLOCKS * SHAKE256_RATE);
    ctr3 = rej_eta(a3->coeffs, N, buf[3].coeffs,
                   POLY_UNIFORM_ETA_NBLOCKS * SHAKE256_RATE);

    while (ctr0 < N || ctr1 < N || ctr2 < N || ctr3 < N) {
        shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs,
                                 buf[3].coeffs, 1, &state);

        ctr0 +=
            rej_eta(a0->coeffs + ctr0, N - ctr0, buf[0].coeffs, SHAKE256_RATE);
        ctr1 +=
            rej_eta(a1->coeffs + ctr1, N - ctr1, buf[1].coeffs, SHAKE256_RATE);
        ctr2 +=
            rej_eta(a2->coeffs + ctr2, N - ctr2, buf[2].coeffs, SHAKE256_RATE);
        ctr3 +=
            rej_eta(a3->coeffs + ctr3, N - ctr3, buf[3].coeffs, SHAKE256_RATE);
    }
}

void poly_rand_eta_4x(loop_queue_kg *loop, const keccakx4_state64 *pre_state, uint16_t nonce0, uint16_t nonce1,
                           uint16_t nonce2, uint16_t nonce3) {
    uint8_t *s_buff1, *s_buff2, *s_buff3, *s_buff4;
    const int offet = STREAM256_BLOCKBYTES;
    keccakx4_state state;

    init_state_64(&state, pre_state, nonce0,nonce1,nonce2,nonce3);

    s_buff1 = loop_kg_next(loop);
    s_buff2 = loop_kg_next(loop);
    s_buff3 = loop_kg_next(loop);
    s_buff4 = loop_kg_next(loop);

    ST_AVX2_shake256x4_squeezeblocks(s_buff1, s_buff2, s_buff3, s_buff4, POLY_UNIFORM_ETA_NBLOCKS, &state);
    store_statex4(s_buff1 + offet, s_buff2 + offet, s_buff3 + offet, s_buff4 + offet, &state);
}

void poly_rej_eta(poly *a, loop_queue_kg *loop) {
    unsigned int ctr;
    uint8_t *s_buff0;
    s_buff0 = loop_kg_dequeue(loop);

    ctr = rej_eta_avx2(a->coeffs, N, s_buff0, POLY_UNIFORM_ETA_NBLOCKS * SHAKE256_RATE);

    while (ctr < N) {
        keccakx4_state state;
        ALIGNED_UINT8(POLY_UNIFORM_ETA_NBLOCKS * STREAM256_BLOCKBYTES + 8) buf[4];
        recover_statex4(s_buff0,buf[1].coeffs, buf[2].coeffs,buf[3].coeffs, &state);
        ST_AVX2_shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs,
                                 buf[3].coeffs, 1, &state);
        ctr += rej_eta(a->coeffs + ctr, N - ctr, buf[0].coeffs, STREAM256_BLOCKBYTES);
    }
}

void poly_rej_eta_4x(poly *a0, poly *a1, poly *a2, poly *a3,
                         loop_queue_kg *loop) {
    unsigned int ctr0, ctr1, ctr2, ctr3;
    uint8_t *s_buff0,*s_buff1, *s_buff2, *s_buff3;
    s_buff0 = loop_kg_dequeue(loop);
    s_buff1 = loop_kg_dequeue(loop);
    s_buff2 = loop_kg_dequeue(loop);
    s_buff3 = loop_kg_dequeue(loop);

    ctr0 = rej_eta_avx2(a0->coeffs, N, s_buff0,
                   POLY_UNIFORM_ETA_NBLOCKS * SHAKE256_RATE);
    ctr1 = rej_eta_avx2(a1->coeffs, N, s_buff1,
                   POLY_UNIFORM_ETA_NBLOCKS * SHAKE256_RATE);
    ctr2 = rej_eta_avx2(a2->coeffs, N, s_buff2,
                   POLY_UNIFORM_ETA_NBLOCKS * SHAKE256_RATE);
    ctr3 = rej_eta_avx2(a3->coeffs, N, s_buff3,
                   POLY_UNIFORM_ETA_NBLOCKS * SHAKE256_RATE);

    while (ctr0 < N || ctr1 < N || ctr2 < N || ctr3 < N) {
        keccakx4_state state;
        ALIGNED_UINT8(POLY_UNIFORM_ETA_NBLOCKS * STREAM256_BLOCKBYTES + 8) buf[4];
        recover_statex4(s_buff0,s_buff1, s_buff2, s_buff3, &state);
        ST_AVX2_shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs,
                                 buf[3].coeffs, 1, &state);

        ctr0 +=
            rej_eta(a0->coeffs + ctr0, N - ctr0, buf[0].coeffs, SHAKE256_RATE);
        ctr1 +=
            rej_eta(a1->coeffs + ctr1, N - ctr1, buf[1].coeffs, SHAKE256_RATE);
        ctr2 +=
            rej_eta(a2->coeffs + ctr2, N - ctr2, buf[2].coeffs, SHAKE256_RATE);
        ctr3 +=
            rej_eta(a3->coeffs + ctr3, N - ctr3, buf[3].coeffs, SHAKE256_RATE);
    }
}

void poly_rej_eta_2x(poly *a0, poly *a1, loop_queue_kg *loop) {
    unsigned int ctr0, ctr1;
    uint8_t *s_buff0,*s_buff1;
    s_buff0 = loop_kg_dequeue(loop);
    s_buff1 = loop_kg_dequeue(loop);


    ctr0 = rej_eta_avx2(a0->coeffs, N, s_buff0,
                   POLY_UNIFORM_ETA_NBLOCKS * SHAKE256_RATE);
    ctr1 = rej_eta_avx2(a1->coeffs, N, s_buff1,
                   POLY_UNIFORM_ETA_NBLOCKS * SHAKE256_RATE);

    while (ctr0 < N || ctr1 < N ) {
        keccakx4_state state;
        ALIGNED_UINT8(POLY_UNIFORM_ETA_NBLOCKS * STREAM256_BLOCKBYTES + 8) buf[4];
        recover_statex4(s_buff0,s_buff1, buf[2].coeffs, buf[3].coeffs, &state);
        ST_AVX2_shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs,
                                 buf[3].coeffs, 1, &state);

        ctr0 +=
            rej_eta(a0->coeffs + ctr0, N - ctr0, buf[0].coeffs, SHAKE256_RATE);
        ctr1 +=
            rej_eta(a1->coeffs + ctr1, N - ctr1, buf[1].coeffs, SHAKE256_RATE);
    }
}

uint8_t hammingWeight_8(uint8_t x) {
    x = (x & 0x55) + (x >> 1 & 0x55);
    x = (x & 0x33) + (x >> 2 & 0x33);
    x = (x & 0x0F) + (x >> 4 & 0x0F);

    return x;
}

/*************************************************
 * Name:        poly_challenge
 *
 * Description: Implementation of challenge. Samples polynomial with TAU 1
 *              coefficients using the output stream of SHAKE256(seed).
 *
 * Arguments:   - poly *c: pointer to output polynomial
 *              - const uint8_t highbits_lsb[]: packed highbits and lsb
 *              - const uint8_t mu[]: hash of pk and message
 **************************************************/
void poly_challenge(poly *c, const uint8_t highbits_lsb[POLYVECK_HIGHBITS_PACKEDBYTES + POLYC_PACKEDBYTES], const uint8_t mu[SEEDBYTES]) {
#if (HAETAE_MODE == 2) || (HAETAE_MODE == 3)
    unsigned int i, b, pos = 0;
    uint8_t buf[XOF256_BLOCKBYTES];
    xof256_state state;

    // H(HighBits(A * y mod 2q), LSB(round(y0) * j), M)
    xof256_absorbe_twice(&state, highbits_lsb,
                         POLYVECK_HIGHBITS_PACKEDBYTES + POLYC_PACKEDBYTES, mu,
                         SEEDBYTES);
    xof256_squeezeblocks(buf, 1, &state);

    for (i = 0; i < N; ++i)
        c->coeffs[i] = 0;
    for (i = N - TAU; i < N; ++i) {
        do {
            if (pos >= XOF256_BLOCKBYTES) {
                xof256_squeezeblocks(buf, 1, &state);
                pos = 0;
            }

            b = buf[pos++];
        } while (b > i);

        c->coeffs[i] = c->coeffs[b];
        c->coeffs[b] = 1;
    }
#elif HAETAE_MODE == 5
    unsigned int i, hwt = 0, cond = 0;
    uint8_t mask = 0, w0 = 0;
    uint8_t buf[32] = {0};
    xof256_state state;

    // H(HighBits(A * y mod 2q), LSB(round(y0) * j), M)
    xof256_absorbe_twice(&state, highbits_lsb,
                         POLYVECK_HIGHBITS_PACKEDBYTES + POLYC_PACKEDBYTES, mu,
                         SEEDBYTES);
    xof256_squeeze(buf, 32, &state);

    for (i = 0; i < 32; ++i)
        hwt += hammingWeight_8(buf[i]);

    cond = (128 - hwt);
    mask = 0xff & (cond >> 8);
    w0 = -(buf[0] & 1);
    mask = w0 ^ ((-(!!cond & 1)) & (mask ^ w0)); // mask = !!cond ? mask : w0
    for (i = 0; i < 32; ++i) {
        buf[i] ^= mask;
        c->coeffs[8 * i] = buf[i] & 1;
        c->coeffs[8 * i + 1] = (buf[i] >> 1) & 1;
        c->coeffs[8 * i + 2] = (buf[i] >> 2) & 1;
        c->coeffs[8 * i + 3] = (buf[i] >> 3) & 1;
        c->coeffs[8 * i + 4] = (buf[i] >> 4) & 1;
        c->coeffs[8 * i + 5] = (buf[i] >> 5) & 1;
        c->coeffs[8 * i + 6] = (buf[i] >> 6) & 1;
        c->coeffs[8 * i + 7] = (buf[i] >> 7) & 1;
    }
#endif
}

void poly_decomposed_pack(uint8_t *buf, const poly *a) {
    unsigned int i;
    for (i = 0; i < N; i++) {
        buf[i] = a->coeffs[i];
    }
}

void poly_decomposed_unpack(poly *a, const uint8_t *buf) {
    unsigned int i;
    for (i = 0; i < N; i++) {
        a->coeffs[i] = (int8_t)buf[i];
    }
}

void poly_pack_highbits(uint8_t *buf, const poly *a) {
    unsigned int i;
    for (i = 0; i < N / 8; i++) {
        buf[9 * i + 0] = a->coeffs[8 * i + 0] & 0xff;

        buf[9 * i + 1] = (a->coeffs[8 * i + 0] >> 8) & 0x01;
        buf[9 * i + 1] |= (a->coeffs[8 * i + 1] << 1) & 0xff;

        buf[9 * i + 2] = (a->coeffs[8 * i + 1] >> 7) & 0x03;
        buf[9 * i + 2] |= (a->coeffs[8 * i + 2] << 2) & 0xff;

        buf[9 * i + 3] = (a->coeffs[8 * i + 2] >> 6) & 0x07;
        buf[9 * i + 3] |= (a->coeffs[8 * i + 3] << 3) & 0xff;

        buf[9 * i + 4] = (a->coeffs[8 * i + 3] >> 5) & 0x0f;
        buf[9 * i + 4] |= (a->coeffs[8 * i + 4] << 4) & 0xff;

        buf[9 * i + 5] = (a->coeffs[8 * i + 4] >> 4) & 0x1f;
        buf[9 * i + 5] |= (a->coeffs[8 * i + 5] << 5) & 0xff;

        buf[9 * i + 6] = (a->coeffs[8 * i + 5] >> 3) & 0x3f;
        buf[9 * i + 6] |= (a->coeffs[8 * i + 6] << 6) & 0xff;

        buf[9 * i + 7] = (a->coeffs[8 * i + 6] >> 2) & 0x7f;
        buf[9 * i + 7] |= (a->coeffs[8 * i + 7] << 7) & 0xff;

        buf[9 * i + 8] = (a->coeffs[8 * i + 7] >> 1) & 0xff;
    }
}

void poly_pack_lsb(uint8_t *buf, const poly *a) {
    unsigned int i;
    for (i = 0; i < N; i++) {
        if ((i % 8) == 0) {
            buf[i / 8] = 0;
        }
        buf[i / 8] |= (a->coeffs[i] & 1) << (i % 8);
    }
}

/*************************************************
 * Name:        polyq_pack
 *
 * Description: Bit-pack polynomial with coefficients in [0, Q - 1].
 *
 * Arguments:   - uint8_t *r: pointer to output byte array with at least
 *                            POLYQ_PACKEDBYTES bytes
 *              - const poly *a: pointer to input polynomial
 **************************************************/
void polyq_pack(uint8_t *r, const poly *a) {
    unsigned int i;
#if D == 1
    int b_idx = 0, d_idx = 0;

    for (i = 0; i < (N >> 3); ++i) {
        b_idx = 15 * i;
        d_idx = 8 * i;

        r[b_idx] = (a->coeffs[d_idx] & 0xff);
        r[b_idx + 1] = ((a->coeffs[d_idx] >> 8) & 0x7f) |
                       ((a->coeffs[d_idx + 1] & 0x1) << 7);
        r[b_idx + 2] = ((a->coeffs[d_idx + 1] >> 1) & 0xff);
        r[b_idx + 3] = ((a->coeffs[d_idx + 1] >> 9) & 0x3f) |
                       ((a->coeffs[d_idx + 2] & 0x3) << 6);
        r[b_idx + 4] = ((a->coeffs[d_idx + 2] >> 2) & 0xff);
        r[b_idx + 5] = ((a->coeffs[d_idx + 2] >> 10) & 0x1f) |
                       ((a->coeffs[d_idx + 3] & 0x7) << 5);
        r[b_idx + 6] = ((a->coeffs[d_idx + 3] >> 3) & 0xff);
        r[b_idx + 7] = ((a->coeffs[d_idx + 3] >> 11) & 0xf) |
                       ((a->coeffs[d_idx + 4] & 0xf) << 4);
        r[b_idx + 8] = ((a->coeffs[d_idx + 4] >> 4) & 0xff);
        r[b_idx + 9] = ((a->coeffs[d_idx + 4] >> 12) & 0x7) |
                       ((a->coeffs[d_idx + 5] & 0x1f) << 3);
        r[b_idx + 10] = ((a->coeffs[d_idx + 5] >> 5) & 0xff);
        r[b_idx + 11] = ((a->coeffs[d_idx + 5] >> 13) & 0x3) |
                        ((a->coeffs[d_idx + 6] & 0x3f) << 2);
        r[b_idx + 12] = ((a->coeffs[d_idx + 6] >> 6) & 0xff);
        r[b_idx + 13] = ((a->coeffs[d_idx + 6] >> 14) & 0x1) |
                        (a->coeffs[d_idx + 7] & 0x7f) << 1;
        r[b_idx + 14] = ((a->coeffs[d_idx + 7] >> 7) & 0xff);
    }
#else
    for (i = 0; i < N / 1; ++i) {
        r[2 * i + 0] = a->coeffs[1 * i + 0] >> 0;
        r[2 * i + 1] = a->coeffs[1 * i + 0] >> 8;
    }
#endif
}

/*************************************************
 * Name:        polyq_unpack
 *
 * Description: Unpack polynomial with coefficients in [0, Q - 1].
 *
 * Arguments:   - poly *r: pointer to output polynomial
 *              - const uint8_t *a: byte array with bit-packed polynomial
 **************************************************/
void polyq_unpack(poly *r, const uint8_t *a) {
    unsigned int i;
#if D == 1
    int b_idx = 0, d_idx = 0;

    for (i = 0; i < (N >> 3); ++i) {
        b_idx = 15 * i;
        d_idx = 8 * i;

        r->coeffs[d_idx] = (a[b_idx] & 0xff) | ((a[b_idx + 1] & 0x7f) << 8);
        r->coeffs[d_idx + 1] = ((a[b_idx + 1] >> 7) & 0x1) |
                               ((a[b_idx + 2] & 0xff) << 1) |
                               ((a[b_idx + 3] & 0x3f) << 9);
        r->coeffs[d_idx + 2] = ((a[b_idx + 3] >> 6) & 0x3) |
                               ((a[b_idx + 4] & 0xff) << 2) |
                               ((a[b_idx + 5] & 0x1f) << 10);
        r->coeffs[d_idx + 3] = ((a[b_idx + 5] >> 5) & 0x7) |
                               ((a[b_idx + 6] & 0xff) << 3) |
                               ((a[b_idx + 7] & 0xf) << 11);
        r->coeffs[d_idx + 4] = ((a[b_idx + 7] >> 4) & 0xf) |
                               ((a[b_idx + 8] & 0xff) << 4) |
                               ((a[b_idx + 9] & 0x7) << 12);
        r->coeffs[d_idx + 5] = ((a[b_idx + 9] >> 3) & 0x1f) |
                               ((a[b_idx + 10] & 0xff) << 5) |
                               ((a[b_idx + 11] & 0x3) << 13);
        r->coeffs[d_idx + 6] = ((a[b_idx + 11] >> 2) & 0x3f) |
                               ((a[b_idx + 12] & 0xff) << 6) |
                               ((a[b_idx + 13] & 0x1) << 14);
        r->coeffs[d_idx + 7] =
            ((a[b_idx + 13] >> 1) & 0x7f) | ((a[b_idx + 14] & 0xff) << 7);
    }

#else
    for (i = 0; i < N / 1; ++i) {
        r->coeffs[1 * i + 0] = a[2 * i + 0] >> 0;
        r->coeffs[1 * i + 0] |= (uint16_t)a[2 * i + 1] << 8;
        r->coeffs[1 * i + 0] &= 0xffff;
    }
#endif
}

/*************************************************
 * Name:        polyeta_pack
 *
 * Description: Bit-pack polynomial with coefficients in [-ETA,ETA].
 *
 * Arguments:   - uint8_t *r: pointer to output byte array with at least
 *                            POLYETA_PACKEDBYTES bytes
 *              - const poly *a: pointer to input polynomial
 **************************************************/
void polyeta_pack(uint8_t *r, const poly *a) {
    unsigned int i;
    uint8_t t[8];

#if ETA == 1
    for (i = 0; i < N / 4; ++i) {
        t[0] = ETA - a->coeffs[4 * i + 0];
        t[1] = ETA - a->coeffs[4 * i + 1];
        t[2] = ETA - a->coeffs[4 * i + 2];
        t[3] = ETA - a->coeffs[4 * i + 3];
        r[i] = t[0] >> 0;
        r[i] |= t[1] << 2;
        r[i] |= t[2] << 4;
        r[i] |= t[3] << 6;
    }
#elif ETA == 2
    for (i = 0; i < N / 8; ++i) {
        t[0] = ETA - a->coeffs[8 * i + 0];
        t[1] = ETA - a->coeffs[8 * i + 1];
        t[2] = ETA - a->coeffs[8 * i + 2];
        t[3] = ETA - a->coeffs[8 * i + 3];
        t[4] = ETA - a->coeffs[8 * i + 4];
        t[5] = ETA - a->coeffs[8 * i + 5];
        t[6] = ETA - a->coeffs[8 * i + 6];
        t[7] = ETA - a->coeffs[8 * i + 7];

        r[3 * i + 0] = (t[0] >> 0) | (t[1] << 3) | (t[2] << 6);
        r[3 * i + 1] = (t[2] >> 2) | (t[3] << 1) | (t[4] << 4) | (t[5] << 7);
        r[3 * i + 2] = (t[5] >> 1) | (t[6] << 2) | (t[7] << 5);
    }
#endif
}

/*************************************************
 * Name:        polyeta_unpack
 *
 * Description: Unpack polynomial with coefficients in [-ETA,ETA].
 *
 * Arguments:   - poly *r: pointer to output polynomial
 *              - const uint8_t *a: byte array with bit-packed polynomial
 **************************************************/
void polyeta_unpack(poly *r, const uint8_t *a) {
    unsigned int i;

#if ETA == 1
    for (i = 0; i < N / 4; ++i) {
        r->coeffs[4 * i + 0] = a[i] >> 0;
        r->coeffs[4 * i + 0] &= 0x3;

        r->coeffs[4 * i + 1] = a[i] >> 2;
        r->coeffs[4 * i + 1] &= 0x3;

        r->coeffs[4 * i + 2] = a[i] >> 4;
        r->coeffs[4 * i + 2] &= 0x3;

        r->coeffs[4 * i + 3] = a[i] >> 6;
        r->coeffs[4 * i + 3] &= 0x3;

        r->coeffs[4 * i + 0] = ETA - r->coeffs[4 * i + 0];
        r->coeffs[4 * i + 1] = ETA - r->coeffs[4 * i + 1];
        r->coeffs[4 * i + 2] = ETA - r->coeffs[4 * i + 2];
        r->coeffs[4 * i + 3] = ETA - r->coeffs[4 * i + 3];
    }

#elif ETA == 2
    for (i = 0; i < N / 8; ++i) {
        r->coeffs[8 * i + 0] = (a[3 * i + 0] >> 0) & 7;
        r->coeffs[8 * i + 1] = (a[3 * i + 0] >> 3) & 7;
        r->coeffs[8 * i + 2] = ((a[3 * i + 0] >> 6) | (a[3 * i + 1] << 2)) & 7;
        r->coeffs[8 * i + 3] = (a[3 * i + 1] >> 1) & 7;
        r->coeffs[8 * i + 4] = (a[3 * i + 1] >> 4) & 7;
        r->coeffs[8 * i + 5] = ((a[3 * i + 1] >> 7) | (a[3 * i + 2] << 1)) & 7;
        r->coeffs[8 * i + 6] = (a[3 * i + 2] >> 2) & 7;
        r->coeffs[8 * i + 7] = (a[3 * i + 2] >> 5) & 7;

        r->coeffs[8 * i + 0] = ETA - r->coeffs[8 * i + 0];
        r->coeffs[8 * i + 1] = ETA - r->coeffs[8 * i + 1];
        r->coeffs[8 * i + 2] = ETA - r->coeffs[8 * i + 2];
        r->coeffs[8 * i + 3] = ETA - r->coeffs[8 * i + 3];
        r->coeffs[8 * i + 4] = ETA - r->coeffs[8 * i + 4];
        r->coeffs[8 * i + 5] = ETA - r->coeffs[8 * i + 5];
        r->coeffs[8 * i + 6] = ETA - r->coeffs[8 * i + 6];
        r->coeffs[8 * i + 7] = ETA - r->coeffs[8 * i + 7];
    }
#endif
}

/*************************************************
 * Name:        poly2eta_pack
 *
 * Description: Bit-pack polynomial with coefficients in [-ETA-1,ETA+1].
 *
 * Arguments:   - uint8_t *r: pointer to output byte array with at least
 *                            POLYETA_PACKEDBYTES bytes
 *              - const poly *a: pointer to input polynomial
 **************************************************/
void poly2eta_pack(uint8_t *r, const poly *a) {
    unsigned int i;
    uint8_t t[8];

#if ETA == 1
    for (i = 0; i < N / 8; ++i) {
        t[0] = 2 * ETA - a->coeffs[8 * i + 0];
        t[1] = 2 * ETA - a->coeffs[8 * i + 1];
        t[2] = 2 * ETA - a->coeffs[8 * i + 2];
        t[3] = 2 * ETA - a->coeffs[8 * i + 3];
        t[4] = 2 * ETA - a->coeffs[8 * i + 4];
        t[5] = 2 * ETA - a->coeffs[8 * i + 5];
        t[6] = 2 * ETA - a->coeffs[8 * i + 6];
        t[7] = 2 * ETA - a->coeffs[8 * i + 7];

        r[3 * i + 0] = (t[0] >> 0) | (t[1] << 3) | (t[2] << 6);
        r[3 * i + 1] = (t[2] >> 2) | (t[3] << 1) | (t[4] << 4) | (t[5] << 7);
        r[3 * i + 2] = (t[5] >> 1) | (t[6] << 2) | (t[7] << 5);
    }
#elif ETA == 2
#error "not yet implemented"
#endif
}

/*************************************************
 * Name:        poly2eta_unpack
 *
 * Description: Unpack polynomial with coefficients in [-ETA-1,ETA+1].
 *
 * Arguments:   - poly *r: pointer to output polynomial
 *              - const uint8_t *a: byte array with bit-packed polynomial
 **************************************************/
void poly2eta_unpack(poly *r, const uint8_t *a) {
    unsigned int i;

#if ETA == 1
    for (i = 0; i < N / 8; ++i) {
        r->coeffs[8 * i + 0] = (a[3 * i + 0] >> 0) & 7;
        r->coeffs[8 * i + 1] = (a[3 * i + 0] >> 3) & 7;
        r->coeffs[8 * i + 2] = ((a[3 * i + 0] >> 6) | (a[3 * i + 1] << 2)) & 7;
        r->coeffs[8 * i + 3] = (a[3 * i + 1] >> 1) & 7;
        r->coeffs[8 * i + 4] = (a[3 * i + 1] >> 4) & 7;
        r->coeffs[8 * i + 5] = ((a[3 * i + 1] >> 7) | (a[3 * i + 2] << 1)) & 7;
        r->coeffs[8 * i + 6] = (a[3 * i + 2] >> 2) & 7;
        r->coeffs[8 * i + 7] = (a[3 * i + 2] >> 5) & 7;

        r->coeffs[8 * i + 0] = 2 * ETA - r->coeffs[8 * i + 0];
        r->coeffs[8 * i + 1] = 2 * ETA - r->coeffs[8 * i + 1];
        r->coeffs[8 * i + 2] = 2 * ETA - r->coeffs[8 * i + 2];
        r->coeffs[8 * i + 3] = 2 * ETA - r->coeffs[8 * i + 3];
        r->coeffs[8 * i + 4] = 2 * ETA - r->coeffs[8 * i + 4];
        r->coeffs[8 * i + 5] = 2 * ETA - r->coeffs[8 * i + 5];
        r->coeffs[8 * i + 6] = 2 * ETA - r->coeffs[8 * i + 6];
        r->coeffs[8 * i + 7] = 2 * ETA - r->coeffs[8 * i + 7];
    }
#elif ETA == 2
#error "not yet implemented"
#endif
}

static const union {
  __m256i vec[3];
  int32_t arr[3*8];
} crt_data_avx = {.arr = {
  Q, Q, Q, Q, Q, Q, Q, Q,
  1,1,1,1,1,1,1,1,
  0,0,0,0,0,0,0,0
}};
void poly_fromcrt(poly *w, const poly *u, const poly *v) {
    unsigned int i;
    __m256i tmp;

    for (i = 0; i < N/8; i++) {
        tmp = _mm256_xor_si256(u->vec[i], v->vec[i]);
        tmp = _mm256_and_si256(tmp, crt_data_avx.vec[1]);
        tmp = _mm256_sub_epi32(crt_data_avx.vec[2], tmp);
        tmp = _mm256_and_si256(tmp, crt_data_avx.vec[0]);
        w->vec[i] = _mm256_add_epi32(u->vec[i], tmp);
    }
}

void poly_fromcrt0(poly *w, const poly *u) {
    unsigned int i;
    __m256i tmp;

    for (i = 0; i < N/8; i++) {
        tmp = _mm256_and_si256(u->vec[i], crt_data_avx.vec[1]);
        tmp = _mm256_sub_epi32(crt_data_avx.vec[2], tmp);
        tmp = _mm256_and_si256(tmp, crt_data_avx.vec[0]);
        w->vec[i] = _mm256_add_epi32(u->vec[i], tmp);
    }
}

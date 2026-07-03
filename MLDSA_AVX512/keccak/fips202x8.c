//
// Created by xurq on 2022/10/19.
//

#include "fips202x8.h"

#include <string.h>
#include <stdio.h>

#include "cpucycles.h"


/**
 *  only works for bits of in_i < shake128_rate
 * @param state
 * @param shake_rate  bytes of shake rate
 * @param in0
 * @param in1
 * @param in2
 * @param in3
 * @param in4
 * @param in5
 * @param in6
 * @param in7
 * @param inlen  bytes of in_i
 */
void shake128x8_absorb(keccakx8_state *state,
                       const uint8_t *in0,
                       const uint8_t *in1,
                       const uint8_t *in2,
                       const uint8_t *in3,
                       const uint8_t *in4,
                       const uint8_t *in5,
                       const uint8_t *in6,
                       const uint8_t *in7,
                       int inlen) {
    __m512i idx = _mm512_set_epi64(in7, in6, in5, in4, in3, in2, in1, in0);
    __m512i f;
    int tail = inlen & 7;
    int lanes = inlen >> 3;
    int i = 0;
    while (inlen >= SHAKE128_RATE) {
        for (int j = 0; j < 21; ++j) {
            f = _mm512_i64gather_epi64(idx, i + j * 8, 1);
            state->s[j] = _mm512_xor_si512(state->s[j], f);
        }
        i += SHAKE128_RATE;
        XURQ_keccak8x_permute(state->s);
        inlen -= SHAKE128_RATE;
    }

    state->s[20] ^= _mm512_set1_epi64(1ULL << 63);
}

void shake256x8_absorb(keccakx8_state *state,
                       const uint8_t *in0,
                       const uint8_t *in1,
                       const uint8_t *in2,
                       const uint8_t *in3,
                       const uint8_t *in4,
                       const uint8_t *in5,
                       const uint8_t *in6,
                       const uint8_t *in7,
                       int inlen) {
    __m512i idx = _mm512_set_epi64(in7, in6, in5, in4, in3, in2, in1, in0);

    __m512i f;
    int tail = inlen & 7;
    int lanes = inlen >> 3;
    int i = 0;
    while (inlen >= SHAKE256_RATE) {
        for (int j = 0; j < 17; ++j) {
            f = _mm512_i64gather_epi64(idx, i + j * 8, 1);
            state->s[j] = _mm512_xor_si512(state->s[j], f);
        }
        i += SHAKE256_RATE;
        XURQ_keccak8x_permute(state->s);
        inlen -= SHAKE256_RATE;
    }

    state->s[16] ^= _mm512_set1_epi64(1ULL << 63);
}



int keccak_squeezeblocks8x(unsigned char *h0,
                                   unsigned char *h1,
                                   unsigned char *h2,
                                   unsigned char *h3,
                                   unsigned char *h4,
                                   unsigned char *h5,
                                   unsigned char *h6,
                                   unsigned char *h7,
                                   unsigned long long int nblocks,
                                   __m512i *s,
                                   unsigned int r)
{
    unsigned int i;

    __m128d t;
    uint64_t cpuc, time = 0;
    while(nblocks > 0)
    {
        XURQ_keccak8x_permute(s);
        // cpuc = cpucycles();
        for(i=0;i<(r>>3);i++)
        {
            t = _mm_castsi128_pd(_mm512_castsi512_si128(s[i]));
            _mm_storel_pd((__attribute__((__may_alias__)) double *)&h0[8*i], t);
            _mm_storeh_pd((__attribute__((__may_alias__)) double *)&h1[8*i], t);
            t = _mm_castsi128_pd(_mm512_extracti64x2_epi64(s[i],1));
            _mm_storel_pd((__attribute__((__may_alias__)) double *)&h2[8*i], t);
            _mm_storeh_pd((__attribute__((__may_alias__)) double *)&h3[8*i], t);
            t = _mm_castsi128_pd(_mm512_extracti64x2_epi64(s[i],2));
            _mm_storel_pd((__attribute__((__may_alias__)) double *)&h4[8*i], t);
            _mm_storeh_pd((__attribute__((__may_alias__)) double *)&h5[8*i], t);
            t = _mm_castsi128_pd(_mm512_extracti64x2_epi64(s[i],3));
            _mm_storel_pd((__attribute__((__may_alias__)) double *)&h6[8*i], t);
            _mm_storeh_pd((__attribute__((__may_alias__)) double *)&h7[8*i], t);

        }
        h0 += r;
        h1 += r;
        h2 += r;
        h3 += r;
        h4 += r;
        h5 += r;
        h6 += r;
        h7 += r;
        nblocks--;
        // time += cpucycles() - cpuc;
    }
    return time;
}

int XURQ_AVX512_shake128x8_squeezeblocks(keccakx8_state *state,
                              uint8_t *out0,
                              uint8_t *out1,
                              uint8_t *out2,
                              uint8_t *out3,
                              uint8_t *out4,
                              uint8_t *out5,
                              uint8_t *out6,
                              uint8_t *out7,
                              int nblocks) {
    uint8_t *temp;
    __m512i t,t0, t1, t2, t3, t4, t5, t6, t7;
    __m512i f0, f1, f2, f3, f4, f5, f6, f7;
    __m256i z0,z1,z2,z3;
    uint64_t cpuc, time = 0;
    for (int i = 0; i < nblocks; ++i) {
        XURQ_keccak8x_permute(state->s);
        // cpuc = cpucycles();
        t0 = _mm512_unpacklo_epi64(state->s[0], state->s[1]); //66 44 22 00
        t1 = _mm512_unpackhi_epi64(state->s[0], state->s[1]); //77 55 33 11
        t2 = _mm512_unpacklo_epi64(state->s[2], state->s[3]); //66 44 22 00
        t3 = _mm512_unpackhi_epi64(state->s[2], state->s[3]); //77 55 33 11
        t4 = _mm512_unpacklo_epi64(state->s[4], state->s[5]); //66 44 22 00
        t5 = _mm512_unpackhi_epi64(state->s[4], state->s[5]); //77 55 33 11
        t6 = _mm512_unpacklo_epi64(state->s[6], state->s[7]); //66 44 22 00
        t7 = _mm512_unpackhi_epi64(state->s[6], state->s[7]); //77 55 33 11

        f0 = _mm512_shuffle_i32x4(t0, t2, 0x44);//22 00 22 00
        f2 = _mm512_shuffle_i32x4(t0, t2, 0xee);//66 44 66 44
        f4 = _mm512_shuffle_i32x4(t4, t6, 0x44);//22 00 22 00
        f6 = _mm512_shuffle_i32x4(t4, t6, 0xee);//66 44 66 44
        f1 = _mm512_shuffle_i32x4(t1, t3, 0x44); //33 11 33 11
        f3 = _mm512_shuffle_i32x4(t1, t3, 0xee);//77 55 77 55
        f5 = _mm512_shuffle_i32x4(t5, t7, 0x44); //33 11 33 11
        f7 = _mm512_shuffle_i32x4(t5, t7, 0xee);//77 55 77 55

        _mm512_storeu_si512(out0, _mm512_shuffle_i32x4(f0, f4, 0x88));
        _mm512_storeu_si512(out1, _mm512_shuffle_i32x4(f1, f5, 0x88));
        _mm512_storeu_si512(out2, _mm512_shuffle_i32x4(f0, f4, 0xdd));
        _mm512_storeu_si512(out3, _mm512_shuffle_i32x4(f1, f5, 0xdd));
        _mm512_storeu_si512(out4, _mm512_shuffle_i32x4(f2, f6, 0x88));
        _mm512_storeu_si512(out5, _mm512_shuffle_i32x4(f3, f7, 0x88));
        _mm512_storeu_si512(out6, _mm512_shuffle_i32x4(f2, f6, 0xdd));
        _mm512_storeu_si512(out7, _mm512_shuffle_i32x4(f3, f7, 0xdd));

        t0 = _mm512_unpacklo_epi64(state->s[8], state->s[9]); //66 44 22 00
        t1 = _mm512_unpackhi_epi64(state->s[8], state->s[9]); //77 55 33 11
        t2 = _mm512_unpacklo_epi64(state->s[10], state->s[11]); //66 44 22 00
        t3 = _mm512_unpackhi_epi64(state->s[10], state->s[11]); //77 55 33 11
        t4 = _mm512_unpacklo_epi64(state->s[12], state->s[13]); //66 44 22 00
        t5 = _mm512_unpackhi_epi64(state->s[12], state->s[13]); //77 55 33 11
        t6 = _mm512_unpacklo_epi64(state->s[14], state->s[15]); //66 44 22 00
        t7 = _mm512_unpackhi_epi64(state->s[14], state->s[15]); //77 55 33 11

        f0 = _mm512_shuffle_i32x4(t0, t2, 0x44);//22 00 22 00
        f2 = _mm512_shuffle_i32x4(t0, t2, 0xee);//66 44 66 44
        f4 = _mm512_shuffle_i32x4(t4, t6, 0x44);//22 00 22 00
        f6 = _mm512_shuffle_i32x4(t4, t6, 0xee);//66 44 66 44
        f1 = _mm512_shuffle_i32x4(t1, t3, 0x44); //33 11 33 11
        f3 = _mm512_shuffle_i32x4(t1, t3, 0xee);//77 55 77 55
        f5 = _mm512_shuffle_i32x4(t5, t7, 0x44); //33 11 33 11
        f7 = _mm512_shuffle_i32x4(t5, t7, 0xee);//77 55 77 55

        _mm512_storeu_si512(out0 + 64, _mm512_shuffle_i32x4(f0, f4, 0x88));
        _mm512_storeu_si512(out1 + 64, _mm512_shuffle_i32x4(f1, f5, 0x88));
        _mm512_storeu_si512(out2 + 64, _mm512_shuffle_i32x4(f0, f4, 0xdd));
        _mm512_storeu_si512(out3 + 64, _mm512_shuffle_i32x4(f1, f5, 0xdd));
        _mm512_storeu_si512(out4 + 64, _mm512_shuffle_i32x4(f2, f6, 0x88));
        _mm512_storeu_si512(out5 + 64, _mm512_shuffle_i32x4(f3, f7, 0x88));
        _mm512_storeu_si512(out6 + 64, _mm512_shuffle_i32x4(f2, f6, 0xdd));
        _mm512_storeu_si512(out7 + 64, _mm512_shuffle_i32x4(f3, f7, 0xdd));

        t0 = _mm512_unpacklo_epi64(state->s[16], state->s[17]); //66 44 22 00
        t1 = _mm512_unpackhi_epi64(state->s[16], state->s[17]); //77 55 33 11
        t2 = _mm512_unpacklo_epi64(state->s[18], state->s[19]); //66 44 22 00
        t3 = _mm512_unpackhi_epi64(state->s[18], state->s[19]); //77 55 33 11

        z0 = _mm512_extracti32x8_epi32(t0,0);//22 00
        z1 = _mm512_extracti32x8_epi32(t2,0);//22 00
        z2 = _mm256_permute2x128_si256(z0,z1,0x20);// 00 00
        z3 = _mm256_permute2x128_si256(z0,z1,0x31);// 22 22
        _mm256_storeu_si256(out0 + 128,z2);
        _mm256_storeu_si256(out2 + 128,z3);

        z0 = _mm512_extracti32x8_epi32(t0,1);//66 44
        z1 = _mm512_extracti32x8_epi32(t2,1);//66 44
        z2 = _mm256_permute2x128_si256(z0,z1,0x20);// 44 44
        z3 = _mm256_permute2x128_si256(z0,z1,0x31);// 66 66
        _mm256_storeu_si256(out4 + 128,z2);
        _mm256_storeu_si256(out6 + 128,z3);

        z0 = _mm512_extracti32x8_epi32(t1,0);//33 11
        z1 = _mm512_extracti32x8_epi32(t3,0);//33 11
        z2 = _mm256_permute2x128_si256(z0,z1,0x20);// 11 11
        z3 = _mm256_permute2x128_si256(z0,z1,0x31);// 33 33
        _mm256_storeu_si256(out1 + 128,z2);
        _mm256_storeu_si256(out3 + 128,z3);

        z0 = _mm512_extracti32x8_epi32(t1,1);//77 55
        z1 = _mm512_extracti32x8_epi32(t3,1);//77 55
        z2 = _mm256_permute2x128_si256(z0,z1,0x20);// 55 55
        z3 = _mm256_permute2x128_si256(z0,z1,0x31);// 77 77
        _mm256_storeu_si256(out5 + 128,z2);
        _mm256_storeu_si256(out7 + 128,z3);


        temp = (uint8_t *) (&state->s[20]);
        memcpy(out0 + 160, temp, 8);
        memcpy(out1 + 160, temp + 8, 8);
        memcpy(out2 + 160, temp + 16, 8);
        memcpy(out3 + 160, temp + 24, 8);
        memcpy(out4 + 160, temp + 32, 8);
        memcpy(out5 + 160, temp + 40, 8);
        memcpy(out6 + 160, temp + 48, 8);
        memcpy(out7 + 160, temp + 56, 8);

        out0 += 168;
        out1 += 168;
        out2 += 168;
        out3 += 168;
        out4 += 168;
        out5 += 168;
        out6 += 168;
        out7 += 168;
        // time += cpucycles() - cpuc;
    }
    return time;
}



int XURQ_AVX512_shake256x8_squeezeblocks(keccakx8_state *state,
                              uint8_t *out0,
                              uint8_t *out1,
                              uint8_t *out2,
                              uint8_t *out3,
                              uint8_t *out4,
                              uint8_t *out5,
                              uint8_t *out6,
                              uint8_t *out7,
                              int nblocks) {
    uint8_t *temp;
    __m512i t,t0, t1, t2, t3, t4, t5, t6, t7;
    __m512i f0, f1, f2, f3, f4, f5, f6, f7;
    uint64_t cpuc, time = 0;
    for (int i = 0; i < nblocks; ++i) {
        XURQ_keccak8x_permute(state->s);
        // cpuc = cpucycles();
        t0 = _mm512_unpacklo_epi64(state->s[0], state->s[1]); //66 44 22 00
        t1 = _mm512_unpackhi_epi64(state->s[0], state->s[1]); //77 55 33 11
        t2 = _mm512_unpacklo_epi64(state->s[2], state->s[3]); //66 44 22 00
        t3 = _mm512_unpackhi_epi64(state->s[2], state->s[3]); //77 55 33 11
        t4 = _mm512_unpacklo_epi64(state->s[4], state->s[5]); //66 44 22 00
        t5 = _mm512_unpackhi_epi64(state->s[4], state->s[5]); //77 55 33 11
        t6 = _mm512_unpacklo_epi64(state->s[6], state->s[7]); //66 44 22 00
        t7 = _mm512_unpackhi_epi64(state->s[6], state->s[7]); //77 55 33 11

        f0 = _mm512_shuffle_i32x4(t0, t2, 0x44);//22 00 22 00
        f2 = _mm512_shuffle_i32x4(t0, t2, 0xee);//66 44 66 44
        f4 = _mm512_shuffle_i32x4(t4, t6, 0x44);//22 00 22 00
        f6 = _mm512_shuffle_i32x4(t4, t6, 0xee);//66 44 66 44
        f1 = _mm512_shuffle_i32x4(t1, t3, 0x44); //33 11 33 11
        f3 = _mm512_shuffle_i32x4(t1, t3, 0xee);//77 55 77 55
        f5 = _mm512_shuffle_i32x4(t5, t7, 0x44); //33 11 33 11
        f7 = _mm512_shuffle_i32x4(t5, t7, 0xee);//77 55 77 55

        _mm512_storeu_si512(out0, _mm512_shuffle_i32x4(f0, f4, 0x88));
        _mm512_storeu_si512(out1, _mm512_shuffle_i32x4(f1, f5, 0x88));
        _mm512_storeu_si512(out2, _mm512_shuffle_i32x4(f0, f4, 0xdd));
        _mm512_storeu_si512(out3, _mm512_shuffle_i32x4(f1, f5, 0xdd));
        _mm512_storeu_si512(out4, _mm512_shuffle_i32x4(f2, f6, 0x88));
        _mm512_storeu_si512(out5, _mm512_shuffle_i32x4(f3, f7, 0x88));
        _mm512_storeu_si512(out6, _mm512_shuffle_i32x4(f2, f6, 0xdd));
        _mm512_storeu_si512(out7, _mm512_shuffle_i32x4(f3, f7, 0xdd));

        t0 = _mm512_unpacklo_epi64(state->s[8], state->s[9]); //66 44 22 00
        t1 = _mm512_unpackhi_epi64(state->s[8], state->s[9]); //77 55 33 11
        t2 = _mm512_unpacklo_epi64(state->s[10], state->s[11]); //66 44 22 00
        t3 = _mm512_unpackhi_epi64(state->s[10], state->s[11]); //77 55 33 11
        t4 = _mm512_unpacklo_epi64(state->s[12], state->s[13]); //66 44 22 00
        t5 = _mm512_unpackhi_epi64(state->s[12], state->s[13]); //77 55 33 11
        t6 = _mm512_unpacklo_epi64(state->s[14], state->s[15]); //66 44 22 00
        t7 = _mm512_unpackhi_epi64(state->s[14], state->s[15]); //77 55 33 11

        f0 = _mm512_shuffle_i32x4(t0, t2, 0x44);//22 00 22 00
        f2 = _mm512_shuffle_i32x4(t0, t2, 0xee);//66 44 66 44
        f4 = _mm512_shuffle_i32x4(t4, t6, 0x44);//22 00 22 00
        f6 = _mm512_shuffle_i32x4(t4, t6, 0xee);//66 44 66 44
        f1 = _mm512_shuffle_i32x4(t1, t3, 0x44); //33 11 33 11
        f3 = _mm512_shuffle_i32x4(t1, t3, 0xee);//77 55 77 55
        f5 = _mm512_shuffle_i32x4(t5, t7, 0x44); //33 11 33 11
        f7 = _mm512_shuffle_i32x4(t5, t7, 0xee);//77 55 77 55

        _mm512_storeu_si512(out0 + 64, _mm512_shuffle_i32x4(f0, f4, 0x88));
        _mm512_storeu_si512(out1 + 64, _mm512_shuffle_i32x4(f1, f5, 0x88));
        _mm512_storeu_si512(out2 + 64, _mm512_shuffle_i32x4(f0, f4, 0xdd));
        _mm512_storeu_si512(out3 + 64, _mm512_shuffle_i32x4(f1, f5, 0xdd));
        _mm512_storeu_si512(out4 + 64, _mm512_shuffle_i32x4(f2, f6, 0x88));
        _mm512_storeu_si512(out5 + 64, _mm512_shuffle_i32x4(f3, f7, 0x88));
        _mm512_storeu_si512(out6 + 64, _mm512_shuffle_i32x4(f2, f6, 0xdd));
        _mm512_storeu_si512(out7 + 64, _mm512_shuffle_i32x4(f3, f7, 0xdd));


        temp = (uint8_t *) (&state->s[16]);
        memcpy(out0 + 128, temp, 8);
        memcpy(out1 + 128, temp + 8, 8);
        memcpy(out2 + 128, temp + 16, 8);
        memcpy(out3 + 128, temp + 24, 8);
        memcpy(out4 + 128, temp + 32, 8);
        memcpy(out5 + 128, temp + 40, 8);
        memcpy(out6 + 128, temp + 48, 8);
        memcpy(out7 + 128, temp + 56, 8);

        out0 += 136;
        out1 += 136;
        out2 += 136;
        out3 += 136;
        out4 += 136;
        out5 += 136;
        out6 += 136;
        out7 += 136;
        // time += cpucycles() - cpuc;
    }
    return time;
}

static uint64_t load64(const uint8_t *x) {
    uint64_t r;
    memcpy(&r, x, sizeof(uint64_t));
    return r;
}

int XURQ_AVX512_shake128x8_absorb(keccakx8_state *state,
                              uint8_t *out0,
                              uint8_t *out1,
                              uint8_t *out2,
                              uint8_t *out3,
                              uint8_t *out4,
                              uint8_t *out5,
                              uint8_t *out6,
                              uint8_t *out7,
                              int len) {
    uint8_t *temp;
    __m512i t0, t1, t2, t3, t4, t5, t6, t7;
    __m512i f0, f1, f2, f3, f4, f5, f6, f7;

    uint64_t t;
    while (len >= SHAKE128_RATE) {
        f0 = _mm512_loadu_si512(out0);
        f1 = _mm512_loadu_si512(out0+64);
        f2 = _mm512_loadu_si512(out0+128);


        t0 = _mm512_unpacklo_epi64(f0, f0); //66 44 22 00
        t1 = _mm512_unpackhi_epi64(f0, f0); //77 55 33 11

        f0 = _mm512_shuffle_i32x4(t0, t0, 0x44);//22 00 22 00
        f2 = _mm512_shuffle_i32x4(t0, t0, 0xee);//66 44 66 44
        f1 = _mm512_shuffle_i32x4(t1, t1, 0x44); //33 11 33 11
        f3 = _mm512_shuffle_i32x4(t1, t1, 0xee);//77 55 77 55

        state->s[0] = _mm512_xor_si512 (state->s[0], _mm512_shuffle_i32x4(f0, f0, 0x88));
        state->s[1] = _mm512_xor_si512 (state->s[1], _mm512_shuffle_i32x4(f1, f1, 0x88));
        state->s[2] = _mm512_xor_si512 (state->s[2], _mm512_shuffle_i32x4(f0, f0, 0xdd));
        state->s[3] = _mm512_xor_si512 (state->s[3], _mm512_shuffle_i32x4(f1, f1, 0xdd));
        state->s[4] = _mm512_xor_si512 (state->s[4], _mm512_shuffle_i32x4(f2, f2, 0x88));
        state->s[5] = _mm512_xor_si512 (state->s[5], _mm512_shuffle_i32x4(f3, f3, 0x88));
        state->s[6] = _mm512_xor_si512 (state->s[6], _mm512_shuffle_i32x4(f2, f2, 0xdd));
        state->s[7] = _mm512_xor_si512 (state->s[7], _mm512_shuffle_i32x4(f3, f3, 0xdd));



        t0 = _mm512_unpacklo_epi64(f1, f1); //66 44 22 00
        t1 = _mm512_unpackhi_epi64(f1, f1); //77 55 33 11

        f0 = _mm512_shuffle_i32x4(t0, t0, 0x44);//22 00 22 00
        f2 = _mm512_shuffle_i32x4(t0, t0, 0xee);//66 44 66 44
        f1 = _mm512_shuffle_i32x4(t1, t1, 0x44); //33 11 33 11
        f3 = _mm512_shuffle_i32x4(t1, t1, 0xee);//77 55 77 55

        state->s[8]  =   _mm512_xor_si512 (state->s[8] , _mm512_shuffle_i32x4(f0, f0, 0x88));
        state->s[9]  =   _mm512_xor_si512 (state->s[9] , _mm512_shuffle_i32x4(f1, f1, 0x88));
        state->s[10] =   _mm512_xor_si512 (state->s[10], _mm512_shuffle_i32x4(f0, f0, 0xdd));
        state->s[11] =   _mm512_xor_si512 (state->s[11], _mm512_shuffle_i32x4(f1, f1, 0xdd));
        state->s[12] =   _mm512_xor_si512 (state->s[12], _mm512_shuffle_i32x4(f2, f2, 0x88));
        state->s[13] =   _mm512_xor_si512 (state->s[13], _mm512_shuffle_i32x4(f3, f3, 0x88));
        state->s[14] =   _mm512_xor_si512 (state->s[14], _mm512_shuffle_i32x4(f2, f2, 0xdd));
        state->s[15] =   _mm512_xor_si512 (state->s[15], _mm512_shuffle_i32x4(f3, f3, 0xdd));


        t0 = _mm512_unpacklo_epi64(f2, f2); //66 44 22 00
        t1 = _mm512_unpackhi_epi64(f2, f2); //77 55 33 11

        f0 = _mm512_shuffle_i32x4(t0, t0, 0x44);//22 00 22 00
        f2 = _mm512_shuffle_i32x4(t0, t0, 0xee);//66 44 66 44
        f1 = _mm512_shuffle_i32x4(t1, t1, 0x44); //33 11 33 11

        state->s[16]  =   _mm512_xor_si512 (state->s[8] , _mm512_shuffle_i32x4(f0, f0, 0x88));
        state->s[17]  =   _mm512_xor_si512 (state->s[9] , _mm512_shuffle_i32x4(f1, f1, 0x88));
        state->s[18] =   _mm512_xor_si512 (state->s[10], _mm512_shuffle_i32x4(f0, f0, 0xdd));
        state->s[19] =   _mm512_xor_si512 (state->s[11], _mm512_shuffle_i32x4(f1, f1, 0xdd));
        state->s[20] =   _mm512_xor_si512 (state->s[12], _mm512_shuffle_i32x4(f2, f2, 0x88));

        XURQ_keccak8x_permute(state->s);
        len -= SHAKE128_RATE;
        out0 += 168;

    }
    int i = 0;
    while (len > 0) {
        t = load64(out0 + i * 8);
        f0 = _mm512_set1_epi64(t);
        state->s[i] = _mm512_xor_si512(state->s[i], f0);
        len -= i * 8;
        i++;
    }
    f0 = _mm512_set1_epi64(0x1ULL << 63);
    state->s[i] = _mm512_xor_si512(state->s[i], f0);

    return 0;
}


int XURQ_AVX512_shake256x8_absorb(keccakx8_state *state,
                              uint8_t *out0,
                              uint8_t *out1,
                              uint8_t *out2,
                              uint8_t *out3,
                              uint8_t *out4,
                              uint8_t *out5,
                              uint8_t *out6,
                              uint8_t *out7,
                              int len) {
    uint8_t *temp;
    __m512i t0, t1, t2, t3, t4, t5, t6, t7;
    __m512i f0, f1, f2, f3, f4, f5, f6, f7;

    uint64_t t;
    while (len >= SHAKE256_RATE) {
        f0 = _mm512_loadu_si512(out0);
        f1 = _mm512_loadu_si512(out0+64);

        t0 = _mm512_unpacklo_epi64(f0, f0); //66 44 22 00
        t1 = _mm512_unpackhi_epi64(f0, f0); //77 55 33 11

        f0 = _mm512_shuffle_i32x4(t0, t0, 0x44);//22 00 22 00
        f2 = _mm512_shuffle_i32x4(t0, t0, 0xee);//66 44 66 44
        f1 = _mm512_shuffle_i32x4(t1, t1, 0x44); //33 11 33 11
        f3 = _mm512_shuffle_i32x4(t1, t1, 0xee);//77 55 77 55

        state->s[0] = _mm512_xor_si512 (state->s[0], _mm512_shuffle_i32x4(f0, f0, 0x88));
        state->s[1] = _mm512_xor_si512 (state->s[1], _mm512_shuffle_i32x4(f1, f1, 0x88));
        state->s[2] = _mm512_xor_si512 (state->s[2], _mm512_shuffle_i32x4(f0, f0, 0xdd));
        state->s[3] = _mm512_xor_si512 (state->s[3], _mm512_shuffle_i32x4(f1, f1, 0xdd));
        state->s[4] = _mm512_xor_si512 (state->s[4], _mm512_shuffle_i32x4(f2, f2, 0x88));
        state->s[5] = _mm512_xor_si512 (state->s[5], _mm512_shuffle_i32x4(f3, f3, 0x88));
        state->s[6] = _mm512_xor_si512 (state->s[6], _mm512_shuffle_i32x4(f2, f2, 0xdd));
        state->s[7] = _mm512_xor_si512 (state->s[7], _mm512_shuffle_i32x4(f3, f3, 0xdd));


        t0 = _mm512_unpacklo_epi64(f1, f1); //66 44 22 00
        t1 = _mm512_unpackhi_epi64(f1, f1); //77 55 33 11

        f0 = _mm512_shuffle_i32x4(t0, t0, 0x44);//22 00 22 00
        f2 = _mm512_shuffle_i32x4(t0, t0, 0xee);//66 44 66 44
        f1 = _mm512_shuffle_i32x4(t1, t1, 0x44); //33 11 33 11
        f3 = _mm512_shuffle_i32x4(t1, t1, 0xee);//77 55 77 55

        state->s[8]  =   _mm512_xor_si512 (state->s[8] , _mm512_shuffle_i32x4(f0, f0, 0x88));
        state->s[9]  =   _mm512_xor_si512 (state->s[9] , _mm512_shuffle_i32x4(f1, f1, 0x88));
        state->s[10] =   _mm512_xor_si512 (state->s[10], _mm512_shuffle_i32x4(f0, f0, 0xdd));
        state->s[11] =   _mm512_xor_si512 (state->s[11], _mm512_shuffle_i32x4(f1, f1, 0xdd));
        state->s[12] =   _mm512_xor_si512 (state->s[12], _mm512_shuffle_i32x4(f2, f2, 0x88));
        state->s[13] =   _mm512_xor_si512 (state->s[13], _mm512_shuffle_i32x4(f3, f3, 0x88));
        state->s[14] =   _mm512_xor_si512 (state->s[14], _mm512_shuffle_i32x4(f2, f2, 0xdd));
        state->s[15] =   _mm512_xor_si512 (state->s[15], _mm512_shuffle_i32x4(f3, f3, 0xdd));

        t = load64(out0 + 128);
        f0 = _mm512_set1_epi64(t);
        state->s[16] = _mm512_xor_si512(state->s[16], f0);

        XURQ_keccak8x_permute(state->s);
        len -= SHAKE256_RATE;
        out0 += 136;

    }
    int i = 0;
    while (len > 0) {
        t = load64(out0 + i * 8);
        f0 = _mm512_set1_epi64(t);
        state->s[i] = _mm512_xor_si512(state->s[i], f0);
        len -= i * 8;
        i++;
    }
    f0 = _mm512_set1_epi64(0x1ULL << 63);
    state->s[i] = _mm512_xor_si512(state->s[i], f0);

    return 0;
}
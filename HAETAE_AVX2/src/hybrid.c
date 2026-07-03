//
// Created by 59531 on 2025/11/29.
//

#include "../include/hybrid.h"

#include <stdio.h>
#include <string.h>
#include "fips202x4.h"


uint8_t *loop_dequeue(loop_queue *loop) {
    int now = loop->start;
    loop->start = (now + 1) % LOOP_CAPABILITY;
    loop->size -= 1;
    return loop->buf[now].coeffs;
}

uint8_t *loop_next(loop_queue *loop) {
    int next = (loop->start + loop->size) % LOOP_CAPABILITY;
    loop->size += 1;
    return loop->buf[next].coeffs;
}

void print_loop_state(loop_queue *loop) {
    printf("loop.start: %d, loop.szie: %d\n", loop->start, loop->size);
}

uint8_t *loop_kg_dequeue(loop_queue_kg *loop) {
    int now = loop->start;
    loop->start = (now + 1) % LOOP_KG_CAPABILITY;
    loop->size -= 1;
    return loop->buf[now].coeffs;
}

uint8_t *loop_kg_next(loop_queue_kg *loop) {
    int next = (loop->start + loop->size) % LOOP_KG_CAPABILITY;
    loop->size += 1;
    return loop->buf[next].coeffs;
}

void print_loop_kg_state(loop_queue_kg *loop) {
    printf("loop.start: %d, loop.szie: %d\n", loop->start, loop->size);
}

static uint64_t load64(const uint8_t *x) {
    uint64_t r;
    memcpy(&r, x, sizeof(uint64_t));
    return r;
}

static void absorb_hash_SHAKE256RATE(keccakx4_state *state, const uint8_t *pk) {
    __m256i f;
    __m256i zero = _mm256_setzero_si256();
    for (int i = 0; i < 17; ++i) {
        f = _mm256_insert_epi64(zero, load64(pk + i * 8), 3);
        state->s[i] = _mm256_xor_si256(state->s[i], f);
    }
}

static uint64_t load_in(const uint8_t *x, int len) {
    uint64_t r = 0;
    for (int i = 0; i < len; ++i) {
        r |= (uint64_t)x[i] << (8 * i);
    }
    return r;
}

static void absorb_hash_SHAKE256RATE_tail_process(keccakx4_state *state, const uint8_t *pk, int tail_len) {
    __m256i f;
    int l = tail_len & 7;
    int i;
    for (i = 0; i < tail_len / 8; ++i) {
        f = _mm256_set_epi64x(load64(pk + i * 8), 0, 0, 0);
        state->s[i] = _mm256_xor_si256(state->s[i], f);
    }
    f = _mm256_set_epi64x((0x1FULL << (l * 8)) | load_in(pk + i * 8, l), 0, 0, 0);
    state->s[tail_len / 8] = _mm256_xor_si256(state->s[tail_len / 8], f);
    f = _mm256_set_epi64x(0x1ULL << 63, 0, 0, 0);
    state->s[16] = _mm256_xor_si256(state->s[16], f);
}

static void refresh_ExpandA_states_x3(keccakx4_state *state, const keccakx4_state32 *rhoprime, uint16_t nonce0,
                                             uint16_t nonce1, uint16_t nonce2) {
    const __m256i mask = _mm256_set_epi64x(UINT64_MAX, 0, 0, 0);
    const __m256i f5 = _mm256_set_epi64x(0, 0x1ULL << 63, 0x1ULL << 63, 0x1ULL << 63);
    __m256i f;
    for (int i = 0; i < 25; ++i) {
        state->s[i] = _mm256_and_si256(state->s[i], mask);
    }

    state->s[0] = _mm256_xor_si256(state->s[0], _mm256_andnot_si256(mask, rhoprime->s[0])) ;
    state->s[1] = _mm256_xor_si256(state->s[1], _mm256_andnot_si256(mask, rhoprime->s[1])) ;
    state->s[2] = _mm256_xor_si256(state->s[2], _mm256_andnot_si256(mask, rhoprime->s[2])) ;
    state->s[3] = _mm256_xor_si256(state->s[3], _mm256_andnot_si256(mask, rhoprime->s[3])) ;
    f = _mm256_set_epi64x(0, (0x1f << 16) ^ nonce2, (0x1f << 16) ^ nonce1, (0x1f << 16) ^ nonce0);
    state->s[4] = _mm256_xor_si256(state->s[4],f);
    state->s[20] = _mm256_xor_si256(state->s[20], f5);
}

void extract_lanes_x3_shake128(uint8_t *out0, uint8_t *out1, uint8_t *out2, keccakx4_state *state) {
    __m256i t0, t1, t2, t3, t4, t5, t6, t7;
    __m256i f0, f1, f2, f3, f4, f5, f6, f7;
    __m128i t;


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

    f4 = _mm256_permute2x128_si256(t4, t6, 0x20);
    f5 = _mm256_permute2x128_si256(t5, t7, 0x20);
    f6 = _mm256_permute2x128_si256(t4, t6, 0x31);


    _mm256_storeu_si256((__m256i *) out0, f0);
    _mm256_storeu_si256((__m256i *) out1, f1);
    _mm256_storeu_si256((__m256i *) out2, f2);
    _mm256_storeu_si256((__m256i *) (out0 + 32), f4);
    _mm256_storeu_si256((__m256i *) (out1 + 32), f5);
    _mm256_storeu_si256((__m256i *) (out2 + 32), f6);

    out0 += 64;
    out1 += 64;
    out2 += 64;

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

    f4 = _mm256_permute2x128_si256(t4, t6, 0x20);
    f5 = _mm256_permute2x128_si256(t5, t7, 0x20);
    f6 = _mm256_permute2x128_si256(t4, t6, 0x31);

    _mm256_storeu_si256((__m256i *) out0, f0);
    _mm256_storeu_si256((__m256i *) out1, f1);
    _mm256_storeu_si256((__m256i *) out2, f2);
    _mm256_storeu_si256((__m256i *) (out0 + 32), f4);
    _mm256_storeu_si256((__m256i *) (out1 + 32), f5);
    _mm256_storeu_si256((__m256i *) (out2 + 32), f6);

    out0 += 64;
    out1 += 64;
    out2 += 64;

    t0 = _mm256_unpacklo_epi64(state->s[16], state->s[17]);
    t1 = _mm256_unpackhi_epi64(state->s[16], state->s[17]);
    t2 = _mm256_unpacklo_epi64(state->s[18], state->s[19]);
    t3 = _mm256_unpackhi_epi64(state->s[18], state->s[19]);

    f0 = _mm256_permute2x128_si256(t0, t2, 0x20);
    f1 = _mm256_permute2x128_si256(t1, t3, 0x20);
    f2 = _mm256_permute2x128_si256(t0, t2, 0x31);

    _mm256_storeu_si256((__m256i *) out0, f0);
    _mm256_storeu_si256((__m256i *) out1, f1);
    _mm256_storeu_si256((__m256i *) out2, f2);

    out0 += 32;
    out1 += 32;
    out2 += 32;

    t = _mm256_castsi256_si128(state->s[20]);
    _mm_storeu_si64(out0, t);
    _mm_storeu_si64(out1, _mm_bsrli_si128(t, 8));
    t = _mm256_extracti128_si256(state->s[20], 1);
    _mm_storeu_si64(out2, t);
}

static void shake128x3_squeezeblocks(uint8_t *out0, uint8_t *out1, uint8_t *out2, int nblocks, keccakx4_state *state) {
    __m256i t0, t1, t2, t3, t4, t5, t6, t7;
    __m256i f0, f1, f2, f3, f4, f5, f6, f7;
    __m128i t;

    for (int i = 0; i < nblocks; ++i) {
        // KeccakP1600times4_PermuteAll_24rounds(state->s);
        f1600x4(state->s, KeccakF_RoundConstants);

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

        f4 = _mm256_permute2x128_si256(t4, t6, 0x20);
        f5 = _mm256_permute2x128_si256(t5, t7, 0x20);
        f6 = _mm256_permute2x128_si256(t4, t6, 0x31);

        _mm256_storeu_si256((__m256i *) out0, f0);
        _mm256_storeu_si256((__m256i *) out1, f1);
        _mm256_storeu_si256((__m256i *) out2, f2);
        _mm256_storeu_si256((__m256i *) (out0 + 32), f4);
        _mm256_storeu_si256((__m256i *) (out1 + 32), f5);
        _mm256_storeu_si256((__m256i *) (out2 + 32), f6);

        out0 += 64;
        out1 += 64;
        out2 += 64;

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

        f4 = _mm256_permute2x128_si256(t4, t6, 0x20);
        f5 = _mm256_permute2x128_si256(t5, t7, 0x20);
        f6 = _mm256_permute2x128_si256(t4, t6, 0x31);

        _mm256_storeu_si256((__m256i *) out0, f0);
        _mm256_storeu_si256((__m256i *) out1, f1);
        _mm256_storeu_si256((__m256i *) out2, f2);
        _mm256_storeu_si256((__m256i *) (out0 + 32), f4);
        _mm256_storeu_si256((__m256i *) (out1 + 32), f5);
        _mm256_storeu_si256((__m256i *) (out2 + 32), f6);

        out0 += 64;
        out1 += 64;
        out2 += 64;

        t0 = _mm256_unpacklo_epi64(state->s[16], state->s[17]);
        t1 = _mm256_unpackhi_epi64(state->s[16], state->s[17]);
        t2 = _mm256_unpacklo_epi64(state->s[18], state->s[19]);
        t3 = _mm256_unpackhi_epi64(state->s[18], state->s[19]);

        f0 = _mm256_permute2x128_si256(t0, t2, 0x20);
        f1 = _mm256_permute2x128_si256(t1, t3, 0x20);
        f2 = _mm256_permute2x128_si256(t0, t2, 0x31);

        _mm256_storeu_si256((__m256i *) out0, f0);
        _mm256_storeu_si256((__m256i *) out1, f1);
        _mm256_storeu_si256((__m256i *) out2, f2);

        out0 += 32;
        out1 += 32;
        out2 += 32;

        t = _mm256_castsi256_si128(state->s[20]);
        *(uint64_t *) out0 = _mm_extract_epi64(t, 0);
        *(uint64_t *) out1 = _mm_extract_epi64(t, 1);
        t = _mm256_extracti128_si256(state->s[20], 1);
        *(uint64_t *) out2 = _mm_extract_epi64(t, 0);

        out0 += 8;
        out1 += 8;
        out2 += 8;
    }
}


void polymatkl_expand_part(polyvecl mat[K], const keccakx4_state32 *rho) {
    unsigned int i, j;


#if K == 2 && L == 4
return;
#elif K == 3 && L == 6
    poly tmp;
    poly_uniform_4x_state(
        &tmp,
        &mat[1].vec[5],
        &mat[2].vec[1],
        &mat[2].vec[2],
        rho,
        (1 << 8) + 3,
        (1 << 8) + 4,
        (2 << 8) + 0,
        (2 << 8) + 1
        );
    poly_uniform_4x_state(
        &mat[2].vec[3],
        &mat[2].vec[4],
        &mat[2].vec[5],
        &tmp,
        rho,
        (2 << 8) + 2,
        (2 << 8) + 3,
        (2 << 8) + 4,
        (9 << 8) + 9
        );
#elif K == 4 && L == 7

  poly_uniform_4x_state(&mat[0].vec[4],
                        &mat[1].vec[4],
                        &mat[2].vec[4],
                        &mat[3].vec[4],
                        rho,
                        (0 << 8) + 3,
                        (1 << 8) + 3,
                        (2 << 8) + 3,
                        (3 << 8) + 3);

  poly_uniform_4x_state(&mat[0].vec[5],
                        &mat[1].vec[5],
                        &mat[2].vec[5],
                        &mat[3].vec[5],
                        rho,
                        (0 << 8) + 4,
                        (1 << 8) + 4,
                        (2 << 8) + 4,
                        (3 << 8) + 4);

  poly_uniform_4x_state(&mat[0].vec[6],
                        &mat[1].vec[6],
                        &mat[2].vec[6],
                        &mat[3].vec[6],
                        rho,
                        (0 << 8) + 5,
                        (1 << 8) + 5,
                        (2 << 8) + 5,
                        (3 << 8) + 5);
#else
#error
#endif
}

struct coroutine {
    int max;
    int round;
    unsigned int ctr[3];
};

static void uniform_x3(keccakx4_state *state, polyvecl mat[K], uint8_t buf[4][192], const keccakx4_state32 *rho,
                struct coroutine *cor) {
    poly *a0, *a1, *a2;
    int nonce0, nonce1, nonce2;
#if K == 2 && L == 4
    switch (cor->round) {
        case 0:
            a0 = &mat[0].vec[1];
            a1 = &mat[0].vec[2];
            a2 = &mat[0].vec[3];
            //nonces for next round
            nonce0 = (1 << 8) + 0;
            nonce1 = (1 << 8) + 1;
            nonce2 = (1 << 8) + 2;
            break;
        case 1:
            a0 = &mat[1].vec[1];
            a1 = &mat[1].vec[2];
            a2 = &mat[1].vec[3];
            break;
        default:
            return;
    }
#elif K == 3 && L == 6
    switch (cor->round) {
        case 0:
            a0 = &mat[0].vec[1];
            a1 = &mat[0].vec[2];
            a2 = &mat[0].vec[3];
            //nonces for next round
            nonce0 = (0 << 8) + 3;
            nonce1 = (0 << 8) + 4;
            nonce2 = (1 << 8) + 0;
            break;
        case 1:
            a0 = &mat[0].vec[4];
            a1 = &mat[0].vec[5];
            a2 = &mat[1].vec[1];
            //nonces for next round
            nonce0 = (1 << 8) + 1;
            nonce1 = (1 << 8) + 2;
            nonce2 = (1 << 8) + 3;
            break;
        case 2:
            a0 = &mat[1].vec[2];
            a1 = &mat[1].vec[3];
            a2 = &mat[1].vec[4];
            break;
        default:
            return;
    }
#elif K == 4 && L == 7
    switch (cor->round) {
        case 0:
            a0 = &mat[0].vec[1];
            a1 = &mat[1].vec[1];
            a2 = &mat[2].vec[1];
            //nonces for next round
            nonce0 = (3 << 8) + 0;
            nonce1 = (0 << 8) + 1;
            nonce2 = (1 << 8) + 1;
            break;
        case 1:
            a0 = &mat[3].vec[1];
            a1 = &mat[0].vec[2];
            a2 = &mat[1].vec[2];
            //nonces for next round
            nonce0 = (2 << 8) + 1;
            nonce1 = (3 << 8) + 1;
            nonce2 = (0 << 8) + 2;
            break;
        case 2:
            a0 = &mat[2].vec[2];
            a1 = &mat[3].vec[2];
            a2 = &mat[0].vec[3];
            //nonces for next round
            nonce0 = (1 << 8) + 2;
            nonce1 = (2 << 8) + 2;
            nonce2 = (3 << 8) + 2;
            break;
        case 3:
            a0 = &mat[1].vec[3];
            a1 = &mat[2].vec[3];
            a2 = &mat[3].vec[3];
            break;
        default:
            return;
    }
#endif

     if (cor->ctr[0] < N || cor->ctr[1] < N || cor->ctr[2] < N) {
        cor->ctr[0] += rej_uniform(a0->coeffs + cor->ctr[0],N - cor->ctr[0], buf[0], SHAKE128_RATE);
        cor->ctr[1] += rej_uniform(a1->coeffs + cor->ctr[1],N - cor->ctr[1], buf[1], SHAKE128_RATE);
        cor->ctr[2] += rej_uniform(a2->coeffs + cor->ctr[2],N - cor->ctr[2], buf[2], SHAKE128_RATE);
        if (cor->ctr[0] >= N && cor->ctr[1] >= N && cor->ctr[2] >= N && cor->round < cor->max) {
            cor->round++;
            refresh_ExpandA_states_x3(state, rho, nonce0, nonce1, nonce2);
            cor->ctr[0] = 0;
            cor->ctr[1] = 0;
            cor->ctr[2] = 0;
        }
    }
}

void hybrid_hash_pk_and_ExpandA(uint8_t *hash_out, const uint8_t *pkm, int hash_in_len, polyvecl mat[K], const keccakx4_state32 *rhoprhime) {
    struct coroutine cor = {.max = K - 1, .round = 0, .ctr = {0}};
    __attribute__ ((aligned(32))) uint8_t buf[4][192];
    keccakx4_state state;
    uint8_t *pk_ptr = pkm;
    int tail_len = hash_in_len % SHAKE256_RATE;

    for (int j = 0; j < 25; ++j) state.s[j] = _mm256_setzero_si256();
#if K == 4 && L == 7
    refresh_ExpandA_states_x3(&state, rhoprhime, (0 << 8) + 0, (1 << 8) + 0, (2 << 8) + 0);
#else
    refresh_ExpandA_states_x3(&state, rhoprhime, 0, 1, 2);
#endif

    while (pk_ptr + SHAKE256_RATE <= pkm + hash_in_len) {
        absorb_hash_SHAKE256RATE(&state, pk_ptr);
        pk_ptr += SHAKE256_RATE;
        f1600x4(state.s, KeccakF_RoundConstants);
        extract_lanes_x3_shake128(buf[0], buf[1], buf[2], &state);
        uniform_x3(&state, mat, buf, rhoprhime, &cor);
    }
    absorb_hash_SHAKE256RATE_tail_process(&state, pk_ptr, tail_len);
    ST_AVX2_shake128x4_squeezeblocks(buf[0], buf[1], buf[2], buf[3], 1, &state);
    // memcpy(hash_out, buf[3], CRHBYTES);
    for (int i = 0; i < CRHBYTES; ++i) {
        hash_out[i] = buf[3][i];
    }
    uniform_x3(&state, mat, buf, rhoprhime, &cor);
    while (cor.ctr[0] < N || cor.ctr[1] < N || cor.ctr[2] < N) {
        shake128x3_squeezeblocks(buf[0], buf[1], buf[2], 1, &state);
        cor.ctr[0] += rej_uniform(mat[0].vec[3].coeffs + cor.ctr[0], N - cor.ctr[0], buf[0], SHAKE128_RATE);
        cor.ctr[1] += rej_uniform(mat[1].vec[0].coeffs + cor.ctr[1], N - cor.ctr[1], buf[1], SHAKE128_RATE);
        cor.ctr[2] += rej_uniform(mat[1].vec[1].coeffs + cor.ctr[2], N - cor.ctr[2], buf[2], SHAKE128_RATE);
    }

}

void shuffle_matkm(polyvecl mat[K]) {
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < M; ++j)
            poly_nttunpack(&mat[i].vec[j + 1]);
}
//
// Created by 59531 on 2025/11/29.
//

#ifndef HAETAE_HYBRID_H
#define HAETAE_HYBRID_H

#include "align.h"
#include "polyvec.h"
#include "sampler.h"
#include "symmetric.h"


#define GAUSS_RAND (72 + 16 + 48)
#define GAUSS_RAND_BYTES ((GAUSS_RAND + 7) / 8)
#define NUM_GAUSSIANS 278 // ca 80% require up to 272 Gaussian samples
#define POLY_HYPERBALL_BUFLEN_4X (GAUSS_RAND_BYTES * NUM_GAUSSIANS + N / 8)
#define POLY_HYPERBALL_NBLOCKS_4X ((POLY_HYPERBALL_BUFLEN_4X + STREAM256_BLOCKBYTES - 1) / STREAM256_BLOCKBYTES)
#define SHAK256_TAIL 64
#define LOOP_BUF_LENGTH (POLY_HYPERBALL_NBLOCKS_4X * STREAM256_BLOCKBYTES + SHAK256_TAIL)
#define LOOP_CAPABILITY (L + K + 3)
typedef struct {
    ALIGNED_UINT8(LOOP_BUF_LENGTH) buf[LOOP_CAPABILITY];
    int start;
    int size;
} loop_queue;


#define POLY_UNIFORM_ETA_NBLOCKS 1
#define LOOP_KG_BUF_LENGTH (POLY_UNIFORM_ETA_NBLOCKS * STREAM256_BLOCKBYTES + SHAK256_TAIL)
#define LOOP_KG_CAPABILITY (M + K + 4)
typedef struct {
    ALIGNED_UINT8(LOOP_KG_BUF_LENGTH) buf[LOOP_KG_CAPABILITY];
    int start;
    int size;
} loop_queue_kg;

uint8_t *loop_dequeue(loop_queue *loop);

uint8_t *loop_next(loop_queue *loop);

void print_loop_state(loop_queue *loop);

uint8_t *loop_kg_dequeue(loop_queue_kg *loop);

uint8_t *loop_kg_next(loop_queue_kg *loop);

void print_loop_kg_state(loop_queue_kg *loop);

#define GAUSS_N_BUF_LENGTH_4X (POLY_HYPERBALL_NBLOCKS_4X * STREAM256_BLOCKBYTES * 4)
typedef ALIGNED_UINT8(GAUSS_N_BUF_LENGTH_4X) guss_N_buf_x4;
void sample_gauss_N_Rnd_4x(loop_queue *loop, const uint8_t seed[CRHBYTES], uint16_t nonce0, uint16_t nonce1, uint16_t nonce2, uint16_t nonce3);
void sample_gauss_N_RejS_4x(loop_queue *loop,uint64_t *r0, uint64_t *r1, uint64_t *r2, uint64_t *r3,
    uint8_t *signs0, uint8_t *signs1, uint8_t *signs2, uint8_t *signs3,
    fp96_76 *sqsum,
    size_t len0, size_t len1, size_t len2, size_t len3);

void sample_gauss_N_RejS_3x(loop_queue *loop,uint64_t *r0, uint64_t *r1, uint64_t *r2,
    uint8_t *signs0, uint8_t *signs1, uint8_t *signs2,
    fp96_76 *sqsum,
    size_t len0, size_t len1, size_t len2);

void sample_gauss_N_RejS_2x(loop_queue *loop,uint64_t *r0, uint64_t *r1,
    uint8_t *signs0, uint8_t *signs1,
    fp96_76 *sqsum,
    size_t len0, size_t len1);

void sample_gauss_N_RejS_1x(loop_queue *loop,uint64_t *r0,
    uint8_t *signs0,
    fp96_76 *sqsum,
    size_t len0);

void poly_rand_eta_4x(loop_queue_kg *loop, const keccakx4_state64 *pre_state, uint16_t nonce0, uint16_t nonce1,
                           uint16_t nonce2, uint16_t nonce3);

void poly_rej_eta_4x(poly *a0, poly *a1, poly *a2, poly *a3,
                         loop_queue_kg *loop);

void poly_rej_eta_2x(poly *a0, poly *a1, loop_queue_kg *loop);

void poly_rej_eta(poly *a, loop_queue_kg *loop);

int hybrid_poly_uniform_2x_eta_2x(poly *a0, poly *a1, loop_queue_kg *loop,
                     const keccakx4_state32 *A_state, uint16_t Anonce0, uint16_t Anonce1,
                     const keccakx4_state64 *S_state, uint16_t Snonce);
void poly_uniform_4x_state(poly *a0, poly *a1, poly *a2, poly *a3,
                     const keccakx4_state32 *pre_state, uint16_t nonce0, uint16_t nonce1,
                     uint16_t nonce2, uint16_t nonce3);

void sample_gauss_RejS_N(loop_queue *loop,uint64_t *r, uint8_t *signs, fp96_76 *sqsum,
                    size_t len);

int polyvecmk_uniform_eta_loop(polyvecm *u, polyveck *v, loop_queue_kg *loop,
                           const keccakx4_state64 *pre_state, uint16_t nonce);

int hybrid_polymatkm_expand(polyvecm mat[K], const uint8_t rho[SEEDBYTES],
    const uint8_t sigma[CRHBYTES], uint16_t nonce, loop_queue_kg *loop);

int hybrid_polymatkm_and_k_expand_eta(polyvecm mat[K], polyveck *v, const keccakx4_state32 *pre_state, const keccakx4_state64 *S_state, uint16_t Snonce, loop_queue_kg *loop);

void hybrid_hash_pk_and_ExpandA(uint8_t *hash_out, const uint8_t *pkm, int hash_in_len, polyvecl mat[K], const keccakx4_state32 *rhoprhime);

void shuffle_matkm(polyvecl mat[K]);

void polymatkl_expand_part(polyvecl mat[K], const keccakx4_state32 *rho);

#endif //HAETAE_HYBRID_H
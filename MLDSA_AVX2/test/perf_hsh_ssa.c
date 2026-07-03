#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hybrid.h"
#include "packing.h"
#include "params.h"
#include "poly.h"
#include "rejsample.h"

#ifndef PERF_ITERS
#define PERF_ITERS 200000
#endif

static uint8_t hash_out[CRHBYTES] __attribute__((aligned(32)));
static uint8_t hash_in[CRHBYTES + 256] __attribute__((aligned(32)));
static uint8_t rho[SEEDBYTES] __attribute__((aligned(32)));
static uint8_t rhoprime[CRHBYTES] __attribute__((aligned(32)));
static uint8_t w1buf[K * POLYW1_PACKEDBYTES + CRHBYTES] __attribute__((aligned(32)));
static poly a0 __attribute__((aligned(32)));
static poly a1 __attribute__((aligned(32)));
static poly a2 __attribute__((aligned(32)));
static poly a3 __attribute__((aligned(32)));
static loop_queue loop __attribute__((aligned(32)));

static void init_inputs(void) {
    for (size_t i = 0; i < sizeof(hash_in); i++) hash_in[i] = (uint8_t)(i * 13u + 7u);
    for (size_t i = 0; i < sizeof(rho); i++) rho[i] = (uint8_t)(i * 17u + 3u);
    for (size_t i = 0; i < sizeof(rhoprime); i++) rhoprime[i] = (uint8_t)(i * 19u + 5u);
    for (size_t i = 0; i < sizeof(w1buf); i++) w1buf[i] = (uint8_t)(i * 23u + 11u);
}

static uint64_t checksum_poly(const poly *p) {
    uint64_t s = 0;
    for (size_t i = 0; i < N; i += 17) {
        s = (s << 7) ^ (s >> 3) ^ (uint32_t)p->coeffs[i];
    }
    return s;
}

static uint64_t checksum_loop(void) {
    uint64_t s = (uint64_t)loop.start ^ ((uint64_t)loop.size << 32);
    for (int i = 0; i < LOOP_SIZE; i++) {
        s ^= ((uint64_t)loop.buf[i].coeffs[(i * 29) % LOOP_BUF_LENGTH]) << ((i & 7) * 8);
    }
    return s;
}

static uint64_t checksum_bytes(const uint8_t *p, size_t n) {
    uint64_t s = 0;
    for (size_t i = 0; i < n; i += 13) {
        s = (s * 131u) ^ p[i];
    }
    return s;
}

static uint64_t bench_hsh_hash_expanda(int iters) {
    uint64_t s = 0;
    for (int i = 0; i < iters; i++) {
        hybrid_hash_ExpandA_shuffled(hash_out, CRHBYTES, hash_in, CRHBYTES + 59,
                                     &a0, &a1, &a2, rho,
                                     (uint16_t)(i + 0), (uint16_t)(i + 1), (uint16_t)(i + 2));
        s ^= checksum_bytes(hash_out, sizeof(hash_out));
        s ^= checksum_poly(&a0) ^ checksum_poly(&a1) ^ checksum_poly(&a2);
    }
    return s;
}

static uint64_t bench_hsh_uniform2_expandrand(int iters) {
    uint64_t s = 0;
    for (int i = 0; i < iters; i++) {
        loop.start = 0;
        loop.size = 0;
        hybrid_uniform_2x_and_ExpandRand(&a0, &a1, &loop, rho, rhoprime,
                                         (uint16_t)(256 + i), (uint16_t)(257 + i),
                                         (uint16_t)(i + 0), (uint16_t)(i + 1));
        s ^= checksum_poly(&a0) ^ checksum_poly(&a1) ^ checksum_loop();
    }
    return s;
}

static uint64_t bench_hsh_uniform3_expandrand(int iters) {
    uint64_t s = 0;
    for (int i = 0; i < iters; i++) {
        loop.start = 0;
        loop.size = 0;
        hybrid_uniform_3x_and_ExpandRand(&a0, &a1, &a2, &loop, rho, rhoprime,
                                         (uint16_t)(512 + i), (uint16_t)(513 + i),
                                         (uint16_t)(514 + i), (uint16_t)i);
        s ^= checksum_poly(&a0) ^ checksum_poly(&a1) ^ checksum_poly(&a2) ^ checksum_loop();
    }
    return s;
}

static uint64_t bench_ssa_challenge_expandrand(int iters) {
    uint64_t s = 0;
    for (int i = 0; i < iters; i++) {
        loop.start = 0;
        loop.size = 0;
        hybrid_challenge_and_ExpandRand_x3(hash_out, rhoprime, w1buf, &loop, rhoprime, (uint16_t)i);
        s ^= checksum_bytes(hash_out, CTILDEBYTES) ^ checksum_loop();
    }
    return s;
}

static uint64_t bench_nsh_expanda4(int iters) {
    uint64_t s = 0;
    for (int i = 0; i < iters; i++) {
        poly_uniform_4x_op13(&a0, &a1, &a2, &a3, rho,
                             (uint16_t)(768 + i), (uint16_t)(769 + i),
                             (uint16_t)(770 + i), (uint16_t)(771 + i));
        s ^= checksum_poly(&a0) ^ checksum_poly(&a1) ^ checksum_poly(&a2) ^ checksum_poly(&a3);
    }
    return s;
}

static uint64_t bench_nsh_expandrand4(int iters) {
    uint64_t s = 0;
    for (int i = 0; i < iters; i++) {
        loop.start = 0;
        loop.size = 0;
        poly_generate_random_gamma1_4x(&loop, rhoprime,
                                       (uint16_t)(i + 0), (uint16_t)(i + 1),
                                       (uint16_t)(i + 2), (uint16_t)(i + 3));
        s ^= checksum_loop();
    }
    return s;
}

int main(int argc, char **argv) {
    const char *which = argc > 1 ? argv[1] : "ssa_challenge_expandrand";
    int iters = argc > 2 ? atoi(argv[2]) : PERF_ITERS;
    uint64_t s;

    init_inputs();

    if (strcmp(which, "hsh_hash_expanda") == 0) {
        s = bench_hsh_hash_expanda(iters);
    } else if (strcmp(which, "hsh_uniform2_expandrand") == 0) {
        s = bench_hsh_uniform2_expandrand(iters);
    } else if (strcmp(which, "hsh_uniform3_expandrand") == 0) {
        s = bench_hsh_uniform3_expandrand(iters);
    } else if (strcmp(which, "ssa_challenge_expandrand") == 0) {
        s = bench_ssa_challenge_expandrand(iters);
    } else if (strcmp(which, "nsh_expanda4") == 0) {
        s = bench_nsh_expanda4(iters);
    } else if (strcmp(which, "nsh_expandrand4") == 0) {
        s = bench_nsh_expandrand4(iters);
    } else {
        fprintf(stderr, "unknown benchmark: %s\n", which);
        return 2;
    }

    printf("%s %d checksum=%llu\n", which, iters, (unsigned long long)s);
    return 0;
}

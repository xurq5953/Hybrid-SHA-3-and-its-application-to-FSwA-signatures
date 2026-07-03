#include <stdio.h>
#include <time.h>

#include "../include/randombytes.h"
#include "cpucycles.h"
#include "fft.h"
#include "ntt.h"
#include "params.h"
#include "polyfix.h"
#include "polymat.h"
#include "polyvec.h"
#include "sign.h"
#include "consts.h"
#include "fips202x4.h"
#include "speed_print.h"
#include "symmetric.h"

#define NTESTS 20000

uint64_t t[NTESTS];

int main() {
    uint8_t pk[CRYPTO_PUBLICKEYBYTES], byte;
    uint8_t sk[CRYPTO_SECRETKEYBYTES];
    uint8_t sig[CRYPTO_BYTES];
    uint8_t msg[SEEDBYTES * 2];
    size_t siglen;
    int i = 0;
    clock_t srt, ed;
    clock_t overhead;
    polyvecm mat[K];
    polyvecl matl[K] = {0};
    polyfixvecl y1;
    polyfixveck y2;
    polyvecl vecl;
    polyvecm s1;
    polyveck s2;
    poly *a = &mat[0].vec[0];
    poly *b = &mat[0].vec[1];
    poly *c = &mat[0].vec[2];
    poly_complex_fp32_16 fft_in;
    loop_queue loop;
    loop.start = 0;
    loop.size = 0;

    randombytes(msg, SEEDBYTES);
    overhead = clock();
    cpucycles();
    overhead = clock() - overhead;
    keccakx4_state state;
    ALIGNED_UINT8(CRHBYTES+2) buf[4];
    ALIGNED_UINT8(POLY_HYPERBALL_NBLOCKS_4X * STREAM256_BLOCKBYTES * 4) outbuf;
    uint64_t samples[N * (L + K)];
    fp96_76 sqsum, invsqrt;
    uint8_t signs[N * (L + K) / 8];

    #define REJ_UNIFORM_NBLOCKS                                                    \
    ((512 + STREAM128_BLOCKBYTES - 1) / STREAM128_BLOCKBYTES)
    #define REJ_UNIFORM_BUFLEN (REJ_UNIFORM_NBLOCKS * STREAM128_BLOCKBYTES)
    printf("POLY_HYPERBALL_NBLOCKS_4X: %d\n", POLY_HYPERBALL_NBLOCKS_4X);
    printf("POLYVECK_HIGHBITS_PACKEDBYTES + POLYC_PACKEDBYTES: %d\n", POLYVECK_HIGHBITS_PACKEDBYTES + POLYC_PACKEDBYTES);
    printf("REJ_UNIFORM_NBLOCKS: %d\n", REJ_UNIFORM_NBLOCKS);
    printf("POLYVECK_HIGHBITS_PACKEDBYTES + POLYC_PACKEDBYTES: %d\n", POLYVECK_HIGHBITS_PACKEDBYTES + POLYC_PACKEDBYTES);

    crypto_sign_keypair(pk, sk);
    poly r1,r2;
    rej_eta(r1.coeffs,N,pk,136);
    rej_eta_avx2(r2.coeffs,N,pk,136);

    for (int j = 0; j < N; ++j) {
        if (r1.coeffs[j] != r2.coeffs[j]) {
            printf("%d ",j);
        }
    }

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        polymatkm_expand(mat, msg);
    }
    print_results("polymatkm_expand:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        polymatkl_double(matl);
    }
    print_results("\npolymatkl_double:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        rej_eta(r1.coeffs,N,pk,136);
    }
    print_results("\nrej_eta:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        rej_eta_avx2(r2.coeffs,N,pk,136);
    }
    print_results("\nrej_eta_avx2:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        polyvecmk_uniform_eta(&s1, &s2, msg, i);
    }
    print_results("\npolyvecmk_uniform_eta:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        fft_init_and_bitrev(&fft_in, &s2.vec[i % K]);
        fft(&fft_in);
    }
    print_results("\nfft_bitrev + fft:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        ntt_avx(&a->vec[0], qdata.vec);
    }
    print_results("\nntt:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        polymatkm_pointwise_montgomery(&s2, mat, &s1);
    }
    print_results("\npolymatkm_pointwise_montgomery:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        polyfixvecl_round(&vecl, &y1);
        polyfixveck_round(&s2, &y2);
    }
    print_results("\npolyfixvecl_round + polyfixveck_round:", t, NTESTS);

    srt = clock();
    for (i = 0; i < NTESTS; i++) {
        t[i] = cpucycles();
        crypto_sign_keypair(pk, sk);
    }
    ed = clock();
    print_results("\ncrypto_sign_keypair: ", t, NTESTS);
    printf("time elapsed: %.8fms\n\n", (double)(ed - srt - overhead * NTESTS) *
                                           1000 / CLOCKS_PER_SEC / NTESTS);
    crypto_sign_keypair(pk, sk);
    srt = clock();
    for (i = 0; i < NTESTS; i++) {
        t[i] = cpucycles();
        crypto_sign_signature(sig, &siglen, sig, SEEDBYTES, sk);
    }
    ed = clock();
    print_results("crypto_sign_signature: ", t, NTESTS);
    printf("time elapsed: %.8fms\n\n", (double)(ed - srt - overhead * NTESTS) *
                                           1000 / CLOCKS_PER_SEC / NTESTS);



    srt = clock();
    size_t cnt = 0;
    while ((double)(clock() - srt) / CLOCKS_PER_SEC < 10) {
        crypto_sign_signature(sig, &siglen, sig, SEEDBYTES, sk);
        cnt += 1;
    }
    ed = clock();
    printf("crypto_sign_signature: %lu iterations ", cnt);
    printf("time elapsed: %.8fms\n\n", (double)(ed - srt - overhead * cnt) *
                                           1000 / CLOCKS_PER_SEC / cnt);



    crypto_sign_signature(sig, &siglen, msg, SEEDBYTES, sk);
    srt = clock();
    for (i = 0; i < NTESTS; i++) {
        t[i] = cpucycles();
        crypto_sign_verify(sig, siglen, msg, SEEDBYTES, pk);
    }
    ed = clock();
    print_results("crypto_sign_verify: ", t, NTESTS);
    printf("time elapsed: %.8fms\n\n", (double)(ed - srt - overhead * NTESTS) *
                                           1000 / CLOCKS_PER_SEC / NTESTS);
    return 0;
}

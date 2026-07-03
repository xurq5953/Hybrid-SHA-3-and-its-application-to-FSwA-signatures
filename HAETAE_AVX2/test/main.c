#include "decompose.h"
#include "fips202.h"
#include "packing.h"
#include "polyvec.h"
#include "randombytes.h"
#include "reduce.h"
#include "sign.h"
#include <stdio.h>
#include <string.h>

#define BUF_BYTES (1)
// #define Iterations (1000000)
#define Iterations (100)

uint8_t print_flag = 1;
uint8_t extended_flag = 0;

int sign_verify_test(void);
int extended_test(void);

int main(void) {
    printf("HAETAE mode = %d\n", HAETAE_MODE);
    size_t count = 1;
    for (int i = 0; i < Iterations; ++i) {
        if (sign_verify_test()) {
            printf("Invalid on %d-th test\n", i);
            break;
        }
        if(extended_flag && extended_test()) {
            printf("Extended %d-th test failed\n", i);
            break;            
        }

        if (!(i % (Iterations / 10))) {
            printf("...%lu%%", count * 10);
            fflush(stdout);
            ++count;
        }
    }
    printf("\n");

    return 0;
}

int sign_verify_test(void) {
    if (print_flag) {
        printf(">> sign and verify test\n");
        print_flag = 0;
    }

    uint8_t pk[CRYPTO_PUBLICKEYBYTES] = {0};
    uint8_t sk[CRYPTO_SECRETKEYBYTES] = {0};

    crypto_sign_keypair(pk, sk);

    size_t siglen = 0;
    uint8_t sig[CRYPTO_BYTES] = {0};
    uint8_t msg[SEEDBYTES] = {0};
    randombytes(msg, SEEDBYTES);
    crypto_sign_signature(sig, &siglen, msg, SEEDBYTES, sk);

    if (crypto_sign_verify(sig, siglen, msg, SEEDBYTES, pk)) {
        return 1;
    }

    return 0;
}

int extended_test(void) {
    uint8_t pk1[CRYPTO_PUBLICKEYBYTES] = {0};
    uint8_t sk1[CRYPTO_SECRETKEYBYTES] = {0};
    uint8_t pk2[CRYPTO_PUBLICKEYBYTES] = {0};
    uint8_t sk2[CRYPTO_SECRETKEYBYTES] = {0};

    crypto_sign_keypair(pk1, sk1);
    crypto_sign_keypair(pk2, sk2);

    size_t siglen1 = 0;
    uint8_t sig1[CRYPTO_BYTES] = {0};
    uint8_t msg1[SEEDBYTES] = {0};
    size_t siglen2 = 0;
    uint8_t sig2[CRYPTO_BYTES] = {0};
    uint8_t msg2[SEEDBYTES] = {0};

    randombytes(msg1, SEEDBYTES);
    crypto_sign_signature(sig1, &siglen1, msg1, SEEDBYTES, sk1);

    if (!crypto_sign_verify(sig1, siglen1, msg1, SEEDBYTES, pk2)) {
        printf("False positive: key missmatch\n");
        return 1; // exptect a failure due to sk - pk missmatch
    }

    randombytes(msg2, SEEDBYTES);

    if (!crypto_sign_verify(sig1, siglen1, msg2, SEEDBYTES, pk1)) {
        printf("False positive: msg missmatch\n");
        return 1; // exptect a failure due to sig - msg missmatch
    }

    uint8_t rand[2];
    uint16_t r_index = 0xFFFF;
    uint8_t r_bit = 1;
    while(r_index >= CRYPTO_BYTES) {
        randombytes(rand, 2);
        r_index = (((uint16_t) rand[1])<< 4) ^ (uint16_t) rand[0];
    }
    r_bit = 1 << (rand[1] & 0b111);

    memcpy(sig2, sig1, siglen1);
    sig2[r_index] ^= r_bit; // flip a random bit
    
    if (!crypto_sign_verify(sig2, siglen1, msg1, SEEDBYTES, pk1)) {
        printf(" False positive: sig tampering: index = %u, bit = %u\n", r_index, r_bit);
        return 1; // exptect a failure due to tampered signature
    }

    return 0;
}

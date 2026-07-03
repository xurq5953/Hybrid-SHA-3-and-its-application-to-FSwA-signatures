#include <stdint.h>
#include <stdio.h>

#include "hybrid.h"
#include "../sign.h"
#include "../poly.h"
#include "../polyvec.h"
#include "../params.h"
#include "cpucycles.h"
#include "fips202x8.h"
#include "packing.h"
#include "rejsample.h"
#include "speed_print.h"

#define NTESTS 20000

uint64_t t[NTESTS];


void print_keccak_instance_for_y(int rounds) {
  int P = 4;
  int loop_size;
  loop_size = (K * L -1) % P;
  // loop_size = 0;
  int count = 0;
  for (int i = 1; i <= rounds; ++i) {
    printf("rounds: %d ,loop_size :%d \n", i, loop_size);
    while (loop_size < L) {
      count ++;
      loop_size +=4;
    }
    loop_size -= L;
    loop_size +=3;
    printf("%d  %d\n", i, count);
  }
}

int main(void)
{
  size_t smlen;
  uint8_t pk[CRYPTO_PUBLICKEYBYTES];
  uint8_t sk[CRYPTO_SECRETKEYBYTES];
  uint8_t sm[CRYPTO_BYTES + CRHBYTES];
  __attribute__((aligned(32)))
  uint8_t seed[CRHBYTES] = {0};
  polyvecl mat[K];
  poly *a = &mat[0].vec[0];
  poly *b = &mat[0].vec[1];
  poly *c = &mat[0].vec[2];
  loop_queue loop;
  loop.start = 0;
  loop.size = 0;
  polyvecl y, z;
  keccakx8_state state;
    ALIGN(64) uint8_t buf[8][3360];

    printf("warm up********************************************\n");

    for (int i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        shake128x8_absorb(&state,buf[0], buf[1], buf[2], buf[3],buf[4], buf[5], buf[6], buf[7],10 *SHAKE128_RATE);
        keccak_squeezeblocks8x(buf[0], buf[1], buf[2], buf[3],buf[4], buf[5], buf[6], buf[7], 10, state.s,  SHAKE128_RATE);
    }
    print_results("SHAKE128x8:", t, NTESTS);

    printf("SHAKE256 Comparison********************************************\n");
    for (int round = 1; round < 21; ++round) {
        printf("round :%d\n", round);

        for (int i = 0; i < NTESTS; ++i) {
            t[i] = cpucycles();
            shake256x8_absorb(&state,buf[0], buf[1], buf[2], buf[3],buf[4], buf[5], buf[6], buf[7],round *SHAKE256_RATE);
            keccak_squeezeblocks8x(buf[0], buf[1], buf[2], buf[3],buf[4], buf[5], buf[6], buf[7], round, state.s,  SHAKE256_RATE);
        }
        print_results("SHAKE2568x:", t, NTESTS);
        printf("buf%d \n", buf[0][0]);

        for (int i = 0; i < NTESTS; ++i) {
            t[i] = cpucycles();
            XURQ_AVX512_shake256x8_absorb(&state, buf[0], buf[1], buf[2], buf[3],buf[4], buf[5], buf[6], buf[7], round * SHAKE256_RATE);
            XURQ_AVX512_shake256x8_squeezeblocks(&state, buf[0], buf[1], buf[2], buf[3],buf[4], buf[5], buf[6], buf[7], round);
        }
        print_results("XURQ_AVX512_shake256x8:", t, NTESTS);
        printf("buf%d \n", buf[0][0]);


    }


    printf("SHAKE128 Comparison********************************************\n");
    for (int round = 1; round < 21; ++round) {
        printf("round :%d\n", round);

        for (int i = 0; i < NTESTS; ++i) {
            t[i] = cpucycles();
            shake128x8_absorb(&state,buf[0], buf[1], buf[2], buf[3],buf[4], buf[5], buf[6], buf[7],round *SHAKE128_RATE);
            keccak_squeezeblocks8x(buf[0], buf[1], buf[2], buf[3],buf[4], buf[5], buf[6], buf[7], round, state.s,  SHAKE128_RATE);
        }
        print_results("SHAKE128x8:", t, NTESTS);
        printf("buf%d \n", buf[0][0]);

        for (int i = 0; i < NTESTS; ++i) {
            t[i] = cpucycles();
            XURQ_AVX512_shake128x8_absorb(&state, buf[0], buf[1], buf[2], buf[3],buf[4], buf[5], buf[6], buf[7], round * SHAKE128_RATE);
            XURQ_AVX512_shake128x8_squeezeblocks(&state, buf[0], buf[1], buf[2], buf[3],buf[4], buf[5], buf[6], buf[7], round);
        }
        print_results("XURQ_AVX512_shake128x8_squeezeblocks:", t, NTESTS);
        printf("buf%d \n", buf[0][0]);

        printf("\n");
    }

  return 0;
}

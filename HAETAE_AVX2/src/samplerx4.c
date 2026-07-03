#include "consts.h"
#include "sampler.h"
#include "fixpoint.h"
#include "symmetric.h"
#include "fips202x4.h"
#include <stdint.h>
#include <immintrin.h>
#include <stdio.h>
#include <string.h>

#include "hybrid.h"


#define CDTLEN 64
static const union {
    __m256i vec[(CDTLEN+7)/8 * 4];
    uint32_t arr[CDTLEN*4];
} CDT_avx = {
    .arr = {
 3266,  6520,  3266,  6520,  3266,  6520,  3266,  6520, 
 9748, 12938,  9748, 12938,  9748, 12938,  9748, 12938, 
16079, 19159, 16079, 19159, 16079, 19159, 16079, 19159, 
22168, 25096, 22168, 25096, 22168, 25096, 22168, 25096, 
27934, 30674, 27934, 30674, 27934, 30674, 27934, 30674, 
33309, 35833, 33309, 35833, 33309, 35833, 33309, 35833, 
38241, 40531, 38241, 40531, 38241, 40531, 38241, 40531, 
42698, 44742, 42698, 44742, 42698, 44742, 42698, 44742, 
46663, 48460, 46663, 48460, 46663, 48460, 46663, 48460, 
50135, 51690, 50135, 51690, 50135, 51690, 50135, 51690, 
53128, 54454, 53128, 54454, 53128, 54454, 53128, 54454, 
55670, 56781, 55670, 56781, 55670, 56781, 55670, 56781, 
57794, 58712, 57794, 58712, 57794, 58712, 57794, 58712, 
59541, 60287, 59541, 60287, 59541, 60287, 59541, 60287, 
60956, 61554, 60956, 61554, 60956, 61554, 60956, 61554, 
62085, 62556, 62085, 62556, 62085, 62556, 62085, 62556, 
62972, 63337, 62972, 63337, 62972, 63337, 62972, 63337, 
63657, 63936, 63657, 63936, 63657, 63936, 63657, 63936, 
64178, 64388, 64178, 64388, 64178, 64388, 64178, 64388, 
64569, 64724, 64569, 64724, 64569, 64724, 64569, 64724, 
64857, 64970, 64857, 64970, 64857, 64970, 64857, 64970, 
65066, 65148, 65066, 65148, 65066, 65148, 65066, 65148, 
65216, 65273, 65216, 65273, 65216, 65273, 65216, 65273, 
65321, 65361, 65321, 65361, 65321, 65361, 65321, 65361, 
65394, 65422, 65394, 65422, 65394, 65422, 65394, 65422, 
65444, 65463, 65444, 65463, 65444, 65463, 65444, 65463, 
65478, 65490, 65478, 65490, 65478, 65490, 65478, 65490, 
65500, 65508, 65500, 65508, 65500, 65508, 65500, 65508, 
65514, 65519, 65514, 65519, 65514, 65519, 65514, 65519, 
65523, 65527, 65523, 65527, 65523, 65527, 65523, 65527, 
65529, 65531, 65529, 65531, 65529, 65531, 65529, 65531, 
65533, 65534, 65533, 65534, 65533, 65534, 65533, 65534
    }
};

static const union {
    __m256i vec;
    uint32_t arr[8];
} zero_avx = { .arr = {0} };

static const union {
    __m256i vec[2];
    uint16_t arr[32];
} mask_avx = { .arr = {
  0xffff, 0, 0, 0, 0xffff, 0, 0, 0, 0xffff, 0, 0, 0, 0xffff, 0, 0, 0,
  0, 0, 0xffff, 0, 0, 0, 0xffff, 0, 0, 0, 0xffff, 0, 0, 0, 0xffff, 0
  }};



static const union {
    __m256i vec[3];
    uint64_t arr[12];
} mask72_avx = { .arr = {
  (1ULL<<48)-1, (1ULL<<48)-1, (1ULL<<48)-1, (1ULL<<48)-1, 
  (1ULL<<24)-1, (1ULL<<24)-1, (1ULL<<24)-1, (1ULL<<24)-1, 
  (1ULL<<16)-1, (1ULL<<16)-1, (1ULL<<16)-1, (1ULL<<16)-1
  }};

static const union {
    __m256i vec[3];
    uint64_t arr[12];
} consts_avx = { .arr = {
  1<<14, 1<<14, 1<<14, 1<<14, 
  1<<11, 1<<11, 1<<11, 1<<11, 
  1,1,1,1
  }};


//static void preprocessing_x4(sample_candidates.vec, sqr.vec, rejection.vec, &outbuf.vec[4]); // first four vectors are just signs
static void preprocessing_x4(__m256i *sc, __m256i *sqr, __m256i *rej, const __m256i *buf)
{
  size_t i,j;
  __m256i reg[16];

  reg[7] = _mm256_load_si256(&mask72_avx.vec[0]); // 48 bit mask
  reg[8] = _mm256_load_si256(&mask72_avx.vec[1]); // 24 bit mask
  reg[9] = _mm256_load_si256(&mask72_avx.vec[2]); // 16 bit mask

  j = 0;
  for (i = 0; i < NUM_GAUSSIANS; i++)
  {
    reg[0] = _mm256_load_si256(&buf[j++]);
    reg[1] = _mm256_load_si256(&buf[j++]);
    reg[2] = _mm256_load_si256(&buf[j]);

    // reg[3] = cdt out
    // reg[4] = rej out
    // reg[5:6] = fill out

    // c = cdt randomness byte
    // r = rejection randomness byte
    // f = filling randomness byte
    switch (i%8)
    {
      case 0:
        // reg[0]: rrrrrrcc
        // reg[1]: ffffffff
        // reg[2]: xxxxxxxf
        reg[3] = _mm256_and_si256(reg[0], reg[9]); // mask 16 bit
        reg[4] = _mm256_srli_epi64(reg[0], 16);
        reg[5] = _mm256_and_si256(reg[1], reg[7]); // mask 48 bit
        reg[6] = _mm256_srli_epi64(reg[1], 48);
        reg[10] = _mm256_slli_epi64(reg[2], 16);
        reg[6] = _mm256_or_si256(reg[6], reg[10]);
        reg[6] = _mm256_and_si256(reg[6], reg[8]); // mask 24 bit
        break;

      case 1:
        // reg[0]: rrrrrccx
        // reg[1]: fffffffr
        // reg[2]: xxxxxxff
        reg[3] = _mm256_srli_epi64(reg[0], 8);
        reg[3] = _mm256_and_si256(reg[3], reg[9]); // mask 16 bit
        reg[4] = _mm256_srli_epi64(reg[0], 24);
        reg[10] = _mm256_slli_epi64(reg[1], 40);
        reg[4] = _mm256_or_si256(reg[4], reg[10]);
        reg[4] = _mm256_and_si256(reg[4], reg[7]); // mask 48 bit
        reg[5] = _mm256_srli_epi64(reg[1], 8);
        reg[5] = _mm256_and_si256(reg[5], reg[7]); // mask 48 bit
        reg[6] = _mm256_srli_epi64(reg[1], 56);
        reg[10] = _mm256_slli_epi64(reg[2], 8);
        reg[6] = _mm256_or_si256(reg[6], reg[10]);
        reg[6] = _mm256_and_si256(reg[6], reg[8]); // mask 24 bit
        break;

      case 2:
        // reg[0]: rrrrccxx
        // reg[1]: ffffffrr
        // reg[2]: xxxxxfff
        reg[3] = _mm256_srli_epi64(reg[0], 16);
        reg[3] = _mm256_and_si256(reg[3], reg[9]); // mask 16 bit
        reg[4] = _mm256_srli_epi64(reg[0], 32);
        reg[10] = _mm256_slli_epi64(reg[1], 32);
        reg[4] = _mm256_or_si256(reg[4], reg[10]);
        reg[4] = _mm256_and_si256(reg[4], reg[7]); // mask 48 bit
        reg[5] = _mm256_srli_epi64(reg[1], 16);
        reg[6] = _mm256_and_si256(reg[2], reg[8]); // mask 24 bit
        break;

      case 3:
        // reg[0]: rrrccxxx
        // reg[1]: fffffrrr
        // reg[2]: xxxxffff
        reg[3] = _mm256_srli_epi64(reg[0], 24);
        reg[3] = _mm256_and_si256(reg[3], reg[9]); // mask 16 bit
        reg[4] = _mm256_srli_epi64(reg[0], 40);
        reg[10] = _mm256_slli_epi64(reg[1], 24);
        reg[4] = _mm256_or_si256(reg[4], reg[10]);
        reg[4] = _mm256_and_si256(reg[4], reg[7]); // mask 48 bit
        reg[5] = _mm256_srli_epi64(reg[1], 24);
        reg[10] = _mm256_slli_epi64(reg[2], 40);
        reg[5] = _mm256_or_si256(reg[5], reg[10]);
        reg[5] = _mm256_and_si256(reg[5], reg[7]); // mask 48 bit
        reg[6] = _mm256_srli_epi64(reg[2], 8);
        reg[6] = _mm256_and_si256(reg[6], reg[8]); // mask 24 bit
        break;

      case 4:
        // reg[0]: rrccxxxx
        // reg[1]: ffffrrrr
        // reg[2]: xxxfffff
        reg[3] = _mm256_srli_epi64(reg[0], 32);
        reg[3] = _mm256_and_si256(reg[3], reg[9]); // mask 16 bit
        reg[4] = _mm256_srli_epi64(reg[0], 48);
        reg[10] = _mm256_slli_epi64(reg[1], 16);
        reg[4] = _mm256_or_si256(reg[4], reg[10]);
        reg[4] = _mm256_and_si256(reg[4], reg[7]); // mask 48 bit
        reg[5] = _mm256_srli_epi64(reg[1], 32);
        reg[10] = _mm256_slli_epi64(reg[2], 32);
        reg[5] = _mm256_or_si256(reg[5], reg[10]);
        reg[5] = _mm256_and_si256(reg[5], reg[7]); // mask 48 bit
        reg[6] = _mm256_srli_epi64(reg[2], 16);
        reg[6] = _mm256_and_si256(reg[6], reg[8]); // mask 24 bit
        break;

      case 5:
        // reg[0]: rccxxxxx
        // reg[1]: fffrrrrr
        // reg[2]: xxffffff
        reg[3] = _mm256_srli_epi64(reg[0], 40);
        reg[3] = _mm256_and_si256(reg[3], reg[9]); // mask 16 bit
        reg[4] = _mm256_srli_epi64(reg[0], 56);
        reg[10] = _mm256_slli_epi64(reg[1], 8);
        reg[4] = _mm256_or_si256(reg[4], reg[10]);
        reg[4] = _mm256_and_si256(reg[4], reg[7]); // mask 48 bit
        reg[5] = _mm256_srli_epi64(reg[1], 40);
        reg[10] = _mm256_slli_epi64(reg[2], 24);
        reg[5] = _mm256_or_si256(reg[5], reg[10]);
        reg[5] = _mm256_and_si256(reg[5], reg[7]); // mask 48 bit
        reg[6] = _mm256_srli_epi64(reg[2], 24);
        reg[6] = _mm256_and_si256(reg[6], reg[8]); // mask 24 bit
        break;

      case 6:
        // reg[0]: ccxxxxxx
        // reg[1]: ffrrrrrr
        // reg[2]: xfffffff
        reg[3] = _mm256_srli_epi64(reg[0], 48);
        reg[3] = _mm256_and_si256(reg[3], reg[9]); // mask 16 bit
        reg[4] = _mm256_and_si256(reg[1], reg[7]); // mask 48 bit
        reg[5] = _mm256_srli_epi64(reg[1], 48);
        reg[10] = _mm256_slli_epi64(reg[2], 16);
        reg[5] = _mm256_or_si256(reg[5], reg[10]);
        reg[5] = _mm256_and_si256(reg[5], reg[7]); // mask 48 bit
        reg[6] = _mm256_srli_epi64(reg[2], 32);
        reg[6] = _mm256_and_si256(reg[6], reg[8]); // mask 24 bit
        break;

      default:
      case 7:
        // reg[0]: cxxxxxxx
        // reg[1]: frrrrrrc
        // reg[2]: ffffffff
        reg[3] = _mm256_srli_epi64(reg[0], 56);
        reg[10] = _mm256_slli_epi64(reg[1], 8);
        reg[3] = _mm256_or_si256(reg[3], reg[10]);
        reg[3] = _mm256_and_si256(reg[3], reg[9]); // mask 16 bit
        reg[4] = _mm256_srli_epi64(reg[1], 8);
        reg[4] = _mm256_and_si256(reg[4], reg[7]); // mask 48 bit
        reg[5] = _mm256_srli_epi64(reg[1], 56);
        reg[10] = _mm256_slli_epi64(reg[2], 8);
        reg[5] = _mm256_or_si256(reg[5], reg[10]);
        reg[5] = _mm256_and_si256(reg[5], reg[7]); // mask 48 bit
        reg[6] = _mm256_srli_epi64(reg[2], 40);
        reg[6] = _mm256_and_si256(reg[6], reg[8]); // mask 24 bit
        j += 1;
        break;
    }
    // duplicate CDT randomness
    reg[10] = _mm256_slli_epi64(reg[3], 32);
    reg[3] = _mm256_or_si256(reg[3], reg[10]);

    _mm256_store_si256(&sc[i], reg[3]);
    _mm256_store_si256(&rej[i], reg[4]);
    _mm256_store_si256(&sqr[2*i+0], reg[5]);
    _mm256_store_si256(&sqr[2*i+1], reg[6]);
  }
}

// samples 4x NUM_GAUSSIANS derivates from the CDT distribution
static void sample_gauss16_x4(__m256i *r) {
    size_t i,j;
    __m256i sum[NUM_GAUSSIANS];
    __m256i reg[16];

    reg[0]  = _mm256_load_si256(&mask_avx.vec[0]);
    reg[1]  = _mm256_load_si256(&mask_avx.vec[1]);
    reg[15]  = _mm256_load_si256(&zero_avx.vec);
    

    // first part: load 11 comparison values, compare everything
    for (i = 0; i < 11; i++) {
      reg[i] = _mm256_load_si256(&CDT_avx.vec[i]);
    }
    reg[11] = _mm256_load_si256(&zero_avx.vec);

    // iterate over whole input buffer
    for (i = 0; i < NUM_GAUSSIANS; i++) {
      // load
      reg[12] = _mm256_load_si256(&r[i]);

      // compare with redundant CDT
      reg[13] = _mm256_cmpgt_epi32(reg[12], reg[0]);
      for (j = 1; j < 11; j++) {
          reg[14] = _mm256_cmpgt_epi32(reg[12], reg[j]);
          reg[13] = _mm256_add_epi32(reg[14], reg[13]);
      }

      // store
      _mm256_store_si256(&sum[i], reg[13]);
    }

    // second part: load 11 comparison values, compare everything
    for (i = 0; i < 11; i++) {
      reg[i] = _mm256_load_si256(&CDT_avx.vec[i+11]);
    }

    // iterate over whole input buffer
    for (i = 0; i < NUM_GAUSSIANS; i++) {
      // load
      reg[12] = _mm256_load_si256(&r[i]);
      reg[13] = _mm256_load_si256(&sum[i]); // load intermediate sum

      // compare with redundant CDT
      for (j = 0; j < 11; j++) {
          reg[14] = _mm256_cmpgt_epi32(reg[12], reg[j]);
          reg[13] = _mm256_add_epi32(reg[14], reg[13]);
      }

      // store
      _mm256_store_si256(&sum[i], reg[13]);
    }





    // third part: load 10 comparison values, compare everything
    for (i = 0; i < 10; i++) {
      reg[i] = _mm256_load_si256(&CDT_avx.vec[i+22]);
    }

    // iterate over whole input buffer
    for (i = 0; i < NUM_GAUSSIANS; i++) {
      // load
      reg[12] = _mm256_load_si256(&r[i]);
      reg[13] = _mm256_load_si256(&sum[i]); // load intermediate sum

      // compare with redundant CDT
      for (j = 0; j < 10; j++) {
          reg[14] = _mm256_cmpgt_epi32(reg[12], reg[j]);
          reg[13] = _mm256_add_epi32(reg[14], reg[13]);
      }

      // sum up both intermediates
      reg[14] = _mm256_srli_epi64(reg[13], 32);
      reg[13] = _mm256_add_epi32(reg[13], reg[14]);

      // map to positive
      reg[13] = _mm256_sub_epi32(reg[15], reg[13]);

      // only select the lower 32 bit of each 64-bit value (remove redundancy)
      reg[13] = _mm256_blend_epi32(reg[13], reg[11], 0xaa);

      // store
      _mm256_store_si256(&r[i], reg[13]);
    }
}

static void sample_candidates_x4(__m256i *sc, __m256i *sqr, __m256i *rej) {
  size_t i;
  __m256i reg[16];

  reg[7] = _mm256_load_si256(&mask72_avx.vec[0]);
  reg[8] = _mm256_load_si256(&mask72_avx.vec[1]);
  reg[13] = _mm256_load_si256(&consts_avx.vec[0]);
  reg[14] = _mm256_load_si256(&consts_avx.vec[1]);
  reg[15] = _mm256_load_si256(&consts_avx.vec[2]);

  for (i = 0; i < NUM_GAUSSIANS; i++) {
    //temporary registers: reg[6,9:12]
    
    // copy sc into register (contains CDT samples)
    reg[0] = _mm256_load_si256(&sc[i]);

    // construct precise candidate (load 72-bit randomness from buf)
    //reg[1:2]
    reg[1] = _mm256_load_si256(&sqr[2*i+0]);
    reg[2] = _mm256_load_si256(&sqr[2*i+1]);
    reg[6] = _mm256_slli_epi64(reg[0], 24);
    reg[2] = _mm256_or_si256(reg[2], reg[6]);

    // square candidate
    //reg[3:4]
    // (a[31:0] + 2^32*a[47:32] + 2^48*a[78:48])^2
    // = a[31:0]^2  +  2 * 2^32a[31:0]a[47:32]  +  2 * 2^48a[31:0]a[78:48]  +  2^64a[47:32]^2  +  2 * 2^80a[47:32]a[78:48]  +  2^96a[78:48]^2
    reg[6] = _mm256_srli_epi64(reg[1], 32); // a[47:32]
    reg[3] = _mm256_mul_epu32(reg[1], reg[1]); // a[31:0]^2
    reg[4] = _mm256_mul_epu32(reg[1], reg[6]); // a[31:0]a[47:32]
    reg[9] = _mm256_mul_epu32(reg[1], reg[2]); // a[31:0]a[78:48]
    reg[10] = _mm256_mul_epu32(reg[6], reg[6]); // a[47:32]^2
    reg[11] = _mm256_mul_epu32(reg[6], reg[2]); // a[47:32]a[78:48]
    reg[12] = _mm256_mul_epu32(reg[2], reg[2]); // a[78:48]^2

    reg[3] = _mm256_srli_epi64(reg[3], 33); // a[31:0]^2 >> 33
    reg[3] = _mm256_add_epi64(reg[3], reg[4]); // (a[31:0]^2  +  2 * 2^32a[31:0]a[47:32]) >> 33
    //reg[3] = _mm256_add_epi64(reg[3], reg[13]); // add 1<<14
    reg[3] = _mm256_srli_epi64(reg[3], 16); // (a[31:0]^2  +  2 * 2^32a[31:0]a[47:32]) >> 49
    reg[3] = _mm256_add_epi64(reg[3], reg[9]); // (a[31:0]^2  +  2 * 2^32a[31:0]a[47:32]  +  2 * 2^48a[31:0]a[78:48]) >> 49
    reg[3] = _mm256_srli_epi64(reg[3], 15);
    reg[3] = _mm256_add_epi64(reg[3], reg[10]); // (a[31:0]^2  +  2 * 2^32a[31:0]a[47:32]  +  2 * 2^48a[31:0]a[78:48]  +  2^64a[47:32]^2) >> 64
    //reg[3] = _mm256_add_epi64(reg[3], reg[14]); // add 1<<11
    reg[3] = _mm256_srli_epi64(reg[3], 12); // (a[31:0]^2  +  2 * 2^32a[31:0]a[47:32]  +  2 * 2^48a[31:0]a[78:48]  +  2^64a[47:32]^2) >> 76
    reg[6] = _mm256_slli_epi64(reg[11], 5); // a[47:32]a[78:48] << 5
    reg[9] = _mm256_slli_epi64(reg[12], 20); // a[78:48]^2 << 20
    reg[4] = _mm256_srli_epi64(reg[12], 28); // a[78:48]^2 >> 28 --- the 36 high bits of reg[12]
    reg[9] = _mm256_and_si256(reg[9], reg[7]); // (a[78:48]^2 << 20) & ((1UL<<48)-1) --- the 28 low bits of reg[12]

    reg[3] = _mm256_add_epi64(reg[6], reg[3]); // LOW[(a[31:0]^2  +  2 * 2^32a[31:0]a[47:32]  +  2 * 2^48a[31:0]a[78:48]  +  2^64a[47:32]^2  +  2 * 2^80a[47:32]a[78:48]) >> 76]
    reg[3] = _mm256_add_epi64(reg[9], reg[3]); // LOW[(a[31:0]^2  +  2 * 2^32a[31:0]a[47:32]  +  2 * 2^48a[31:0]a[78:48]  +  2^64a[47:32]^2  +  2 * 2^80a[47:32]a[78:48]  +  2^96a[78:48]^2) >> 76]
    reg[6] = _mm256_srli_epi64(reg[3], 48); // copy upper bits
    reg[4] = _mm256_add_epi64(reg[4], reg[6]); // HIGH[(a[31:0]^2  +  2 * 2^32a[31:0]a[47:32]  +  2 * 2^48a[31:0]a[78:48]  +  2^64a[47:32]^2  +  2 * 2^80a[47:32]a[78:48]  +  2^96a[78:48]^2) >> 76]
    reg[3] = _mm256_and_si256(reg[3], reg[7]); // mask with (1ULL<<48)-1

    // round the candidate
    //reg[1]
    reg[1] = _mm256_srli_epi64(reg[1], 15);
    reg[2] = _mm256_slli_epi64(reg[2], 33);
    reg[1] = _mm256_or_si256(reg[1], reg[2]);
    reg[1] = _mm256_add_epi64(reg[1], reg[15]); // add 1
    reg[1] = _mm256_srli_epi64(reg[1], 1);

    // construct rejection value (exp_in := sqr - ((x*x) << 68))
    //reg[5]
    reg[0] = _mm256_mullo_epi16(reg[0], reg[0]); // this is faster, and the sc value is <= 64
    reg[6] = _mm256_slli_epi64(reg[0], 68-48);
    reg[5] = _mm256_sub_epi64(reg[4], reg[6]);
    reg[5] = _mm256_slli_epi64(reg[5], 20);
    reg[9] = _mm256_srli_epi64(reg[3], 28);
    reg[5] = _mm256_or_si256(reg[5], reg[9]);
    reg[5] = _mm256_add_epi64(reg[5], reg[15]); // add 1
    reg[5] = _mm256_srli_epi64(reg[5], 1);

    // write precise square to sqr
    _mm256_store_si256(&sqr[2*i+0], reg[3]);
    _mm256_store_si256(&sqr[2*i+1], reg[4]);

    // write rejection value to rej
    _mm256_store_si256(&rej[i], reg[5]);

    // write rounded candidate to sc
    _mm256_store_si256(&sc[i], reg[1]);
  }
}

static inline void smulh48_avx(
    __m256i *a, const __m256i *b, 
    __m256i *ah,  __m256i *bh, 
    __m256i *al,  __m256i *bl, 
    __m256i *tmp, __m256i *tmp2,
    const __m256i *mask24,
    const __m256i *one23) {
  *ah = _mm256_srli_epi64(*a, 24);
  *tmp = _mm256_srli_epi64(*ah, 24); // obtain 16 sign bits
  *tmp2 = _mm256_slli_epi64(*tmp, 40); // shift up, MSByte has still no sign bits
  *tmp = _mm256_slli_epi64(*tmp, 48); // shift up incl MSByte
  *ah = _mm256_or_si256(*ah, *tmp2); // set sign bits
  *ah = _mm256_or_si256(*ah, *tmp); // set sign bits
  *tmp = _mm256_slli_epi64(*ah, 24);
  *al = _mm256_sub_epi64(*a, *tmp);
  *bl = _mm256_and_si256(*b, *mask24);
  *bh = _mm256_srli_epi64(*b, 24);
  
  *a = _mm256_mul_epi32(*al, *bl);
  *tmp = _mm256_mul_epi32(*al, *bh); // last usage of al
  *tmp2 = _mm256_mul_epi32(*ah, *bl); // last usage of bl
  *tmp = _mm256_add_epi64(*tmp2, *tmp);
  *tmp = _mm256_add_epi64(*one23, *tmp); // rounding
  *a = _mm256_srli_epi64(*a, 24);
  *bl = _mm256_srli_epi64(*a, 24); // now we only have 16 sign bits left as LSBs
  *al = _mm256_slli_epi64(*bl, 40); // shift sign bits up, but the MSByte still has no sign bits
  *bl = _mm256_slli_epi64(*bl, 48); // shift sign bits up, including the MSByte
  *a = _mm256_or_si256(*a, *al); // set sign bits
  *a = _mm256_or_si256(*a, *bl); // set sign bits
  *a = _mm256_add_epi64(*a, *tmp);
  
  *a = _mm256_srli_epi64(*a, 24);
  *tmp = _mm256_srli_epi64(*a, 32); // now we only have 8 sign bits left as LSBs
  *tmp2 = _mm256_slli_epi64(*tmp, 40); // shift sign bits up 
  *a = _mm256_or_si256(*a, *tmp2); // set sign bits for one byte
  *tmp2 = _mm256_slli_epi64(*tmp, 48); // shift sign bits up
  *a = _mm256_or_si256(*a, *tmp2); // set sign bits for next byte
  *tmp2 = _mm256_slli_epi64(*tmp, 56); // shift sign bits up
  *a = _mm256_or_si256(*a, *tmp2); // set sign bits for next byte
  *tmp = _mm256_mul_epi32(*ah, *bh);
  *a = _mm256_add_epi64(*a, *tmp);
}

static const union {
  __m256i vec[9];
  int64_t arr[9*4];
} exp_avx = {
  .arr = {
    -0x0000B6C6340925AELL, -0x0000B6C6340925AELL, -0x0000B6C6340925AELL, -0x0000B6C6340925AELL, 
     0x0000B4BD4DF85227LL,  0x0000B4BD4DF85227LL,  0x0000B4BD4DF85227LL,  0x0000B4BD4DF85227LL, 
    -0x0000887F727491E2LL, -0x0000887F727491E2LL, -0x0000887F727491E2LL, -0x0000887F727491E2LL, 
     0x0000AAAA643C7E8DLL,  0x0000AAAA643C7E8DLL,  0x0000AAAA643C7E8DLL,  0x0000AAAA643C7E8DLL, 
    -0x0000AAAAA98179E6LL, -0x0000AAAAA98179E6LL, -0x0000AAAAA98179E6LL, -0x0000AAAAA98179E6LL, 
     0x0000FFFFFFFB2E7ALL,  0x0000FFFFFFFB2E7ALL,  0x0000FFFFFFFB2E7ALL,  0x0000FFFFFFFB2E7ALL, 
    -0x0000FFFFFFFFF85FLL, -0x0000FFFFFFFFF85FLL, -0x0000FFFFFFFFF85FLL, -0x0000FFFFFFFFF85FLL, 
     0x0000FFFFFFFFFFFCLL,  0x0000FFFFFFFFFFFCLL,  0x0000FFFFFFFFFFFCLL,  0x0000FFFFFFFFFFFCLL,
     1<<23, 1<<23, 1<<23, 1<<23
  }
};

static void approx_exp_4x(__m256i *rej) {
    __m256i reg[16];
    size_t i;
    reg[1] = _mm256_load_si256(&consts_avx.vec[2]); // 1
    reg[2] = _mm256_load_si256(&mask72_avx.vec[1]); // (1<<24)-1
    reg[3] = _mm256_load_si256(&exp_avx.vec[8]); // 1<<23

    reg[11] = _mm256_load_si256(&exp_avx.vec[1]);

    for (i = 0; i < NUM_GAUSSIANS; i++) 
    {
      reg[0] = _mm256_load_si256(&exp_avx.vec[0]);
      reg[4] = _mm256_load_si256(&rej[i]);

      reg[12] = _mm256_load_si256(&exp_avx.vec[2]);
      reg[13] = _mm256_load_si256(&exp_avx.vec[3]);
      reg[14] = _mm256_load_si256(&exp_avx.vec[4]);

      reg[1] = _mm256_slli_epi64(reg[1], 2);

      smulh48_avx(&reg[0], &reg[4], &reg[5], &reg[6], &reg[7], &reg[8], &reg[9], &reg[10], &reg[2], &reg[3]);
      reg[0] = _mm256_add_epi64(reg[0], reg[1]);
      reg[0] = _mm256_srli_epi64(reg[0], 3);
      reg[0] = _mm256_add_epi64(reg[0], reg[11]);

      smulh48_avx(&reg[0], &reg[4], &reg[5], &reg[6], &reg[7], &reg[8], &reg[9], &reg[10], &reg[2], &reg[3]);
      reg[0] = _mm256_add_epi64(reg[0], reg[1]);
      reg[0] = _mm256_srli_epi64(reg[0], 3);
      reg[0] = _mm256_add_epi64(reg[0], reg[12]);

      reg[1] = _mm256_srli_epi64(reg[1], 1);

      smulh48_avx(&reg[0], &reg[4], &reg[5], &reg[6], &reg[7], &reg[8], &reg[9], &reg[10], &reg[2], &reg[3]);
      reg[0] = _mm256_add_epi64(reg[0], reg[1]);
      reg[0] = _mm256_srli_epi64(reg[0], 2);
      reg[0] = _mm256_add_epi64(reg[0], reg[13]);

      smulh48_avx(&reg[0], &reg[4], &reg[5], &reg[6], &reg[7], &reg[8], &reg[9], &reg[10], &reg[2], &reg[3]);
      reg[0] = _mm256_add_epi64(reg[0], reg[1]);
      reg[0] = _mm256_srli_epi64(reg[0], 2);
      reg[0] = _mm256_add_epi64(reg[0], reg[14]);

      reg[12] = _mm256_load_si256(&exp_avx.vec[5]);
      reg[13] = _mm256_load_si256(&exp_avx.vec[6]);
      reg[14] = _mm256_load_si256(&exp_avx.vec[7]);

      reg[1] = _mm256_srli_epi64(reg[1], 1);

      smulh48_avx(&reg[0], &reg[4], &reg[5], &reg[6], &reg[7], &reg[8], &reg[9], &reg[10], &reg[2], &reg[3]);
      reg[0] = _mm256_add_epi64(reg[0], reg[1]);
      reg[0] = _mm256_srli_epi64(reg[0], 1);
      reg[0] = _mm256_add_epi64(reg[0], reg[12]);

      smulh48_avx(&reg[0], &reg[4], &reg[5], &reg[6], &reg[7], &reg[8], &reg[9], &reg[10], &reg[2], &reg[3]);
      reg[0] = _mm256_add_epi64(reg[0], reg[1]);
      reg[0] = _mm256_srli_epi64(reg[0], 1);
      reg[0] = _mm256_add_epi64(reg[0], reg[13]);

      smulh48_avx(&reg[0], &reg[4], &reg[5], &reg[6], &reg[7], &reg[8], &reg[9], &reg[10], &reg[2], &reg[3]);
      reg[0] = _mm256_add_epi64(reg[0], reg[14]);

      _mm256_store_si256(&rej[i], reg[0]);
    }
}

static const union {
  __m256i vec;
  uint64_t arr[4];
} nolsb_avx = {
  .arr = {
    ~((uint64_t)1), ~((uint64_t)1), ~((uint64_t)1), ~((uint64_t)1)
  }
};

static inline void _rejcond(__m256i ioreg[3], __m256i reg[5], const __m256i *nolsb, const __m256i *one, const __m256i *zero)
{
  reg[0] = _mm256_and_si256(ioreg[0], *nolsb); // without lsb
  //reg[1] = _mm256_and_si256(ioreg[0], *one); // only lsb

  reg[2] = _mm256_cmpgt_epi64(ioreg[1], reg[0]);
  reg[3] = _mm256_cmpeq_epi64(ioreg[2], *zero);
  reg[3] = _mm256_add_epi64(reg[3], *one); // reg[3] is zero when ioreg[2] is zero, else one
  reg[4] = _mm256_or_si256(reg[3], ioreg[0]);
  reg[2] = _mm256_and_si256(reg[2], reg[4]);
  reg[2] = _mm256_and_si256(reg[2], *one); // extract only lsb
  ioreg[0] = _mm256_sub_epi64(*zero, reg[2]); // all-ones if reg[2] was 0, else zero
}

//static void rej_msk_4x(rejection.vec, exp.vec);
static void rej_msk_4x(__m256i *rej, const __m256i *exp, const __m256i *sc)
{
  __m256i reg[16];
  size_t i;

  reg[15] = _mm256_load_si256(&nolsb_avx.vec); // ~1
  reg[14] = _mm256_load_si256(&consts_avx.vec[2]); // 1
  reg[13] = _mm256_load_si256(&zero_avx.vec);

  for (i = 0; i < NUM_GAUSSIANS; i += 2)
  {
    reg[0] = _mm256_load_si256(&rej[i]);
    reg[1] = _mm256_load_si256(&exp[i]);
    reg[2] = _mm256_load_si256(&sc[i]);
    reg[8] = _mm256_load_si256(&rej[i+1]);
    reg[9] = _mm256_load_si256(&exp[i+1]);
    reg[10] = _mm256_load_si256(&sc[i+1]);

    _rejcond(&reg[0], &reg[3], &reg[15], &reg[14], &reg[13]);
    _rejcond(&reg[8], &reg[3], &reg[15], &reg[14], &reg[13]);

    _mm256_store_si256(&rej[i], reg[0]);
    _mm256_store_si256(&rej[i+1], reg[8]);
  }
}

static const union {
  __m256i vec;
  uint64_t arr[4];
} minusone_avx = {
  .arr = {
    -1, -1, -1, -1
  }
};

static void sum_sqr_4x(uint64_t sqsum[2], const __m256i *sqr, const __m256i *rej, __m256i *len)
{
  size_t i, j;
  __m256i reg[16];
  union {
    __m256i vec[2];
    uint64_t arr[8];
  } tmp = {.arr={0}};

  reg[0] = _mm256_load_si256(&zero_avx.vec);
  reg[1] = _mm256_load_si256(&zero_avx.vec);
  reg[14] = _mm256_load_si256(&minusone_avx.vec);
  reg[15] = _mm256_load_si256(len);

  for (i = 0; i < NUM_GAUSSIANS; i += 2)
  {
    reg[2] = _mm256_load_si256(&rej[i+0]);
    reg[3] = _mm256_load_si256(&rej[i+1]);

    reg[ 6] = _mm256_load_si256(&sqr[2*i+0]);
    reg[ 7] = _mm256_load_si256(&sqr[2*i+1]);
    reg[ 8] = _mm256_load_si256(&sqr[2*i+2]);
    reg[ 9] = _mm256_load_si256(&sqr[2*i+3]);

    for (j = 0; j < 2; j++)
    {
      // subtract
      reg[15] = _mm256_add_epi64(reg[15], reg[j+2]);
      reg[13] = _mm256_cmpgt_epi64(reg[15], reg[14]); // compare if >= 0
      reg[j+2] = _mm256_and_si256(reg[j+2], reg[13]); // set to zero if < 0

      reg[2*j+6] = _mm256_and_si256(reg[2*j+6], reg[j+2]);
      reg[2*j+7] = _mm256_and_si256(reg[2*j+7], reg[j+2]);

      reg[0] = _mm256_add_epi64(reg[0], reg[2*j+6]);
      reg[1] = _mm256_add_epi64(reg[1], reg[2*j+7]);
    }
  }

  _mm256_store_si256(&tmp.vec[0], reg[0]);
  _mm256_store_si256(&tmp.vec[1], reg[1]);
  _mm256_store_si256(len, reg[15]);

  sqsum[0] += tmp.arr[0];
  sqsum[0] += tmp.arr[1];
  sqsum[0] += tmp.arr[2];
  sqsum[0] += tmp.arr[3];
  sqsum[1] += tmp.arr[4];
  sqsum[1] += tmp.arr[5];
  sqsum[1] += tmp.arr[6];
  sqsum[1] += tmp.arr[7];

  // renormalize
  sqsum[1] += sqsum[0] >> 48;
  sqsum[0] &= (1ULL<<48)-1;
}


//static void move_to_mem_4x(r0, r1, r2, r3, &ctr0, &ctr1, &ctr2, &ctr3, sample_candidates.vec, rejection.vec);
static void move_to_mem_4x(
  uint64_t *r0, uint64_t *r1, uint64_t *r2, uint64_t *r3, 
  const size_t len0, const size_t len1, const size_t len2, const size_t len3, 
  const __m256i *sc, const __m256i *rej
  )
{
  size_t i;
  uint64_t accepted[4];
  size_t ctr[4] = {0};
  __m128d t;

  for (i = 0; i < NUM_GAUSSIANS; i++)
  {
    t = _mm_castsi128_pd(_mm256_castsi256_si128(rej[i]));
    _mm_storel_pd((__attribute__((__may_alias__)) double *)&accepted[0], t);
    _mm_storeh_pd((__attribute__((__may_alias__)) double *)&accepted[1], t);
    t = _mm_castsi128_pd(_mm256_extracti128_si256(rej[i],1));
    _mm_storel_pd((__attribute__((__may_alias__)) double *)&accepted[2], t);
    _mm_storeh_pd((__attribute__((__may_alias__)) double *)&accepted[3], t);

    t = _mm_castsi128_pd(_mm256_castsi256_si128(sc[i]));
    if (ctr[0] < len0)
    {
      _mm_storel_pd((__attribute__((__may_alias__)) double *)&r0[ctr[0]], t);
      ctr[0] += accepted[0]&1;
    }
    if (ctr[1] < len1)
    {
      _mm_storeh_pd((__attribute__((__may_alias__)) double *)&r1[ctr[1]], t);
      ctr[1] += accepted[1]&1;
    }
    t = _mm_castsi128_pd(_mm256_extracti128_si256(sc[i],1));
    if (ctr[2] < len2)
    {
      _mm_storel_pd((__attribute__((__may_alias__)) double *)&r2[ctr[2]], t);
      ctr[2] += accepted[2]&1;
    }
    if (ctr[3] < len3)
    {
      _mm_storeh_pd((__attribute__((__may_alias__)) double *)&r3[ctr[3]], t);
      ctr[3] += accepted[3]&1;
    }
  }
}

void sample_gauss_N_4x(uint64_t *r0, uint64_t *r1, uint64_t *r2, uint64_t *r3, 
    uint8_t *signs0, uint8_t *signs1, uint8_t *signs2, uint8_t *signs3,
    fp96_76 *sqsum,  const uint8_t seed[CRHBYTES], 
    const uint16_t nonce0, const uint16_t nonce1, const uint16_t nonce2, const uint16_t nonce3,
    const size_t len0, const size_t len1, const size_t len2, const size_t len3) {
    size_t ctr0, ctr1, ctr2, ctr3;
    size_t bytecnt, i;
    ALIGNED_UINT8(CRHBYTES+2) buf[4];
    ALIGNED_UINT8(POLY_HYPERBALL_NBLOCKS_4X * STREAM256_BLOCKBYTES * 4) outbuf;
    ALIGNED_UINT8(STREAM256_BLOCKBYTES*2) outbuf2[4];
    ALIGNED_INT64(NUM_GAUSSIANS * 4) sample_candidates;
    ALIGNED_INT64(NUM_GAUSSIANS * 4) exp;
    ALIGNED_INT64(NUM_GAUSSIANS * 4) rejection;
    ALIGNED_INT64(NUM_GAUSSIANS * 8) sqr;

    keccakx4_state state;

    __m256i f;
    union {
      __m256i vec;
      int64_t arr[4];
    } len = {
      .arr = {len0, len1, len2, len3}
    };
    uint64_t *seed64 = (uint64_t *) seed;

    // f = _mm256_loadu_si256((__m256i *)seed);
    // _mm256_store_si256(buf[0].vec,f);
    // _mm256_store_si256(buf[1].vec,f);
    // _mm256_store_si256(buf[2].vec,f);
    // _mm256_store_si256(buf[3].vec,f);
    // f = _mm256_loadu_si256((__m256i *)&seed[32]);
    // _mm256_store_si256(&buf[0].vec[1],f);
    // _mm256_store_si256(&buf[1].vec[1],f);
    // _mm256_store_si256(&buf[2].vec[1],f);
    // _mm256_store_si256(&buf[3].vec[1],f);
    //
    // buf[0].coeffs[CRHBYTES+0] = nonce0;
    // buf[0].coeffs[CRHBYTES+1] = nonce0 >> 8;
    // buf[1].coeffs[CRHBYTES+0] = nonce1;
    // buf[1].coeffs[CRHBYTES+1] = nonce1 >> 8;
    // buf[2].coeffs[CRHBYTES+0] = nonce2;
    // buf[2].coeffs[CRHBYTES+1] = nonce2 >> 8;
    // buf[3].coeffs[CRHBYTES+0] = nonce3;
    // buf[3].coeffs[CRHBYTES+1] = nonce3 >> 8;
    //
    // shake256x4_absorb_once(&state, buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, CRHBYTES + 2);

  state.s[0] = _mm256_set1_epi64x(seed64[0]);
  state.s[1] = _mm256_set1_epi64x(seed64[1]);
  state.s[2] = _mm256_set1_epi64x(seed64[2]);
  state.s[3] = _mm256_set1_epi64x(seed64[3]);
  state.s[4] = _mm256_set1_epi64x(seed64[4]);
  state.s[5] = _mm256_set1_epi64x(seed64[5]);
  state.s[6] = _mm256_set1_epi64x(seed64[6]);
  state.s[7] = _mm256_set1_epi64x(seed64[7]);
  state.s[8] = _mm256_set_epi64x((0x1f << 16) ^ nonce3, (0x1f << 16) ^ nonce2, (0x1f << 16) ^ nonce1,
                                 (0x1f << 16) ^ nonce0);

  for (int j = 9; j < 25; ++j) state.s[j] = _mm256_setzero_si256();

  state.s[16] = _mm256_set1_epi64x(0x1ULL << 63);


    shake256x4_squeezeblocks_vec(&outbuf.vec[0], POLY_HYPERBALL_NBLOCKS_4X, &state);

    // copy signs
    for (size_t i = 0; i < len0 / 8; i++) {
        signs0[i] = outbuf.coeffs[(i % 8) + (i / 8) * 32 +  0];
        signs1[i] = outbuf.coeffs[(i % 8) + (i / 8) * 32 +  8];
        signs2[i] = outbuf.coeffs[(i % 8) + (i / 8) * 32 + 16];
        signs3[i] = outbuf.coeffs[(i % 8) + (i / 8) * 32 + 24];
    }

    // step 0: write CDT randomness into sample_candidates, 72 bit randomness into sqr, and rejection randomness into rejection
    preprocessing_x4(sample_candidates.vec, sqr.vec, rejection.vec, &outbuf.vec[4]); // first four vectors are just signs

    // step 1: sample from CDT distribution, store all results
    sample_gauss16_x4(sample_candidates.vec); 

    // step 2: construct the sample candidates, compute sqr and input to exp approximation
    sample_candidates_x4(sample_candidates.vec, sqr.vec, exp.vec);

    // step 3: approximate exponential
    approx_exp_4x(exp.vec);

    // step 4: compute rejection mask
    rej_msk_4x(rejection.vec, exp.vec, sample_candidates.vec);

    // step 5: sum up sqr 
    sum_sqr_4x(&sqsum->limb48[0], sqr.vec, rejection.vec, &len.vec);
    
    // step 6: move accepted samples to correct memory positions
    move_to_mem_4x(r0, r1, r2, r3, len0-len0%N, len1-len1%N, len2-len2%N, len3-len3%N, sample_candidates.vec, rejection.vec);
    
    ctr0 = len.arr[0] > 0 ? len0-len.arr[0] : len0;
    ctr1 = len.arr[1] > 0 ? len1-len.arr[1] : len1;
    ctr2 = len.arr[2] > 0 ? len2-len.arr[2] : len2;
    ctr3 = len.arr[3] > 0 ? len3-len.arr[3] : len3;

    // copy remaining bytes
    bytecnt = POLY_HYPERBALL_NBLOCKS_4X * STREAM256_BLOCKBYTES - N/8 - NUM_GAUSSIANS * GAUSS_RAND_BYTES;
    if (ctr0 < len0)
    {
      uint8_t *bptr = &outbuf.coeffs[bytecnt + NUM_GAUSSIANS * GAUSS_RAND_BYTES + N/8];
      for (i = 0; i < bytecnt; i++)
      {
        outbuf2[0].coeffs[i] = *bptr++;
        if ((i+1)%8 == 0) {
          bptr += 3 * 8; // jump over the next vectors
        }
      }
    }
    if (ctr1 < len1)
    {
      uint8_t *bptr = &outbuf.coeffs[bytecnt + NUM_GAUSSIANS * GAUSS_RAND_BYTES + N/8 + 8];
      for (i = 0; i < bytecnt; i++)
      {
        outbuf2[1].coeffs[i] = *bptr++;
        if ((i+1)%8 == 0) {
          bptr += 3 * 8; // jump over the next vectors
        }
      }
    }
    if (ctr2 < len2)
    {
      uint8_t *bptr = &outbuf.coeffs[bytecnt + NUM_GAUSSIANS * GAUSS_RAND_BYTES + N/8 + 16];
      for (i = 0; i < bytecnt; i++)
      {
        outbuf2[2].coeffs[i] = *bptr++;
        if ((i+1)%8 == 0) {
          bptr += 3 * 8; // jump over the next vectors
        }
      }
    }
    if (ctr3 < len3)
    {
      uint8_t *bptr = &outbuf.coeffs[bytecnt + NUM_GAUSSIANS * GAUSS_RAND_BYTES + N/8 + 24];
      for (i = 0; i < bytecnt; i++)
      {
        outbuf2[3].coeffs[i] = *bptr++;
        if ((i+1)%8 == 0) {
          bptr += 3 * 8; // jump over the next vectors
        }
      }
    }

    while (ctr0 < len0 || ctr1 < len1 || ctr2 < len2 || ctr3 < len3)
    {
        shake256x4_squeezeblocks(&outbuf2[0].coeffs[bytecnt], &outbuf2[1].coeffs[bytecnt], &outbuf2[2].coeffs[bytecnt], &outbuf2[3].coeffs[bytecnt], 1, &state);

        bytecnt += STREAM256_BLOCKBYTES;

        ctr0 += sample_gauss(r0 + ctr0, sqsum, outbuf2[0].coeffs, bytecnt, len0 - ctr0, len0%N);
        ctr1 += sample_gauss(r1 + ctr1, sqsum, outbuf2[1].coeffs, bytecnt, len1 - ctr1, len1%N);
        ctr2 += sample_gauss(r2 + ctr2, sqsum, outbuf2[2].coeffs, bytecnt, len2 - ctr2, len2%N);
        ctr3 += sample_gauss(r3 + ctr3, sqsum, outbuf2[3].coeffs, bytecnt, len3 - ctr3, len3%N);

        if (ctr0 < len0 || ctr1 < len1 || ctr2 < len2 || ctr3 < len3)
        {
            size_t offset = (bytecnt / GAUSS_RAND_BYTES) * GAUSS_RAND_BYTES;
            size_t rem_bytes = bytecnt - offset;
            for (size_t i = 0; i < rem_bytes; i++)
            {
              outbuf2[0].coeffs[i] = outbuf2[0].coeffs[i + offset];
              outbuf2[1].coeffs[i] = outbuf2[1].coeffs[i + offset];
              outbuf2[2].coeffs[i] = outbuf2[2].coeffs[i + offset];
              outbuf2[3].coeffs[i] = outbuf2[3].coeffs[i + offset];
            }
            bytecnt = rem_bytes;
        }
    }
    renormalize(sqsum);
}


void sample_gauss_N_Rnd_4x(loop_queue *loop, const uint8_t seed[CRHBYTES], uint16_t nonce0, uint16_t nonce1,
                           uint16_t nonce2, uint16_t nonce3) {
  ALIGNED_UINT8(CRHBYTES+2) buf[4];
  uint8_t *y_buff1, *y_buff2, *y_buff3, *y_buff4;
  const int offet = POLY_HYPERBALL_NBLOCKS_4X * STREAM256_BLOCKBYTES;
  keccakx4_state state;

  __m256i f;

  f = _mm256_loadu_si256((__m256i *) seed);
  _mm256_store_si256(buf[0].vec, f);
  _mm256_store_si256(buf[1].vec, f);
  _mm256_store_si256(buf[2].vec, f);
  _mm256_store_si256(buf[3].vec, f);
  f = _mm256_loadu_si256((__m256i *) &seed[32]);
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

  shake256x4_absorb_once(&state, buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, CRHBYTES + 2);
  y_buff1 = loop_next(loop);
  y_buff2 = loop_next(loop);
  y_buff3 = loop_next(loop);
  y_buff4 = loop_next(loop);

  ST_AVX2_shake256x4_squeezeblocks(y_buff1, y_buff2, y_buff3, y_buff4, POLY_HYPERBALL_NBLOCKS_4X, &state);
  store_statex4(y_buff1 + offet, y_buff2 + offet, y_buff3 + offet, y_buff4 + offet, &state);
}

void generate_vec_buf_x4(uint8_t *y_buff0,uint8_t *y_buff1,uint8_t  *y_buff2,uint8_t  *y_buff3, guss_N_buf_x4 *outbuf) {
  __m256i t0, t1, t2, t3;
  __m256i f0, f1, f2, f3;
  __m128i t;
  __m256i s[4];

  for (int i = 0; i < POLY_HYPERBALL_NBLOCKS_4X; ++i) {
    for (int j = 0; j < 4; ++j) {
      s[0] = _mm256_loadu_si256(y_buff0 + i * SHAKE256_RATE + j * 32);
      s[1] = _mm256_loadu_si256(y_buff1 + i * SHAKE256_RATE + j * 32);
      s[2] = _mm256_loadu_si256(y_buff2 + i * SHAKE256_RATE + j * 32);
      s[3] = _mm256_loadu_si256(y_buff3 + i * SHAKE256_RATE + j * 32);

      t0 = _mm256_unpacklo_epi64(s[0], s[1]);
      t1 = _mm256_unpackhi_epi64(s[0], s[1]);
      t2 = _mm256_unpacklo_epi64(s[2], s[3]);
      t3 = _mm256_unpackhi_epi64(s[2], s[3]);

      f0 = _mm256_permute2x128_si256(t0, t2, 0x20);
      f1 = _mm256_permute2x128_si256(t1, t3, 0x20);
      f2 = _mm256_permute2x128_si256(t0, t2, 0x31);
      f3 = _mm256_permute2x128_si256(t1, t3, 0x31);

      _mm256_storeu_si256(&outbuf->vec[i * 17 + j * 4 + 0], f0);
      _mm256_storeu_si256(&outbuf->vec[i * 17 + j * 4 + 1], f1);
      _mm256_storeu_si256(&outbuf->vec[i * 17 + j * 4 + 2], f2);
      _mm256_storeu_si256(&outbuf->vec[i * 17 + j * 4 + 3], f3);
    }
    f0 = _mm256_setr_epi64x(*((uint64_t*)(y_buff0 + i * SHAKE256_RATE + 128)),
                           *((uint64_t*)(y_buff1 + i * SHAKE256_RATE + 128)),
                           *((uint64_t*)(y_buff2 + i * SHAKE256_RATE + 128)),
                           *((uint64_t*)(y_buff3 + i * SHAKE256_RATE + 128)));
    _mm256_storeu_si256(&outbuf->vec[i * 17 + 16], f0);
  }
}


void sample_gauss_N_RejS_4x(loop_queue *loop,uint64_t *r0, uint64_t *r1, uint64_t *r2, uint64_t *r3,
    uint8_t *signs0, uint8_t *signs1, uint8_t *signs2, uint8_t *signs3,
    fp96_76 *sqsum,
    size_t len0, size_t len1, size_t len2, size_t len3) {
    size_t ctr0, ctr1, ctr2, ctr3;
    size_t bytecnt, i;
    ALIGNED_UINT8(STREAM256_BLOCKBYTES*2) outbuf2[4];
    ALIGNED_INT64(NUM_GAUSSIANS * 4) sample_candidates;
    ALIGNED_INT64(NUM_GAUSSIANS * 4) exp;
    ALIGNED_INT64(NUM_GAUSSIANS * 4) rejection;
    ALIGNED_INT64(NUM_GAUSSIANS * 8) sqr;
    guss_N_buf_x4 outbuf;
    uint8_t *y_buff0,*y_buff1, *y_buff2, *y_buff3;
    y_buff0 = loop_dequeue(loop);
    y_buff1 = loop_dequeue(loop);
    y_buff2 = loop_dequeue(loop);
    y_buff3 = loop_dequeue(loop);

    generate_vec_buf_x4(y_buff0,y_buff1, y_buff2, y_buff3, &outbuf);

    keccakx4_state state;

    union {
      __m256i vec;
      int64_t arr[4];
    } len = {
      .arr = {len0, len1, len2, len3}
    };

  // copy signs
    memcpy(signs0, y_buff0, len0/8);
    memcpy(signs1, y_buff1, len1/8);
    memcpy(signs2, y_buff2, len2/8);
    memcpy(signs3, y_buff3, len3/8);

    // step 0: write CDT randomness into sample_candidates, 72 bit randomness into sqr, and rejection randomness into rejection
    preprocessing_x4(sample_candidates.vec, sqr.vec, rejection.vec, &outbuf.vec[4]); // first four vectors are just signs

    // step 1: sample from CDT distribution, store all results
    sample_gauss16_x4(sample_candidates.vec);

    // step 2: construct the sample candidates, compute sqr and input to exp approximation
    sample_candidates_x4(sample_candidates.vec, sqr.vec, exp.vec);

    // step 3: approximate exponential
    approx_exp_4x(exp.vec);

    // step 4: compute rejection mask
    rej_msk_4x(rejection.vec, exp.vec, sample_candidates.vec);

    // step 5: sum up sqr
    sum_sqr_4x(&sqsum->limb48[0], sqr.vec, rejection.vec, &len.vec);

    // step 6: move accepted samples to correct memory positions
    move_to_mem_4x(r0, r1, r2, r3, len0-len0%N, len1-len1%N, len2-len2%N, len3-len3%N, sample_candidates.vec, rejection.vec);

    ctr0 = len.arr[0] > 0 ? len0-len.arr[0] : len0;
    ctr1 = len.arr[1] > 0 ? len1-len.arr[1] : len1;
    ctr2 = len.arr[2] > 0 ? len2-len.arr[2] : len2;
    ctr3 = len.arr[3] > 0 ? len3-len.arr[3] : len3;

    // copy remaining bytes
    bytecnt = POLY_HYPERBALL_NBLOCKS_4X * STREAM256_BLOCKBYTES - N/8 - NUM_GAUSSIANS * GAUSS_RAND_BYTES;
    if (ctr0 < len0)
    {
      uint8_t *bptr = &outbuf.coeffs[bytecnt + NUM_GAUSSIANS * GAUSS_RAND_BYTES + N/8];
      for (i = 0; i < bytecnt; i++)
      {
        outbuf2[0].coeffs[i] = *bptr++;
        if ((i+1)%8 == 0) {
          bptr += 3 * 8; // jump over the next vectors
        }
      }
    }
    if (ctr1 < len1)
    {
      uint8_t *bptr = &outbuf.coeffs[bytecnt + NUM_GAUSSIANS * GAUSS_RAND_BYTES + N/8 + 8];
      for (i = 0; i < bytecnt; i++)
      {
        outbuf2[1].coeffs[i] = *bptr++;
        if ((i+1)%8 == 0) {
          bptr += 3 * 8; // jump over the next vectors
        }
      }
    }
    if (ctr2 < len2)
    {
      uint8_t *bptr = &outbuf.coeffs[bytecnt + NUM_GAUSSIANS * GAUSS_RAND_BYTES + N/8 + 16];
      for (i = 0; i < bytecnt; i++)
      {
        outbuf2[2].coeffs[i] = *bptr++;
        if ((i+1)%8 == 0) {
          bptr += 3 * 8; // jump over the next vectors
        }
      }
    }
    if (ctr3 < len3)
    {
      uint8_t *bptr = &outbuf.coeffs[bytecnt + NUM_GAUSSIANS * GAUSS_RAND_BYTES + N/8 + 24];
      for (i = 0; i < bytecnt; i++)
      {
        outbuf2[3].coeffs[i] = *bptr++;
        if ((i+1)%8 == 0) {
          bptr += 3 * 8; // jump over the next vectors
        }
      }
    }

    while (ctr0 < len0 || ctr1 < len1 || ctr2 < len2 || ctr3 < len3)
    {
        const int offset0 = (POLY_HYPERBALL_NBLOCKS_4X - 1) * STREAM256_BLOCKBYTES;
        recover_statex4(y_buff0 + offset0,y_buff1 + offset0, y_buff2 + offset0, y_buff3 + offset0, &state);
        shake256x4_squeezeblocks(&outbuf2[0].coeffs[bytecnt], &outbuf2[1].coeffs[bytecnt], &outbuf2[2].coeffs[bytecnt], &outbuf2[3].coeffs[bytecnt], 1, &state);

        bytecnt += STREAM256_BLOCKBYTES;

        ctr0 += sample_gauss(r0 + ctr0, sqsum, outbuf2[0].coeffs, bytecnt, len0 - ctr0, len0%N);
        ctr1 += sample_gauss(r1 + ctr1, sqsum, outbuf2[1].coeffs, bytecnt, len1 - ctr1, len1%N);
        ctr2 += sample_gauss(r2 + ctr2, sqsum, outbuf2[2].coeffs, bytecnt, len2 - ctr2, len2%N);
        ctr3 += sample_gauss(r3 + ctr3, sqsum, outbuf2[3].coeffs, bytecnt, len3 - ctr3, len3%N);

        if (ctr0 < len0 || ctr1 < len1 || ctr2 < len2 || ctr3 < len3)
        {
            size_t offset = (bytecnt / GAUSS_RAND_BYTES) * GAUSS_RAND_BYTES;
            size_t rem_bytes = bytecnt - offset;
            for (size_t i = 0; i < rem_bytes; i++)
            {
              outbuf2[0].coeffs[i] = outbuf2[0].coeffs[i + offset];
              outbuf2[1].coeffs[i] = outbuf2[1].coeffs[i + offset];
              outbuf2[2].coeffs[i] = outbuf2[2].coeffs[i + offset];
              outbuf2[3].coeffs[i] = outbuf2[3].coeffs[i + offset];
            }
            bytecnt = rem_bytes;
        }
    }
    renormalize(sqsum);
}

void generate_vec_buf_x3(uint8_t *y_buff0,uint8_t *y_buff1,uint8_t  *y_buff2, guss_N_buf_x4 *outbuf) {

  __m256i t0, t1, t2, t3;
  __m256i f0, f1, f2, f3;
  __m128i t;
  __m256i s[4];

  for (int i = 0; i < POLY_HYPERBALL_NBLOCKS_4X; ++i) {
    for (int j = 0; j < 4; ++j) {
      s[0] = _mm256_loadu_si256(y_buff0 + i * SHAKE256_RATE + j * 32);
      s[1] = _mm256_loadu_si256(y_buff1 + i * SHAKE256_RATE + j * 32);
      s[2] = _mm256_loadu_si256(y_buff2 + i * SHAKE256_RATE + j * 32);
      s[3] = _mm256_setzero_si256();

      t0 = _mm256_unpacklo_epi64(s[0], s[1]);
      t1 = _mm256_unpackhi_epi64(s[0], s[1]);
      t2 = _mm256_unpacklo_epi64(s[2], s[3]);
      t3 = _mm256_unpackhi_epi64(s[2], s[3]);

      f0 = _mm256_permute2x128_si256(t0, t2, 0x20);
      f1 = _mm256_permute2x128_si256(t1, t3, 0x20);
      f2 = _mm256_permute2x128_si256(t0, t2, 0x31);
      f3 = _mm256_permute2x128_si256(t1, t3, 0x31);

      _mm256_storeu_si256(&outbuf->vec[i * 17 + j * 4 + 0], f0);
      _mm256_storeu_si256(&outbuf->vec[i * 17 + j * 4 + 1], f1);
      _mm256_storeu_si256(&outbuf->vec[i * 17 + j * 4 + 2], f2);
      _mm256_storeu_si256(&outbuf->vec[i * 17 + j * 4 + 3], f3);
    }
    f0 = _mm256_setr_epi64x(*((uint64_t*)(y_buff0 + i * SHAKE256_RATE + 128)),
                           *((uint64_t*)(y_buff1 + i * SHAKE256_RATE + 128)),
                           *((uint64_t*)(y_buff2 + i * SHAKE256_RATE + 128)),
                           0);
    _mm256_storeu_si256(&outbuf->vec[i * 17 + 16], f0);
  }
}

void sample_gauss_N_RejS_3x(loop_queue *loop,uint64_t *r0, uint64_t *r1, uint64_t *r2,
    uint8_t *signs0, uint8_t *signs1, uint8_t *signs2,
    fp96_76 *sqsum,
    size_t len0, size_t len1, size_t len2) {
    size_t ctr0, ctr1, ctr2, ctr3;
    size_t bytecnt, i;
    ALIGNED_UINT8(STREAM256_BLOCKBYTES*2) outbuf2[4];
    ALIGNED_INT64(NUM_GAUSSIANS * 4) sample_candidates;
    ALIGNED_INT64(NUM_GAUSSIANS * 4) exp;
    ALIGNED_INT64(NUM_GAUSSIANS * 4) rejection;
    ALIGNED_INT64(NUM_GAUSSIANS * 8) sqr;
    guss_N_buf_x4 outbuf;
    uint8_t *y_buff0,*y_buff1, *y_buff2;
    y_buff0 = loop_dequeue(loop);
    y_buff1 = loop_dequeue(loop);
    y_buff2 = loop_dequeue(loop);

    generate_vec_buf_x3(y_buff0,y_buff1, y_buff2, &outbuf);
    uint64_t samples[N * 2];

    keccakx4_state state;

    union {
      __m256i vec;
      int64_t arr[4];
    } len = {
      .arr = {len0, len1, len2, 0}
    };

    // copy signs
  memcpy(signs0, y_buff0, len0/8);
  memcpy(signs1, y_buff1, len1/8);
  memcpy(signs2, y_buff2, len2/8);

    // step 0: write CDT randomness into sample_candidates, 72 bit randomness into sqr, and rejection randomness into rejection
    preprocessing_x4(sample_candidates.vec, sqr.vec, rejection.vec, &outbuf.vec[4]); // first four vectors are just signs

    // step 1: sample from CDT distribution, store all results
    sample_gauss16_x4(sample_candidates.vec);

    // step 2: construct the sample candidates, compute sqr and input to exp approximation
    sample_candidates_x4(sample_candidates.vec, sqr.vec, exp.vec);

    // step 3: approximate exponential
    approx_exp_4x(exp.vec);

    // step 4: compute rejection mask
    rej_msk_4x(rejection.vec, exp.vec, sample_candidates.vec);

    // step 5: sum up sqr
    sum_sqr_4x(&sqsum->limb48[0], sqr.vec, rejection.vec, &len.vec);

    // step 6: move accepted samples to correct memory positions
    move_to_mem_4x(r0, r1, r2, samples, len0-len0%N, len1-len1%N, len2-len2%N, 0, sample_candidates.vec, rejection.vec);

    ctr0 = len.arr[0] > 0 ? len0-len.arr[0] : len0;
    ctr1 = len.arr[1] > 0 ? len1-len.arr[1] : len1;
    ctr2 = len.arr[2] > 0 ? len2-len.arr[2] : len2;

    // copy remaining bytes
    bytecnt = POLY_HYPERBALL_NBLOCKS_4X * STREAM256_BLOCKBYTES - N/8 - NUM_GAUSSIANS * GAUSS_RAND_BYTES;
    if (ctr0 < len0)
    {
      uint8_t *bptr = &outbuf.coeffs[bytecnt + NUM_GAUSSIANS * GAUSS_RAND_BYTES + N/8];
      for (i = 0; i < bytecnt; i++)
      {
        outbuf2[0].coeffs[i] = *bptr++;
        if ((i+1)%8 == 0) {
          bptr += 3 * 8; // jump over the next vectors
        }
      }
    }
    if (ctr1 < len1)
    {
      uint8_t *bptr = &outbuf.coeffs[bytecnt + NUM_GAUSSIANS * GAUSS_RAND_BYTES + N/8 + 8];
      for (i = 0; i < bytecnt; i++)
      {
        outbuf2[1].coeffs[i] = *bptr++;
        if ((i+1)%8 == 0) {
          bptr += 3 * 8; // jump over the next vectors
        }
      }
    }
    if (ctr2 < len2)
    {
      uint8_t *bptr = &outbuf.coeffs[bytecnt + NUM_GAUSSIANS * GAUSS_RAND_BYTES + N/8 + 16];
      for (i = 0; i < bytecnt; i++)
      {
        outbuf2[2].coeffs[i] = *bptr++;
        if ((i+1)%8 == 0) {
          bptr += 3 * 8; // jump over the next vectors
        }
      }
    }


    while (ctr0 < len0 || ctr1 < len1 || ctr2 < len2 )
    {
      const int offset0 = (POLY_HYPERBALL_NBLOCKS_4X - 1) * STREAM256_BLOCKBYTES;
      recover_statex4(y_buff0 + offset0,y_buff1 + offset0, y_buff2 + offset0, y_buff2, &state);
      shake256x4_squeezeblocks(&outbuf2[0].coeffs[bytecnt], &outbuf2[1].coeffs[bytecnt], &outbuf2[2].coeffs[bytecnt], &outbuf2[3].coeffs[bytecnt], 1, &state);

        bytecnt += STREAM256_BLOCKBYTES;

        ctr0 += sample_gauss(r0 + ctr0, sqsum, outbuf2[0].coeffs, bytecnt, len0 - ctr0, len0%N);
        ctr1 += sample_gauss(r1 + ctr1, sqsum, outbuf2[1].coeffs, bytecnt, len1 - ctr1, len1%N);
        ctr2 += sample_gauss(r2 + ctr2, sqsum, outbuf2[2].coeffs, bytecnt, len2 - ctr2, len2%N);

        if (ctr0 < len0 || ctr1 < len1 || ctr2 < len2 )
        {
            size_t offset = (bytecnt / GAUSS_RAND_BYTES) * GAUSS_RAND_BYTES;
            size_t rem_bytes = bytecnt - offset;
            for (size_t i = 0; i < rem_bytes; i++)
            {
              outbuf2[0].coeffs[i] = outbuf2[0].coeffs[i + offset];
              outbuf2[1].coeffs[i] = outbuf2[1].coeffs[i + offset];
              outbuf2[2].coeffs[i] = outbuf2[2].coeffs[i + offset];
            }
            bytecnt = rem_bytes;
        }
    }
    renormalize(sqsum);
}

void generate_vec_buf_x2(uint8_t *y_buff0,uint8_t *y_buff1,guss_N_buf_x4 *outbuf) {

  __m256i t0, t1, t2, t3;
  __m256i f0, f1, f2, f3;
  __m128i t;
  __m256i s[4];

  for (int i = 0; i < POLY_HYPERBALL_NBLOCKS_4X; ++i) {
    for (int j = 0; j < 4; ++j) {
      s[0] = _mm256_loadu_si256(y_buff0 + i * SHAKE256_RATE + j * 32);
      s[1] = _mm256_loadu_si256(y_buff1 + i * SHAKE256_RATE + j * 32);
      t2 = _mm256_setzero_si256();
      t3 = _mm256_setzero_si256();

      t0 = _mm256_unpacklo_epi64(s[0], s[1]);
      t1 = _mm256_unpackhi_epi64(s[0], s[1]);

      f0 = _mm256_permute2x128_si256(t0, t2, 0x20);
      f1 = _mm256_permute2x128_si256(t1, t3, 0x20);
      f2 = _mm256_permute2x128_si256(t0, t2, 0x31);
      f3 = _mm256_permute2x128_si256(t1, t3, 0x31);

      _mm256_storeu_si256(&outbuf->vec[i * 17 + j * 4 + 0], f0);
      _mm256_storeu_si256(&outbuf->vec[i * 17 + j * 4 + 1], f1);
      _mm256_storeu_si256(&outbuf->vec[i * 17 + j * 4 + 2], f2);
      _mm256_storeu_si256(&outbuf->vec[i * 17 + j * 4 + 3], f3);
    }
    f0 = _mm256_setr_epi64x(*((uint64_t*)(y_buff0 + i * SHAKE256_RATE + 128)),
                           *((uint64_t*)(y_buff1 + i * SHAKE256_RATE + 128)),
                           0, 0);
    _mm256_storeu_si256(&outbuf->vec[i * 17 + 16], f0);
  }
}

void sample_gauss_N_RejS_2x(loop_queue *loop,uint64_t *r0, uint64_t *r1,
    uint8_t *signs0, uint8_t *signs1,
    fp96_76 *sqsum,
    size_t len0, size_t len1) {
    size_t ctr0, ctr1, ctr2, ctr3;
    size_t bytecnt, i;
    ALIGNED_UINT8(STREAM256_BLOCKBYTES*2) outbuf2[4];
    ALIGNED_INT64(NUM_GAUSSIANS * 4) sample_candidates;
    ALIGNED_INT64(NUM_GAUSSIANS * 4) exp;
    ALIGNED_INT64(NUM_GAUSSIANS * 4) rejection;
    ALIGNED_INT64(NUM_GAUSSIANS * 8) sqr;
    guss_N_buf_x4 outbuf;
  uint8_t *y_buff0,*y_buff1;
  y_buff0 = loop_dequeue(loop);
  y_buff1 = loop_dequeue(loop);
    generate_vec_buf_x2(y_buff0,y_buff1, &outbuf);
    uint64_t samples[N * 2];

    keccakx4_state state;

    union {
      __m256i vec;
      int64_t arr[4];
    } len = {
      .arr = {len0, len1, 0, 0}
    };

    // copy signs
  memcpy(signs0, y_buff0, len0/8);
  memcpy(signs1, y_buff1, len1/8);

    // step 0: write CDT randomness into sample_candidates, 72 bit randomness into sqr, and rejection randomness into rejection
    preprocessing_x4(sample_candidates.vec, sqr.vec, rejection.vec, &outbuf.vec[4]); // first four vectors are just signs

    // step 1: sample from CDT distribution, store all results
    sample_gauss16_x4(sample_candidates.vec);

    // step 2: construct the sample candidates, compute sqr and input to exp approximation
    sample_candidates_x4(sample_candidates.vec, sqr.vec, exp.vec);

    // step 3: approximate exponential
    approx_exp_4x(exp.vec);

    // step 4: compute rejection mask
    rej_msk_4x(rejection.vec, exp.vec, sample_candidates.vec);

    // step 5: sum up sqr
    sum_sqr_4x(&sqsum->limb48[0], sqr.vec, rejection.vec, &len.vec);

    // step 6: move accepted samples to correct memory positions
    move_to_mem_4x(r0, r1, samples, samples + N, len0-len0%N, len1-len1%N, 0, 0, sample_candidates.vec, rejection.vec);

    ctr0 = len.arr[0] > 0 ? len0-len.arr[0] : len0;
    ctr1 = len.arr[1] > 0 ? len1-len.arr[1] : len1;

    // copy remaining bytes
    bytecnt = POLY_HYPERBALL_NBLOCKS_4X * STREAM256_BLOCKBYTES - N/8 - NUM_GAUSSIANS * GAUSS_RAND_BYTES;
    if (ctr0 < len0)
    {
      uint8_t *bptr = &outbuf.coeffs[bytecnt + NUM_GAUSSIANS * GAUSS_RAND_BYTES + N/8];
      for (i = 0; i < bytecnt; i++)
      {
        outbuf2[0].coeffs[i] = *bptr++;
        if ((i+1)%8 == 0) {
          bptr += 3 * 8; // jump over the next vectors
        }
      }
    }
    if (ctr1 < len1)
    {
      uint8_t *bptr = &outbuf.coeffs[bytecnt + NUM_GAUSSIANS * GAUSS_RAND_BYTES + N/8 + 8];
      for (i = 0; i < bytecnt; i++)
      {
        outbuf2[1].coeffs[i] = *bptr++;
        if ((i+1)%8 == 0) {
          bptr += 3 * 8; // jump over the next vectors
        }
      }
    }

    while (ctr0 < len0 || ctr1 < len1 )
    {
      const int offset0 = (POLY_HYPERBALL_NBLOCKS_4X - 1) * STREAM256_BLOCKBYTES;
      recover_statex4(y_buff0 + offset0,y_buff1 + offset0, y_buff0, y_buff1, &state);
        shake256x4_squeezeblocks(&outbuf2[0].coeffs[bytecnt], &outbuf2[1].coeffs[bytecnt], &outbuf2[2].coeffs[bytecnt], &outbuf2[3].coeffs[bytecnt], 1, &state);
        bytecnt += STREAM256_BLOCKBYTES;

        ctr0 += sample_gauss(r0 + ctr0, sqsum, outbuf2[0].coeffs, bytecnt, len0 - ctr0, len0%N);
        ctr1 += sample_gauss(r1 + ctr1, sqsum, outbuf2[1].coeffs, bytecnt, len1 - ctr1, len1%N);

        if (ctr0 < len0 || ctr1 < len1 )
        {
            size_t offset = (bytecnt / GAUSS_RAND_BYTES) * GAUSS_RAND_BYTES;
            size_t rem_bytes = bytecnt - offset;
            for (size_t i = 0; i < rem_bytes; i++)
            {
              outbuf2[0].coeffs[i] = outbuf2[0].coeffs[i + offset];
              outbuf2[1].coeffs[i] = outbuf2[1].coeffs[i + offset];
            }
            bytecnt = rem_bytes;
        }
    }
    renormalize(sqsum);
}

void generate_vec_buf_x1(loop_queue *loop,guss_N_buf_x4 *outbuf) {
  uint8_t *y_buff0;
  y_buff0 = loop_dequeue(loop);

  __m256i t0, t1, t2, t3;
  __m256i f0, f1, f2, f3;
  __m128i t;
  __m256i s[4];

  for (int i = 0; i < POLY_HYPERBALL_NBLOCKS_4X; ++i) {
    for (int j = 0; j < 4; ++j) {
      s[0] = _mm256_loadu_si256(y_buff0 + i * SHAKE256_RATE + j * 32);
      s[1] = _mm256_setzero_si256();
      s[2] = _mm256_setzero_si256();
      s[3] = _mm256_setzero_si256();

      t0 = _mm256_unpacklo_epi64(s[0], s[1]);
      t1 = _mm256_unpackhi_epi64(s[0], s[1]);
      t2 = _mm256_unpacklo_epi64(s[2], s[3]);
      t3 = _mm256_unpackhi_epi64(s[2], s[3]);

      f0 = _mm256_permute2x128_si256(t0, t2, 0x20);
      f1 = _mm256_permute2x128_si256(t1, t3, 0x20);
      f2 = _mm256_permute2x128_si256(t0, t2, 0x31);
      f3 = _mm256_permute2x128_si256(t1, t3, 0x31);

      _mm256_storeu_si256(&outbuf->vec[i * 17 + j * 4 + 0], f0);
      _mm256_storeu_si256(&outbuf->vec[i * 17 + j * 4 + 1], f1);
      _mm256_storeu_si256(&outbuf->vec[i * 17 + j * 4 + 2], f2);
      _mm256_storeu_si256(&outbuf->vec[i * 17 + j * 4 + 3], f3);
    }
    f0 = _mm256_setr_epi64x(*((uint64_t*)(y_buff0 + i * SHAKE256_RATE + 128)),
                           0,0, 0);
    _mm256_storeu_si256(&outbuf->vec[i * 17 + 16], f0);
  }
}

void sample_gauss_N_RejS_1x(loop_queue *loop,uint64_t *r0,
    uint8_t *signs0,
    fp96_76 *sqsum,
    size_t len0) {
    size_t ctr0;
    size_t bytecnt, i;
    ALIGNED_UINT8(STREAM256_BLOCKBYTES*2) outbuf2[4];
    ALIGNED_INT64(NUM_GAUSSIANS * 4) sample_candidates;
    ALIGNED_INT64(NUM_GAUSSIANS * 4) exp;
    ALIGNED_INT64(NUM_GAUSSIANS * 4) rejection;
    ALIGNED_INT64(NUM_GAUSSIANS * 8) sqr;
    guss_N_buf_x4 outbuf;
    generate_vec_buf_x1(loop, &outbuf);
    uint64_t samples[N * 3];

    keccakx4_state state;

    union {
      __m256i vec;
      int64_t arr[4];
    } len = {
      .arr = {len0, 0, 0, 0}
    };

    // copy signs
    for (size_t i = 0; i < len0 / 8; i++) {
        signs0[i] = outbuf.coeffs[(i % 8) + (i / 8) * 32 +  0];
    }

    // step 0: write CDT randomness into sample_candidates, 72 bit randomness into sqr, and rejection randomness into rejection
    preprocessing_x4(sample_candidates.vec, sqr.vec, rejection.vec, &outbuf.vec[4]); // first four vectors are just signs

    // step 1: sample from CDT distribution, store all results
    sample_gauss16_x4(sample_candidates.vec);

    // step 2: construct the sample candidates, compute sqr and input to exp approximation
    sample_candidates_x4(sample_candidates.vec, sqr.vec, exp.vec);

    // step 3: approximate exponential
    approx_exp_4x(exp.vec);

    // step 4: compute rejection mask
    rej_msk_4x(rejection.vec, exp.vec, sample_candidates.vec);

    // step 5: sum up sqr
    sum_sqr_4x(&sqsum->limb48[0], sqr.vec, rejection.vec, &len.vec);

    // step 6: move accepted samples to correct memory positions
    move_to_mem_4x(r0,  samples, samples + N,samples + 2*N, len0-len0%N, 0, 0, 0, sample_candidates.vec, rejection.vec);

    ctr0 = len.arr[0] > 0 ? len0-len.arr[0] : len0;

    // copy remaining bytes
    bytecnt = POLY_HYPERBALL_NBLOCKS_4X * STREAM256_BLOCKBYTES - N/8 - NUM_GAUSSIANS * GAUSS_RAND_BYTES;
    if (ctr0 < len0)
    {
      uint8_t *bptr = &outbuf.coeffs[bytecnt + NUM_GAUSSIANS * GAUSS_RAND_BYTES + N/8];
      for (i = 0; i < bytecnt; i++)
      {
        outbuf2[0].coeffs[i] = *bptr++;
        if ((i+1)%8 == 0) {
          bptr += 3 * 8; // jump over the next vectors
        }
      }
    }

    while (ctr0 < len0  )
    {
        shake256x4_squeezeblocks(&outbuf2[0].coeffs[bytecnt], &outbuf2[1].coeffs[bytecnt], &outbuf2[2].coeffs[bytecnt], &outbuf2[3].coeffs[bytecnt], 1, &state);
        bytecnt += STREAM256_BLOCKBYTES;

        ctr0 += sample_gauss(r0 + ctr0, sqsum, outbuf2[0].coeffs, bytecnt, len0 - ctr0, len0%N);

        if (ctr0 < len0  )
        {
            size_t offset = (bytecnt / GAUSS_RAND_BYTES) * GAUSS_RAND_BYTES;
            size_t rem_bytes = bytecnt - offset;
            for (size_t i = 0; i < rem_bytes; i++)
            {
              outbuf2[0].coeffs[i] = outbuf2[0].coeffs[i + offset];
            }
            bytecnt = rem_bytes;
        }
    }
    renormalize(sqsum);
}
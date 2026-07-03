#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>
#include <string.h>
#include "fips202.h"
#include "fips202x4.h"

static uint64_t load64(const uint8_t x[8]) {
  unsigned int i;
  uint64_t r = 0;

  for (i = 0; i < 8; ++i) r |= (uint64_t) x[i] << 8 * i;

  return r;
}

static void keccakx4_absorb_once(__m256i s[25], unsigned int r, const uint8_t *in0, const uint8_t *in1,
                                 const uint8_t *in2, const uint8_t *in3, size_t inlen, uint8_t p) {
  size_t i;
  uint64_t pos = 0;
  __m256i t, idx;

  for (i = 0; i < 25; ++i) s[i] = _mm256_setzero_si256();

  idx = _mm256_set_epi64x((long long) in3, (long long) in2, (long long) in1, (long long) in0);
  while (inlen >= r) {
    for (i = 0; i < r / 8; ++i) {
      t = _mm256_i64gather_epi64((long long *) pos, idx, 1);
      s[i] = _mm256_xor_si256(s[i], t);
      pos += 8;
    }
    inlen -= r;

    f1600x4(s, KeccakF_RoundConstants);
  }

  for (i = 0; i < inlen / 8; ++i) {
    t = _mm256_i64gather_epi64((long long *) pos, idx, 1);
    s[i] = _mm256_xor_si256(s[i], t);
    pos += 8;
  }
  inlen -= 8 * i;

  if (inlen) {
    t = _mm256_i64gather_epi64((long long *) pos, idx, 1);
    idx = _mm256_set1_epi64x((1ULL << (8 * inlen)) - 1);
    t = _mm256_and_si256(t, idx);
    s[i] = _mm256_xor_si256(s[i], t);
  }

  t = _mm256_set1_epi64x((uint64_t) p << 8 * inlen);
  s[i] = _mm256_xor_si256(s[i], t);
  t = _mm256_set1_epi64x(1ULL << 63);
  s[r / 8 - 1] = _mm256_xor_si256(s[r / 8 - 1], t);
}

static void keccakx4_squeezeblocks(uint8_t *out0, uint8_t *out1, uint8_t *out2, uint8_t *out3, size_t nblocks,
                                   unsigned int r, __m256i s[25]) {
  unsigned int i;
  __m128d t;

  while (nblocks > 0) {
    f1600x4(s, KeccakF_RoundConstants);
    for (i = 0; i < r / 8; ++i) {
      t = _mm_castsi128_pd(_mm256_castsi256_si128(s[i]));
      _mm_storel_pd((__attribute__((__may_alias__)) double *) &out0[8 * i], t);
      _mm_storeh_pd((__attribute__((__may_alias__)) double *) &out1[8 * i], t);
      t = _mm_castsi128_pd(_mm256_extracti128_si256(s[i], 1));
      _mm_storel_pd((__attribute__((__may_alias__)) double *) &out2[8 * i], t);
      _mm_storeh_pd((__attribute__((__may_alias__)) double *) &out3[8 * i], t);
    }

    out0 += r;
    out1 += r;
    out2 += r;
    out3 += r;
    --nblocks;
  }
}

static void keccakx4_squeezeblocks_vec(__m256i *out, size_t nblocks, unsigned int r, __m256i s[25]) {
  unsigned int i;

  while (nblocks > 0) {
    f1600x4(s, KeccakF_RoundConstants);
    for (i = 0; i < r / 8; ++i) {
      out[i] = s[i];
    }

    out += r / 8;
    --nblocks;
  }
}

void shake128x4_absorb_once(keccakx4_state *state, const uint8_t *in0, const uint8_t *in1, const uint8_t *in2,
                            const uint8_t *in3, size_t inlen) {
  keccakx4_absorb_once(state->s, SHAKE128_RATE, in0, in1, in2, in3, inlen, 0x1F);
}

void shake128x4_squeezeblocks(uint8_t *out0, uint8_t *out1, uint8_t *out2, uint8_t *out3, size_t nblocks,
                              keccakx4_state *state) {
  keccakx4_squeezeblocks(out0, out1, out2, out3, nblocks, SHAKE128_RATE, state->s);
}

void shake256x4_absorb_once(keccakx4_state *state, const uint8_t *in0, const uint8_t *in1, const uint8_t *in2,
                            const uint8_t *in3, size_t inlen) {
  keccakx4_absorb_once(state->s, SHAKE256_RATE, in0, in1, in2, in3, inlen, 0x1F);
}

void shake256x4_squeezeblocks(uint8_t *out0, uint8_t *out1, uint8_t *out2, uint8_t *out3, size_t nblocks,
                              keccakx4_state *state) {
  keccakx4_squeezeblocks(out0, out1, out2, out3, nblocks, SHAKE256_RATE, state->s);
}

void shake256x4_squeezeblocks_vec(__m256i *buf, size_t nblocks, keccakx4_state *state) {
  keccakx4_squeezeblocks_vec(buf, nblocks, SHAKE256_RATE, state->s);
}

void shake128x4(uint8_t *out0, uint8_t *out1, uint8_t *out2, uint8_t *out3, size_t outlen, const uint8_t *in0,
                const uint8_t *in1, const uint8_t *in2, const uint8_t *in3, size_t inlen) {
  unsigned int i;
  size_t nblocks = outlen / SHAKE128_RATE;
  uint8_t t[4][SHAKE128_RATE];
  keccakx4_state state;

  shake128x4_absorb_once(&state, in0, in1, in2, in3, inlen);
  shake128x4_squeezeblocks(out0, out1, out2, out3, nblocks, &state);

  out0 += nblocks * SHAKE128_RATE;
  out1 += nblocks * SHAKE128_RATE;
  out2 += nblocks * SHAKE128_RATE;
  out3 += nblocks * SHAKE128_RATE;
  outlen -= nblocks * SHAKE128_RATE;

  if (outlen) {
    shake128x4_squeezeblocks(t[0], t[1], t[2], t[3], 1, &state);
    for (i = 0; i < outlen; ++i) {
      out0[i] = t[0][i];
      out1[i] = t[1][i];
      out2[i] = t[2][i];
      out3[i] = t[3][i];
    }
  }
}

void shake256x4(uint8_t *out0, uint8_t *out1, uint8_t *out2, uint8_t *out3, size_t outlen, const uint8_t *in0,
                const uint8_t *in1, const uint8_t *in2, const uint8_t *in3, size_t inlen) {
  unsigned int i;
  size_t nblocks = outlen / SHAKE256_RATE;
  uint8_t t[4][SHAKE256_RATE];
  keccakx4_state state;

  shake256x4_absorb_once(&state, in0, in1, in2, in3, inlen);
  shake256x4_squeezeblocks(out0, out1, out2, out3, nblocks, &state);

  out0 += nblocks * SHAKE256_RATE;
  out1 += nblocks * SHAKE256_RATE;
  out2 += nblocks * SHAKE256_RATE;
  out3 += nblocks * SHAKE256_RATE;
  outlen -= nblocks * SHAKE256_RATE;

  if (outlen) {
    shake256x4_squeezeblocks(t[0], t[1], t[2], t[3], 1, &state);
    for (i = 0; i < outlen; ++i) {
      out0[i] = t[0][i];
      out1[i] = t[1][i];
      out2[i] = t[2][i];
      out3[i] = t[3][i];
    }
  }
}


void ST_AVX2_shake128x4_squeezeblocks(uint8_t *out0, uint8_t *out1, uint8_t *out2, uint8_t *out3, int nblocks,
                                        keccakx4_state *state) {
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
  }
}

void ST_AVX2_shake256x4_squeezeblocks(uint8_t *out0, uint8_t *out1, uint8_t *out2, uint8_t *out3, int nblocks,
                                        keccakx4_state *state) {
  __m256i t0, t1, t2, t3, t4, t5, t6, t7;
  __m256i f0, f1, f2, f3, f4, f5, f6, f7;
  __m128i t;

  for (int i = 0; i < nblocks; ++i) {
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


    t = _mm256_castsi256_si128(state->s[16]);
    _mm_storeu_si64(out0, t);
    _mm_storeu_si64(out1, _mm_bsrli_si128(t, 8));
    t = _mm256_extracti128_si256(state->s[16], 1);
    _mm_storeu_si64(out2, t);
    _mm_storeu_si64(out3, _mm_bsrli_si128(t, 8));

    out0 += 8;
    out1 += 8;
    out2 += 8;
    out3 += 8;
  }
}

void shake256x4_squeezeblocks_x1(uint8_t *out0, int nblocks,
                                        keccakx4_state *state) {
  __m256i t0, t1, t2, t3, t4, t5, t6, t7;
  __m256i f0, f1, f2, f3, f4, f5, f6, f7;
  __m128i t;

  for (int i = 0; i < nblocks; ++i) {
    f1600x4(state->s, KeccakF_RoundConstants);

    t0 = _mm256_unpacklo_epi64(state->s[0], state->s[1]);
    t2 = _mm256_unpacklo_epi64(state->s[2], state->s[3]);

    t4 = _mm256_unpacklo_epi64(state->s[4], state->s[5]);
    t6 = _mm256_unpacklo_epi64(state->s[6], state->s[7]);

    f0 = _mm256_permute2x128_si256(t0, t2, 0x20);

    f4 = _mm256_permute2x128_si256(t4, t6, 0x20);

    _mm256_storeu_si256((__m256i *) out0, f0);
    _mm256_storeu_si256((__m256i *) (out0 + 32), f4);

    out0 += 64;

    t0 = _mm256_unpacklo_epi64(state->s[8], state->s[9]);
    t2 = _mm256_unpacklo_epi64(state->s[10], state->s[11]);

    t4 = _mm256_unpacklo_epi64(state->s[12], state->s[13]);
    t6 = _mm256_unpacklo_epi64(state->s[14], state->s[15]);

    f0 = _mm256_permute2x128_si256(t0, t2, 0x20);

    f4 = _mm256_permute2x128_si256(t4, t6, 0x20);

    _mm256_storeu_si256((__m256i *) out0, f0);

    _mm256_storeu_si256((__m256i *) (out0 + 32), f4);

    out0 += 64;

    t = _mm256_castsi256_si128(state->s[16]);
    _mm_storeu_si64(out0, t);

    out0 += 8;

  }
}

void store_statex4(uint8_t *out0, uint8_t *out1, uint8_t *out2, uint8_t *out3, keccakx4_state *state) {
  __m256i t0, t1, t2, t3, t4, t5, t6, t7;
  __m256i f0, f1, f2, f3, f4, f5, f6, f7;
  t0 = _mm256_unpacklo_epi64(state->s[17], state->s[18]);
  t1 = _mm256_unpackhi_epi64(state->s[17], state->s[18]);
  t2 = _mm256_unpacklo_epi64(state->s[19], state->s[20]);
  t3 = _mm256_unpackhi_epi64(state->s[19], state->s[20]);

  t4 = _mm256_unpacklo_epi64(state->s[21], state->s[22]);
  t5 = _mm256_unpackhi_epi64(state->s[21], state->s[22]);
  t6 = _mm256_unpacklo_epi64(state->s[23], state->s[24]);
  t7 = _mm256_unpackhi_epi64(state->s[23], state->s[24]);

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
}

void prepare_state_64(const uint8_t *seed, keccakx4_state64 *pre_state) {
  pre_state->s[0] = _mm256_set_epi64x(load64(seed), load64(seed), load64(seed), load64(seed));
  pre_state->s[1] = _mm256_set_epi64x(load64(seed + 8), load64(seed + 8), load64(seed + 8), load64(seed + 8));
  pre_state->s[2] = _mm256_set_epi64x(load64(seed + 16), load64(seed + 16), load64(seed + 16),load64(seed + 16));
  pre_state->s[3] = _mm256_set_epi64x(load64(seed + 24), load64(seed + 24), load64(seed + 24),load64(seed + 24));
  pre_state->s[4] = _mm256_set_epi64x(load64(seed + 32), load64(seed + 32), load64(seed + 32),load64(seed + 32));
  pre_state->s[5] = _mm256_set_epi64x(load64(seed + 40), load64(seed + 40), load64(seed + 40),load64(seed + 40));
  pre_state->s[6] = _mm256_set_epi64x(load64(seed + 48), load64(seed + 48), load64(seed + 48),load64(seed + 48));
  pre_state->s[7] = _mm256_set_epi64x(load64(seed + 56), load64(seed + 56), load64(seed + 56),load64(seed + 56));
  pre_state->s[8] = _mm256_set_epi64x(0x1ULL << 63, 0x1ULL << 63, 0x1ULL << 63, 0x1ULL << 63);
}

void init_state_64(keccakx4_state *state, const keccakx4_state64 *pre_state, uint16_t nonce0, uint16_t nonce1, uint16_t nonce2, uint16_t nonce3) {
   state->s[0] = pre_state->s[0];
   state->s[1] = pre_state->s[1];
   state->s[2] = pre_state->s[2];
   state->s[3] = pre_state->s[3];
   state->s[4] = pre_state->s[4];
   state->s[5] = pre_state->s[5];
   state->s[6] = pre_state->s[6];
   state->s[7] = pre_state->s[7];
   state->s[8] = _mm256_set_epi64x((0x1f << 16) ^ nonce3, (0x1f << 16) ^ nonce2, (0x1f << 16) ^ nonce1, (0x1f << 16) ^ nonce0);
   for (int i = 9; i < 25; ++i) state->s[i] = _mm256_setzero_si256();
   state->s[16] = pre_state->s[8];
}

void prepare_state_32(const uint8_t *seed, keccakx4_state32 *pre_state) {
  pre_state->s[0] = _mm256_set_epi64x(load64(seed), load64(seed), load64(seed), load64(seed));
  pre_state->s[1] = _mm256_set_epi64x(load64(seed + 8), load64(seed + 8), load64(seed + 8), load64(seed + 8));
  pre_state->s[2] = _mm256_set_epi64x(load64(seed + 16), load64(seed + 16), load64(seed + 16),load64(seed + 16));
  pre_state->s[3] = _mm256_set_epi64x(load64(seed + 24), load64(seed + 24), load64(seed + 24),load64(seed + 24));
  pre_state->s[4] = _mm256_set_epi64x(0x1ULL << 63, 0x1ULL << 63, 0x1ULL << 63, 0x1ULL << 63);
}

void init_state_32(keccakx4_state *state, const keccakx4_state32 *pre_state, uint16_t nonce0, uint16_t nonce1, uint16_t nonce2, uint16_t nonce3) {
  state->s[0] = pre_state->s[0];
  state->s[1] = pre_state->s[1];
  state->s[2] = pre_state->s[2];
  state->s[3] = pre_state->s[3];
  state->s[4] = _mm256_set_epi64x((0x1f << 16) ^ nonce3, (0x1f << 16) ^ nonce2, (0x1f << 16) ^ nonce1, (0x1f << 16) ^ nonce0);
  for (int i = 5; i < 25; ++i) state->s[i] = _mm256_setzero_si256();
  state->s[20] = pre_state->s[4];
}

void recover_statex4(uint8_t *in0, uint8_t *in1, uint8_t *in2, uint8_t *in3, keccakx4_state *state) {
  __m256i t0, t1, t2, t3, t4, t5, t6, t7;
  __m256i f0, f1, f2, f3, f4, f5, f6, f7;

  f0 = _mm256_loadu_si256((__m256i *)in0);
  f1 = _mm256_loadu_si256((__m256i *)in1);
  f2 = _mm256_loadu_si256((__m256i *)in2);
  f3 = _mm256_loadu_si256((__m256i *)in3);
  f4 = _mm256_loadu_si256((__m256i *)(in0 + 32));
  f5 = _mm256_loadu_si256((__m256i *)(in1 + 32));
  f6 = _mm256_loadu_si256((__m256i *)(in2 + 32));
  f7 = _mm256_loadu_si256((__m256i *)(in3 + 32));

  t0 = _mm256_unpacklo_epi64(f0, f1);
  t1 = _mm256_unpackhi_epi64(f0, f1);
  t2 = _mm256_unpacklo_epi64(f2, f3);
  t3 = _mm256_unpackhi_epi64(f2, f3);

  t4 = _mm256_unpacklo_epi64(f4, f5);
  t5 = _mm256_unpackhi_epi64(f4, f5);
  t6 = _mm256_unpacklo_epi64(f6, f7);
  t7 = _mm256_unpackhi_epi64(f6, f7);

  state->s[0] = _mm256_permute2x128_si256(t0, t2, 0x20);
  state->s[1] = _mm256_permute2x128_si256(t1, t3, 0x20);
  state->s[2] = _mm256_permute2x128_si256(t0, t2, 0x31);
  state->s[3] = _mm256_permute2x128_si256(t1, t3, 0x31);

  state->s[4] = _mm256_permute2x128_si256(t4, t6, 0x20);
  state->s[5] = _mm256_permute2x128_si256(t5, t7, 0x20);
  state->s[6] = _mm256_permute2x128_si256(t4, t6, 0x31);
  state->s[7] = _mm256_permute2x128_si256(t5, t7, 0x31);

  in0 += 64;
  in1 += 64;
  in2 += 64;
  in3 += 64;

  f0 = _mm256_loadu_si256((__m256i *)in0);
  f1 = _mm256_loadu_si256((__m256i *)in1);
  f2 = _mm256_loadu_si256((__m256i *)in2);
  f3 = _mm256_loadu_si256((__m256i *)in3);
  f4 = _mm256_loadu_si256((__m256i *)(in0 + 32));
  f5 = _mm256_loadu_si256((__m256i *)(in1 + 32));
  f6 = _mm256_loadu_si256((__m256i *)(in2 + 32));
  f7 = _mm256_loadu_si256((__m256i *)(in3 + 32));

  t0 = _mm256_unpacklo_epi64(f0, f1);
  t1 = _mm256_unpackhi_epi64(f0, f1);
  t2 = _mm256_unpacklo_epi64(f2, f3);
  t3 = _mm256_unpackhi_epi64(f2, f3);

  t4 = _mm256_unpacklo_epi64(f4, f5);
  t5 = _mm256_unpackhi_epi64(f4, f5);
  t6 = _mm256_unpacklo_epi64(f6, f7);
  t7 = _mm256_unpackhi_epi64(f6, f7);

  state->s[8] = _mm256_permute2x128_si256(t0, t2, 0x20);
  state->s[9] = _mm256_permute2x128_si256(t1, t3, 0x20);
  state->s[10] = _mm256_permute2x128_si256(t0, t2, 0x31);
  state->s[11] = _mm256_permute2x128_si256(t1, t3, 0x31);

  state->s[12] = _mm256_permute2x128_si256(t4, t6, 0x20);
  state->s[13] = _mm256_permute2x128_si256(t5, t7, 0x20);
  state->s[14] = _mm256_permute2x128_si256(t4, t6, 0x31);
  state->s[15] = _mm256_permute2x128_si256(t5, t7, 0x31);

  in0 += 64;
  in1 += 64;
  in2 += 64;
  in3 += 64;

  f0 = _mm256_loadu_si256((__m256i *)in0);
  f1 = _mm256_loadu_si256((__m256i *)in1);
  f2 = _mm256_loadu_si256((__m256i *)in2);
  f3 = _mm256_loadu_si256((__m256i *)in3);
  f4 = _mm256_loadu_si256((__m256i *)(in0 + 32));
  f5 = _mm256_loadu_si256((__m256i *)(in1 + 32));
  f6 = _mm256_loadu_si256((__m256i *)(in2 + 32));
  f7 = _mm256_loadu_si256((__m256i *)(in3 + 32));

  t0 = _mm256_unpacklo_epi64(f0, f1);
  t1 = _mm256_unpackhi_epi64(f0, f1);
  t2 = _mm256_unpacklo_epi64(f2, f3);
  t3 = _mm256_unpackhi_epi64(f2, f3);

  t4 = _mm256_unpacklo_epi64(f4, f5);
  t5 = _mm256_unpackhi_epi64(f4, f5);
  t6 = _mm256_unpacklo_epi64(f6, f7);
  t7 = _mm256_unpackhi_epi64(f6, f7);

  state->s[16] = _mm256_permute2x128_si256(t0, t2, 0x20);
  state->s[17] = _mm256_permute2x128_si256(t1, t3, 0x20);
  state->s[18] = _mm256_permute2x128_si256(t0, t2, 0x31);
  state->s[19] = _mm256_permute2x128_si256(t1, t3, 0x31);

  state->s[20] = _mm256_permute2x128_si256(t4, t6, 0x20);
  state->s[21] = _mm256_permute2x128_si256(t5, t7, 0x20);
  state->s[22] = _mm256_permute2x128_si256(t4, t6, 0x31);
  state->s[23] = _mm256_permute2x128_si256(t5, t7, 0x31);

  in0 += 64;
  in1 += 64;
  in2 += 64;
  in3 += 64;

  state->s[24] = _mm256_set_epi64x(load64(in3), load64(in2), load64(in1), load64(in0));
}

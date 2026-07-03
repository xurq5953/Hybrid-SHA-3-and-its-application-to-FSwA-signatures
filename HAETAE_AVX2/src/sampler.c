#include "sampler.h"
#include "fixpoint.h"
#include "symmetric.h"
#include <stdint.h>
#include <stdio.h>

#include "fips202x4.h"
#include "hybrid.h"

/*************************************************
 * Name:        rej_uniform
 *
 * Description: Sample uniformly random coefficients in [0, Q-1] by
 *              performing rejection sampling on array of random bytes.
 *
 * Arguments:   - int32_t *a: pointer to output array (allocated)
 *              - unsigned int len: number of coefficients to be sampled
 *              - const uint8_t *buf: array of random bytes
 *              - unsigned int buflen: length of array of random bytes
 *
 * Returns number of sampled coefficients. Can be smaller than len if not enough
 * random bytes were given.
 **************************************************/
unsigned int rej_uniform(int32_t *a, unsigned int len, const uint8_t *buf,
                         unsigned int buflen) {
    unsigned int ctr, pos;
    uint32_t t;

    ctr = pos = 0;
    while (ctr < len && pos + 2 <= buflen) {
        t = buf[pos++];
        t |= (uint32_t)buf[pos++] << 8;

        if (t < Q)
            a[ctr++] = t;
    }
    return ctr;
}

/*************************************************
 * Name:        rej_eta
 *
 * Description: Sample uniformly random coefficients in [-ETA, ETA] by
 *              performing rejection sampling on array of random bytes.
 *
 * Arguments:   - int32_t *a: pointer to output array (allocated)
 *              - unsigned int len: number of coefficients to be sampled
 *              - const uint8_t *buf: array of random bytes
 *              - unsigned int buflen: length of array of random bytes
 *
 * Returns number of sampled coefficients. Can be smaller than len if not enough
 * random bytes were given.
 **************************************************/
static int32_t mod3(uint8_t t) {
    int32_t r;
    r = (t >> 4) + (t & 0xf);
    r = (r >> 2) + (r & 3);
    r = (r >> 2) + (r & 3);
    r = (r >> 2) + (r & 3);
    return r - (3 * (r >> 1));
}

static __m256i mod3x8(__m256i t) {
    __m256i r;
    __m256i f0,f1,f2,f3;
    const __m256i maskf = _mm256_set1_epi32(0xf);
    const __m256i mask3 = _mm256_set1_epi32(3);
    f0 = _mm256_srli_epi32(t,4);
    f1 = _mm256_and_si256(t,maskf);
    r = _mm256_add_epi32(f0,f1);
    f0 = _mm256_srli_epi32(r,2);
    f1 = _mm256_and_si256(r,mask3);
    r = _mm256_add_epi32(f0,f1);
    f0 = _mm256_srli_epi32(r,2);
    f1 = _mm256_and_si256(r,mask3);
    r = _mm256_add_epi32(f0,f1);
    f0 = _mm256_srli_epi32(r,2);
    f1 = _mm256_and_si256(r,mask3);
    r = _mm256_add_epi32(f0,f1);
    f0 = _mm256_srli_epi32(r,1);
    f0 = _mm256_mullo_epi32(f0, mask3);
    r = _mm256_sub_epi32(r,f0);
    return r;
}

static int32_t mod3_leq26(uint8_t t) {
    int32_t r;
    r = (t >> 4) + (t & 0xf);
    r = (r >> 2) + (r & 3);
    r = (r >> 2) + (r & 3);
    return r - (3 * (r >> 1));
}

static __m256i mod3_leq26x8(__m256i t) {
    __m256i r;
    __m256i f0,f1;
    const __m256i maskf = _mm256_set1_epi32(0xf);
    const __m256i mask3 = _mm256_set1_epi32(3);
    f0 = _mm256_srli_epi32(t,4);
    f1 = _mm256_and_si256(t,maskf);
    r = _mm256_add_epi32(f0,f1);
    f0 = _mm256_srli_epi32(r,2);
    f1 = _mm256_and_si256(r,mask3);
    r = _mm256_add_epi32(f0,f1);
    f0 = _mm256_srli_epi32(r,2);
    f1 = _mm256_and_si256(r,mask3);
    r = _mm256_add_epi32(f0,f1);
    f0 = _mm256_srli_epi32(r,1);
    f0 = _mm256_mullo_epi32(f0, mask3);
    r = _mm256_sub_epi32(r,f0);
    return r;
}

static int32_t mod3_leq8(uint8_t t) {
    int32_t r;
    r = (t >> 2) + (t & 3);
    r = (r >> 2) + (r & 3);
    return r - (3 * (r >> 1));
}

static __m256i mod3_leq8x8(__m256i t) {
    __m256i r;
    __m256i f0,f1;
    const __m256i maskf = _mm256_set1_epi32(0xf);
    const __m256i mask3 = _mm256_set1_epi32(3);
    f0 = _mm256_srli_epi32(t,2);
    f1 = _mm256_and_si256(t,mask3);
    r = _mm256_add_epi32(f0,f1);
    f0 = _mm256_srli_epi32(r,2);
    f1 = _mm256_and_si256(r,mask3);
    r = _mm256_add_epi32(f0,f1);
    f0 = _mm256_srli_epi32(r,1);
    f0 = _mm256_mullo_epi32(f0, mask3);
    r = _mm256_sub_epi32(r,f0);
    return r;
}
unsigned int rej_eta(int32_t *a, unsigned int len, const uint8_t *buf,
                     unsigned int buflen) {
    unsigned int ctr, pos;

    ctr = pos = 0;
    while ((ctr < len - 5) && pos < buflen) {
        uint32_t t = buf[pos++];
        if (t < 243) {
            // reduce mod 3
            a[ctr++] = mod3(t);

            t *= 171; // 171*3 = 1 mod 256
            t >>= 9;
            a[ctr++] = mod3(t);

            t *= 171;
            t >>= 9;
            a[ctr++] = mod3_leq26(t);

            t *= 171;
            t >>= 9;
            a[ctr++] = mod3_leq8(t);

            t *= 171;
            t >>= 9;
            a[ctr++] = (int32_t)t - (int32_t)3 * (t >> 1);
        }
    }
    while (ctr < len && pos < buflen) {
        uint32_t t = buf[pos++];
        if (t < 243) {
            // reduce mod 3
            a[ctr++] = mod3(t);

            if (ctr >= len)
                break;

            t *= 171; // 171*3 = 1 mod 256
            t >>= 9;
            a[ctr++] = mod3(t);

            if (ctr >= len)
                break;

            t *= 171;
            t >>= 9;
            a[ctr++] = mod3_leq26(t);

            if (ctr >= len)
                break;

            t *= 171;
            t >>= 9;
            a[ctr++] = mod3_leq8(t);

            if (ctr >= len)
                break;

            t *= 171;
            t >>= 9;
            a[ctr++] = (int32_t)t - (int32_t)3 * (t >> 1);
        }
    }
    return ctr;
}

static inline void _mm256_store_si32(int32_t *a, __m256i f) {
    _mm_storeu_si32(a,_mm256_castsi256_si128(f));
}

unsigned int rej_eta_avx2(int32_t *a, unsigned int len, const uint8_t *buf,
                     unsigned int buflen) {
    unsigned int ctr, pos;
    __m256i f,g;
    __m256i f0,f1,f2,f3,f4;
    __m128i g0,g1,g2,g3,g4;
    const __m256i mask171 = _mm256_set1_epi32(171);
    const __m256i mask3 = _mm256_set1_epi32(3);
    ctr = pos = 0;
    __attribute__ ((aligned(32))) uint32_t z[8];
    while ((ctr < len - 40) && pos < buflen) {
        int i = 0;
        while (i < 8) {
            uint32_t t = buf[pos++];
            if (t < 243) {
                z[i++] = t;
            }
        }
        f = _mm256_load_si256((__m256i *)z);
        f0 = mod3x8(f);
        f = _mm256_mullo_epi32(f, mask171);
        f = _mm256_srli_epi32(f, 9);
        f1 = mod3x8(f);
        f = _mm256_mullo_epi32(f, mask171);
        f = _mm256_srli_epi32(f, 9);
        f2 = mod3_leq26x8(f);
        f = _mm256_mullo_epi32(f, mask171);
        f = _mm256_srli_epi32(f, 9);
        f3 = mod3_leq8x8(f);
        f = _mm256_mullo_epi32(f, mask171);
        f = _mm256_srli_epi32(f, 9);
        g = _mm256_srli_epi32(f,1);
        g = _mm256_mullo_epi32(g, mask3);
        f4 = _mm256_sub_epi32(f,g);

        g0 = _mm256_extracti128_si256(f0,1);
        g1 = _mm256_extracti128_si256(f1,1);
        g2 = _mm256_extracti128_si256(f2,1);
        g3 = _mm256_extracti128_si256(f3,1);
        g4 = _mm256_extracti128_si256(f4,1);

        _mm256_store_si32(a + ctr ,f0);
        _mm256_store_si32(a + ctr + 1,f1);
        _mm256_store_si32(a + ctr + 2,f2);
        _mm256_store_si32(a + ctr + 3,f3);
        _mm256_store_si32(a + ctr + 4,f4);

        f0 = _mm256_bsrli_epi128(f0, 4);
        f1 = _mm256_bsrli_epi128(f1, 4);
        f2 = _mm256_bsrli_epi128(f2, 4);
        f3 = _mm256_bsrli_epi128(f3, 4);
        f4 = _mm256_bsrli_epi128(f4, 4);

        _mm256_store_si32(a + ctr + 5,f0);
        _mm256_store_si32(a + ctr + 6,f1);
        _mm256_store_si32(a + ctr + 7,f2);
        _mm256_store_si32(a + ctr + 8,f3);
        _mm256_store_si32(a + ctr + 9,f4);

        f0 = _mm256_bsrli_epi128(f0, 4);
        f1 = _mm256_bsrli_epi128(f1, 4);
        f2 = _mm256_bsrli_epi128(f2, 4);
        f3 = _mm256_bsrli_epi128(f3, 4);
        f4 = _mm256_bsrli_epi128(f4, 4);

        _mm256_store_si32(a + ctr + 10,f0);
        _mm256_store_si32(a + ctr + 11,f1);
        _mm256_store_si32(a + ctr + 12,f2);
        _mm256_store_si32(a + ctr + 13,f3);
        _mm256_store_si32(a + ctr + 14,f4);

        f0 = _mm256_bsrli_epi128(f0, 4);
        f1 = _mm256_bsrli_epi128(f1, 4);
        f2 = _mm256_bsrli_epi128(f2, 4);
        f3 = _mm256_bsrli_epi128(f3, 4);
        f4 = _mm256_bsrli_epi128(f4, 4);

        _mm256_store_si32(a + ctr + 15,f0);
        _mm256_store_si32(a + ctr + 16,f1);
        _mm256_store_si32(a + ctr + 17,f2);
        _mm256_store_si32(a + ctr + 18,f3);
        _mm256_store_si32(a + ctr + 19,f4);


        _mm_storeu_si32(a + ctr + 20, g0);
        _mm_storeu_si32(a + ctr + 21, g1);
        _mm_storeu_si32(a + ctr + 22, g2);
        _mm_storeu_si32(a + ctr + 23, g3);
        _mm_storeu_si32(a + ctr + 24, g4);

        g0 = _mm_bsrli_si128(g0, 4);
        g1 = _mm_bsrli_si128(g1, 4);
        g2 = _mm_bsrli_si128(g2, 4);
        g3 = _mm_bsrli_si128(g3, 4);
        g4 = _mm_bsrli_si128(g4, 4);

        _mm_storeu_si32(a + ctr + 25,g0);
        _mm_storeu_si32(a + ctr + 26,g1);
        _mm_storeu_si32(a + ctr + 27,g2);
        _mm_storeu_si32(a + ctr + 28,g3);
        _mm_storeu_si32(a + ctr + 29,g4);

        g0 = _mm_bsrli_si128(g0, 4);
        g1 = _mm_bsrli_si128(g1, 4);
        g2 = _mm_bsrli_si128(g2, 4);
        g3 = _mm_bsrli_si128(g3, 4);
        g4 = _mm_bsrli_si128(g4, 4);

        _mm_storeu_si32(a + ctr + 30,g0);
        _mm_storeu_si32(a + ctr + 31,g1);
        _mm_storeu_si32(a + ctr + 32,g2);
        _mm_storeu_si32(a + ctr + 33,g3);
        _mm_storeu_si32(a + ctr + 34,g4);

        g0 = _mm_bsrli_si128(g0, 4);
        g1 = _mm_bsrli_si128(g1, 4);
        g2 = _mm_bsrli_si128(g2, 4);
        g3 = _mm_bsrli_si128(g3, 4);
        g4 = _mm_bsrli_si128(g4, 4);

        _mm_storeu_si32(a + ctr + 35,g0);
        _mm_storeu_si32(a + ctr + 36,g1);
        _mm_storeu_si32(a + ctr + 37,g2);
        _mm_storeu_si32(a + ctr + 38,g3);
        _mm_storeu_si32(a + ctr + 39,g4);

        ctr+=40;
    }

    while ((ctr < len - 5) && pos < buflen) {
        uint32_t t = buf[pos++];
        if (t < 243) {
            // reduce mod 3
            a[ctr++] = mod3(t);

            t *= 171; // 171*3 = 1 mod 256
            t >>= 9;
            a[ctr++] = mod3(t);

            t *= 171;
            t >>= 9;
            a[ctr++] = mod3_leq26(t);

            t *= 171;
            t >>= 9;
            a[ctr++] = mod3_leq8(t);

            t *= 171;
            t >>= 9;
            a[ctr++] = (int32_t)t - (int32_t)3 * (t >> 1);
        }
    }
    while (ctr < len && pos < buflen) {
        uint32_t t = buf[pos++];
        if (t < 243) {
            // reduce mod 3
            a[ctr++] = mod3(t);

            if (ctr >= len)
                break;

            t *= 171; // 171*3 = 1 mod 256
            t >>= 9;
            a[ctr++] = mod3(t);

            if (ctr >= len)
                break;

            t *= 171;
            t >>= 9;
            a[ctr++] = mod3_leq26(t);

            if (ctr >= len)
                break;

            t *= 171;
            t >>= 9;
            a[ctr++] = mod3_leq8(t);

            if (ctr >= len)
                break;

            t *= 171;
            t >>= 9;
            a[ctr++] = (int32_t)t - (int32_t)3 * (t >> 1);
        }
    }
    return ctr;
}

static uint64_t approx_exp(const uint64_t x) {
    int64_t result;
    result = -0x0000B6C6340925AELL;
    result = ((smulh48(result, x) + (1LL << 2)) >> 3) + 0x0000B4BD4DF85227LL;
    result = ((smulh48(result, x) + (1LL << 2)) >> 3) - 0x0000887F727491E2LL;
    result = ((smulh48(result, x) + (1LL << 1)) >> 2) + 0x0000AAAA643C7E8DLL;
    result = ((smulh48(result, x) + (1LL << 1)) >> 2) - 0x0000AAAAA98179E6LL;
    result = ((smulh48(result, x) + 1LL) >> 1) + 0x0000FFFFFFFB2E7ALL;
    result = ((smulh48(result, x) + 1LL) >> 1) - 0x0000FFFFFFFFF85FLL;
    result = ((smulh48(result, x))) + 0x0000FFFFFFFFFFFCLL;
    return result;
}

#define CDTLEN 64
static const uint32_t CDT[CDTLEN] = {
 3266,  6520,  9748, 12938, 16079, 19159, 22168, 25096,
27934, 30674, 33309, 35833, 38241, 40531, 42698, 44742,
46663, 48460, 50135, 51690, 53128, 54454, 55670, 56781,
57794, 58712, 59541, 60287, 60956, 61554, 62085, 62556,
62972, 63337, 63657, 63936, 64178, 64388, 64569, 64724,
64857, 64970, 65066, 65148, 65216, 65273, 65321, 65361,
65394, 65422, 65444, 65463, 65478, 65490, 65500, 65508,
65514, 65519, 65523, 65527, 65529, 65531, 65533, 65534
};


static uint64_t sample_gauss16(const uint64_t rand16) {
    unsigned int i;
    uint64_t r = 0;
    for (i = 0; i < CDTLEN; i++) {
        r += (((uint64_t)CDT[i] - rand16) >> 63) & 1;
    }
    return r;
}

#define GAUSS_RAND (72 + 16 + 48)
#define GAUSS_RAND_BYTES ((GAUSS_RAND + 7) / 8)
static int sample_gauss_sigma76(uint64_t *r, fp96_76 *sqr,
                                const uint8_t rand[GAUSS_RAND_BYTES]) {
    const uint64_t rand_gauss16 = rand[0] | (((uint64_t) rand[1]) << 8); 
    const uint64_t rand_rej = rand[2] | (((uint64_t) rand[3]) << 8) | (((uint64_t) rand[4]) << 16) | (((uint64_t) rand[5]) << 24)
     | (((uint64_t) rand[6]) << 32) | (((uint64_t) rand[7]) << 40);
    uint64_t x, exp_in;
    fp96_76 y;

    // sample x
    x = sample_gauss16(rand_gauss16);

    // y := append x to y
    // leave 16 bit for carries
    y.limb48[0] = rand[8] | ((uint64_t)rand[9] << 8) |
                  ((uint64_t)rand[10] << 16) | ((uint64_t)rand[11] << 24) |
                  ((uint64_t)rand[12] << 32) | ((uint64_t)rand[13] << 40);
    y.limb48[1] =
        rand[14] | ((uint64_t)rand[15] << 8) | ((uint64_t)rand[16] << 16) |
        (x << 24);

    // r := round y 
    *r = (y.limb48[0] >> 15) ^ (y.limb48[1] << 33);
    *r += 1; // rounding
    *r >>= 1;

    // sqr := y*y
    fixpoint_square(sqr, &y);

    // sqr[1] = y^2 >> (76+48)                // 34 bit
    // sqr[0] = (y^2 >> 76) & ((1UL<<48)-1)   // 48 bit
    // exp_in := sqr - ((x*x) << 68)
    exp_in = sqr->limb48[1] - ((x*x) << (68 - 48));
    exp_in <<= 20;
    exp_in |= sqr->limb48[0] >> 28;
    exp_in += 1; // rounding
    exp_in >>= 1;

    return ((((int64_t)(rand_rej ^
                        (rand_rej & 1)) // set lowest bit to zero in order to
                                        // use it for rejection if sample==0
              - (int64_t)approx_exp(exp_in)) >>
             63) // reject with prob 1-approx_exp(exp_in)
            & (((*r | -*r) >> 63) | rand_rej)) &
           1; // if the sample is zero, clear the return value with prob 1/2
}

int sample_gauss(uint64_t *r, fp96_76 *sqsum, const uint8_t *buf, const size_t buflen, const size_t len, const int dont_write_last)
{
    const uint8_t *pos = buf;
    fp96_76 sqr;
    size_t bytecnt = buflen, coefcnt = 0, cnt = 0;
    int accepted;
    uint64_t dummy;
    
    while (coefcnt < len) {
        if (bytecnt < GAUSS_RAND_BYTES) {
          renormalize(sqsum);
          return coefcnt;
        }

        if (dont_write_last && coefcnt == len-1)
        {
          accepted = sample_gauss_sigma76(&dummy, &sqr, pos);
        } else {
          accepted = sample_gauss_sigma76(&r[coefcnt], &sqr, pos);
        }
        cnt += 1;
        coefcnt += accepted;
        pos += GAUSS_RAND_BYTES;
        bytecnt -= GAUSS_RAND_BYTES;

        sqsum->limb48[0] += sqr.limb48[0] & -(int64_t)accepted;
        sqsum->limb48[1] += sqr.limb48[1] & -(int64_t)accepted;
    }

    renormalize(sqsum);
    return len;
}

#define POLY_HYPERBALL_BUFLEN (GAUSS_RAND_BYTES * N)
#define POLY_HYPERBALL_NBLOCKS ((POLY_HYPERBALL_BUFLEN + STREAM256_BLOCKBYTES - 1) / STREAM256_BLOCKBYTES)
void sample_gauss_N(uint64_t *r, uint8_t *signs, fp96_76 *sqsum,
                    const uint8_t seed[CRHBYTES], const uint16_t nonce,
                    const size_t len) {
    uint8_t buf[POLY_HYPERBALL_NBLOCKS * STREAM256_BLOCKBYTES];
    size_t bytecnt, coefcnt, firstflag = 1;
    stream256_state state;
    stream256_init(&state, seed, nonce);

    stream256_squeezeblocks(buf, POLY_HYPERBALL_NBLOCKS, &state);
    for (size_t i = 0; i < len / 8; i++) {
        signs[i] = buf[i];
    }
    bytecnt = POLY_HYPERBALL_NBLOCKS * STREAM256_BLOCKBYTES - len / 8;
    coefcnt = sample_gauss(r, sqsum, buf + len / 8, bytecnt, len, len%N);
    while (coefcnt < len) {
        size_t off = bytecnt % GAUSS_RAND_BYTES;
        for (size_t i = 0; i < off; i++) {
            buf[i] = buf[bytecnt + len/8*firstflag - off + i];
        }
        stream256_squeezeblocks(buf + off, 1, &state);
        bytecnt = STREAM256_BLOCKBYTES + off;

        coefcnt += sample_gauss(r + coefcnt, sqsum, buf, bytecnt, len - coefcnt, len%N);
        firstflag = 0;
    }
}

void sample_gauss_RejS_N(loop_queue *loop,uint64_t *r, uint8_t *signs, fp96_76 *sqsum,
                    size_t len) {
    size_t bytecnt, coefcnt, firstflag = 1;
    uint8_t *y_buff;
    y_buff = loop_dequeue(loop);

    for (size_t i = 0; i < len / 8; i++) {
        signs[i] = y_buff[i];
    }
    bytecnt = LOOP_BUF_LENGTH - len / 8;
    coefcnt = sample_gauss(r, sqsum, y_buff + len / 8, bytecnt, len, 0);
    while (coefcnt < len) {
        uint8_t buf[POLY_HYPERBALL_NBLOCKS * STREAM256_BLOCKBYTES];
        size_t off = bytecnt % GAUSS_RAND_BYTES;
        keccakx4_state state;
        for (size_t i = 0; i < off; i++) {
            buf[i] = buf[bytecnt + len/8*firstflag - off + i];
        }

        shake256x4_squeezeblocks_x1(buf + off, 1, &state);
        bytecnt = STREAM256_BLOCKBYTES + off;

        coefcnt += sample_gauss(r + coefcnt, sqsum, buf, bytecnt, len - coefcnt, len%N);
        firstflag = 0;
    }
}

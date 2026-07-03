#include <stdio.h>
#include "pspm.h"
#include "polyvec.h"
#include "params.h"
#include <immintrin.h>
#include <string.h>

#if ETA == 2
void poly_emulate_cs(poly *r, const poly *c, sword s1_table[N * 3]) {
    ALIGN(64) uint8_t w[N];
    uint8_t *s;
    __m512i w512, s512, f0, f1, f2, f3, t;

    memset(w, 0, N);
    for (int i = 0; i < N; i++) {
        if (c->coeffs[i] != 0) {
            s = s1_table + N - i + (N & (c->coeffs[i] >> 31));
            for (int j = 0; j < N / 64; j++) {
                w512 = _mm512_loadu_si512(w + j * 64);
                s512 = _mm512_loadu_si512(s + j * 64);
                w512 = _mm512_add_epi8(w512, s512);
                _mm512_storeu_si512(w + j * 64, w512);
            }
        }
    }
    for (int i = 0; i < N / 64; i++) {
        w512 = _mm512_loadu_si512(w + i * 64);

        f0 = _mm512_cvtepi8_epi32(_mm512_extracti64x2_epi64(w512, 0));
        f1 = _mm512_cvtepi8_epi32(_mm512_extracti64x2_epi64(w512, 1));
        f2 = _mm512_cvtepi8_epi32(_mm512_extracti64x2_epi64(w512, 2));
        f3 = _mm512_cvtepi8_epi32(_mm512_extracti64x2_epi64(w512, 3));

        _mm512_storeu_si512(r->coeffs + 64 * i, f0);
        _mm512_storeu_si512(r->coeffs + 64 * i + 16, f1);
        _mm512_storeu_si512(r->coeffs + 64 * i + 32, f2);
        _mm512_storeu_si512(r->coeffs + 64 * i + 48, f3);
    }
}
#else

void poly_emulate_cs(poly *r, const poly *c, sword s2_table[N * 3]) {
    ALIGN(64) uint16_t w[N];
    uint16_t *s;
    __m512i w512, s512;

    memset(w, 0, N * 2);
    for (int i = 0; i < N; i++) {
        if (c->coeffs[i] != 0) {
            s = s2_table + N - i + (N & (c->coeffs[i] >> 31));
            for (int j = 0; j < N / 32; j++) {
                w512 = _mm512_loadu_si512(w + j * 32);
                s512 = _mm512_loadu_si512(s + j * 32);
                w512 = _mm512_add_epi16(w512, s512);
                _mm512_storeu_si512(w + j * 32, w512);
            }
        }
    }
    for (int i = 0; i < N / 32; i++) {
        w512 = _mm512_loadu_si512(w + i * 32);
        s512 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(w512, 0));
        w512 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(w512, 1));
        _mm512_storeu_si512(r->coeffs + i * 32, s512);
        _mm512_storeu_si512(r->coeffs + i * 32 + 16, w512);
    }
}
#endif



#if ETA == 2
void emulate_cs1(polyvecl *r, const poly *c, uint8_t s1_table[L][N * 3])
{
    ALIGN(64) uint8_t w[N];
    uint8_t *s;
    __m256i w256, s256, f0, f1, f2, f3, t;

    for (int l = 0; l < L; l++)
    {
        memset(w, 0, N);
        for (int i = 0; i < N; i++)
        {
            if (c->coeffs[i] != 0)
            {
                s = s1_table[l] + N - i + (N & (c->coeffs[i] >> 31));
                for (int j = 0; j < N / 32; j++)
                {
                    w256 = _mm256_loadu_si256(w + j * 32);
                    s256 = _mm256_loadu_si256(s + j * 32);
                    w256 = _mm256_add_epi8(w256, s256);
                    _mm256_storeu_si256(w + j * 32, w256);
                }
            }
        }
        for (int i = 0; i < N / 32; i++)
        {
            w256 = _mm256_loadu_si256(w + i * 32);
            t = _mm256_bsrli_epi128(w256, 8);

            f0 = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(w256));
            f1 = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(t));
            f2 = _mm256_cvtepi8_epi32(_mm256_extractf128_si256(w256, 1));
            f3 = _mm256_cvtepi8_epi32(_mm256_extractf128_si256(t, 1));

            _mm256_storeu_si256(r->vec[l].coeffs + 32 * i, f0);
            _mm256_storeu_si256(r->vec[l].coeffs + 32 * i + 8, f1);
            _mm256_storeu_si256(r->vec[l].coeffs + 32 * i + 16, f2);
            _mm256_storeu_si256(r->vec[l].coeffs + 32 * i + 24, f3);
        }
    }
}
#endif


#if ETA == 4

void emulate_cs2(polyveck *r, const poly *c, uint16_t s2_table[K][N * 3])
{
    ALIGN(64) uint16_t w[N];
    uint16_t *s;
    __m256i w256, s256, f0, f1, f2, f3, t;

    for (int k = 0; k < K; k++)
    {
        memset(w, 0, N * 2);
        for (int i = 0; i < N; i++)
        {
            if (c->coeffs[i] != 0)
            {
                s = s2_table[k] + N - i + (N & (c->coeffs[i] >> 31));
                for (int j = 0; j < N / 16; j++)
                {
                    w256 = _mm256_loadu_si256(w + j * 16);
                    s256 = _mm256_loadu_si256(s + j * 16);
                    w256 = _mm256_add_epi16(w256, s256);
                    _mm256_storeu_si256(w + j * 16, w256);
                }
            }
        }
        for (int i = 0; i < N / 16; i++)
        {
            w256 = _mm256_loadu_si256(w + i * 16);
            s256 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w256,0));
            w256 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w256,1));
            _mm256_storeu_si256(r->vec[k].coeffs + i * 16, s256);
            _mm256_storeu_si256(r->vec[k].coeffs + i * 16 + 8, w256);
        }
    }
}

#else

void emulate_cs2(polyveck *r, const poly *c, uint8_t s2_table[K][N * 3])
{
    ALIGN(64) uint8_t w[N];
    uint8_t *s;
    __m256i w256, s256, f0, f1, f2, f3, t;

    for (int l = 0; l < K; l++)
    {
        memset(w, 0, N);
        for (int i = 0; i < N; i++)
        {
            if (c->coeffs[i] != 0)
            {
                s = s2_table[l] + N - i + (N & (c->coeffs[i] >> 31));
                for (int j = 0; j < N / 32; j++)
                {
                    w256 = _mm256_loadu_si256(w + j * 32);
                    s256 = _mm256_loadu_si256(s + j * 32);
                    w256 = _mm256_add_epi8(w256, s256);
                    _mm256_storeu_si256(w + j * 32, w256);
                }
            }
        }
        for (int i = 0; i < N / 32; i++)
        {
            w256 = _mm256_loadu_si256(w + i * 32);
            t = _mm256_bsrli_epi128(w256, 8);

            f0 = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(w256));
            f1 = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(t));
            f2 = _mm256_cvtepi8_epi32(_mm256_extractf128_si256(w256, 1));
            f3 = _mm256_cvtepi8_epi32(_mm256_extractf128_si256(t, 1));

            _mm256_storeu_si256(r->vec[l].coeffs + 32 * i, f0);
            _mm256_storeu_si256(r->vec[l].coeffs + 32 * i + 8, f1);
            _mm256_storeu_si256(r->vec[l].coeffs + 32 * i + 16, f2);
            _mm256_storeu_si256(r->vec[l].coeffs + 32 * i + 24, f3);
        }
    }
}
#endif


int emulate_ct(polyveck *r, const poly *c, const polyveck *t)
{
    ALIGN(64) int32_t stable[N * 3];

    int32_t *stable32;
    __m256i w256, s256, f0, f1, f2, f3;
    const __m256i zero = _mm256_setzero_si256();

    for (int k = 0; k < K; k++)
    {
        int32_t *w = r->vec[k].coeffs;
        memset(w, 0, N * 4);
        for (int i = 0; i < N / 8; i++)
        {
            f0 = _mm256_load_si256(t->vec[k].coeffs + i * 8);
            f1 = _mm256_sub_epi32(zero, f0);
            _mm256_store_si256(stable + i * 8, f1);
            _mm256_store_si256(stable + N * 2 + i * 8, f1);
            _mm256_store_si256(stable + N + i * 8, f0);
        }

        for (int i = 0; i < N; i++)
        {
            if (c->coeffs[i] != 0)
            {
                stable32 = stable + N - i + (N & (c->coeffs[i] >> 31));
                for (int j = 0; j < N / 8; j++)
                {
                    w256 = _mm256_load_si256(w + j * 8);
                    s256 = _mm256_loadu_si256(stable32 + j * 8);
                    w256 = _mm256_add_epi32(w256, s256);
                    _mm256_store_si256(w + j * 8, w256);
                }
            }
        }
    }
    return 0;
}

int poly_emulate_ct(poly *r, const poly *c, const poly *t) {
    ALIGN(64) int32_t stable[N * 3];
    int32_t *stable32;
    __m512i w512, s512, f0, f1;
    const __m512i zero = _mm512_setzero_si512();

    int32_t *w = r->coeffs;
    memset(w, 0, N * 4);
    for (int i = 0; i < N / 16; i++) {
        f0 = _mm512_load_si512(t->coeffs + i * 16);
        f1 = _mm512_sub_epi32(zero, f0);
        _mm512_store_si512(stable + i * 16, f1);
        _mm512_store_si512(stable + N * 2 + i * 16, f1);
        _mm512_store_si512(stable + N + i * 16, f0);
    }

    for (int i = 0; i < N; i++) {
        if (c->coeffs[i] != 0) {
            stable32 = stable + N - i + (N & (c->coeffs[i] >> 31));
            for (int j = 0; j < N / 16; j++) {
                w512 = _mm512_load_si512(w + j * 16);
                s512 = _mm512_loadu_si512(stable32 + j * 16);
                w512 = _mm512_add_epi32(w512, s512);
                _mm512_store_si512(w + j * 16, w512);
            }
        }
    }

    return 0;
}

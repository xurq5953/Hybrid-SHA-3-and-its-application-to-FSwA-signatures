#include "params.h"
#include "packing.h"
#include "polyvec.h"
#include "poly.h"

/*************************************************
* Name:        pack_pk
*
* Description: Bit-pack public key pk = (rho, t1).
*
* Arguments:   - uint8_t pk[]: output byte array
*              - const uint8_t rho[]: byte array containing rho
*              - const polyveck *t1: pointer to vector t1
**************************************************/
void pack_pk(uint8_t pk[CRYPTO_PUBLICKEYBYTES],
             const uint8_t rho[SEEDBYTES],
             const polyveck *t1)
{
  unsigned int i;

  for(i = 0; i < SEEDBYTES; ++i)
    pk[i] = rho[i];
  pk += SEEDBYTES;

  for(i = 0; i < K; ++i)
    polyt1_pack(pk + i*POLYT1_PACKEDBYTES, &t1->vec[i]);
}

/*************************************************
* Name:        unpack_pk
*
* Description: Unpack public key pk = (rho, t1).
*
* Arguments:   - const uint8_t rho[]: output byte array for rho
*              - const polyveck *t1: pointer to output vector t1
*              - uint8_t pk[]: byte array containing bit-packed pk
**************************************************/
void unpack_pk(uint8_t rho[SEEDBYTES],
               polyveck *t1,
               const uint8_t pk[CRYPTO_PUBLICKEYBYTES])
{
  unsigned int i;

  for(i = 0; i < SEEDBYTES; ++i)
    rho[i] = pk[i];
  pk += SEEDBYTES;

  for(i = 0; i < K; ++i)
    polyt1_unpack(&t1->vec[i], pk + i*POLYT1_PACKEDBYTES);
}

/*************************************************
* Name:        unpack_sk
*
* Description: Unpack secret key sk = (rho, tr, key, t0, s1, s2).
*
* Arguments:   - const uint8_t rho[]: output byte array for rho
*              - const uint8_t tr[]: output byte array for tr
*              - const uint8_t key[]: output byte array for key
*              - const polyveck *t0: pointer to output vector t0
*              - const polyvecl *s1: pointer to output vector s1
*              - const polyveck *s2: pointer to output vector s2
*              - uint8_t sk[]: byte array containing bit-packed sk
**************************************************/
static void gen_slist(sword slist[N * 3], const uint8_t *a) {
#if ETA == 2
  __m256i f0,f1,f2,f3;
  __m256i t;
  const __m256i mask0 = _mm256_set1_epi32(0x7);
  const __m256i idx0 = _mm256_setr_epi32(0,3,6,9,12,15,18,21);
  const __m256i inx = _mm256_setr_epi8(0,4,8,12,0,0,0,0,0,0,0,0,0,0,0,0,0,4,8,12,0,0,0,0,0,0,0,0,0,0,0,0);
  const __m256i inx1 = _mm256_setr_epi32(0,4,0,0,0,0,0,0);
  const __m256i zero = _mm256_setzero_si256();
  const __m256i eta = _mm256_set1_epi8(ETA);

  for (int i = 0; i < N/32; ++i) {
    t = _mm256_loadu_si256((__m256i *) (a + 12 * i));

    f0 = _mm256_permutevar8x32_epi32(t,zero);
    f0 = _mm256_srlv_epi32(f0,idx0);
    f0 = f0 & mask0;
    f0 = _mm256_shuffle_epi8(f0, inx);
    f0 = _mm256_permutevar8x32_epi32(f0,inx1);

    f1 = _mm256_srli_si256(t,3);
    f1 = _mm256_permutevar8x32_epi32(f1,zero);
    f1 = _mm256_srlv_epi32(f1,idx0);
    f1 = f1 & mask0;
    f1 = _mm256_shuffle_epi8(f1, inx);
    f1 = _mm256_permutevar8x32_epi32(f1,inx1);

    f2 = _mm256_srli_si256(t,6);
    f2 = _mm256_permutevar8x32_epi32(f2,zero);
    f2 = _mm256_srlv_epi32(f2,idx0);
    f2 = f2 & mask0;
    f2 = _mm256_shuffle_epi8(f2, inx);
    f2 = _mm256_permutevar8x32_epi32(f2,inx1);

    f3 = _mm256_srli_si256(t,9);
    f3 = _mm256_permutevar8x32_epi32(f3,zero);
    f3 = _mm256_srlv_epi32(f3,idx0);
    f3 = f3 & mask0;
    f3 = _mm256_shuffle_epi8(f3, inx);
    f3 = _mm256_permutevar8x32_epi32(f3,inx1);

    f1 = _mm256_unpacklo_epi64(f0,f1);
    f3 = _mm256_unpacklo_epi64(f2,f3);
    f1 = _mm256_permute2x128_si256(f1,f3,0x20);
    f2 = _mm256_sub_epi8(f1, eta); //-a
    f3 = _mm256_sub_epi8(eta, f1); // a

    _mm256_store_si256(slist + i * 32, f2);
    _mm256_store_si256(slist + N + i * 32 , f3);
    _mm256_store_si256(slist + 2 * N + i * 32, f2);
  }

#else
  __m256i f0, f1, f2, f3;
  __m256i g0, g1, g2, g3;
  __m256i h0, h1, h2, h3;
  const __m256i mask = _mm256_set1_epi8(0xf);
  const __m256i eta = _mm256_set1_epi8(ETA);
  for(int i = 0; i < N / 64; ++i) {
    f0 = _mm256_loadu_si256(a + i * 32);
    f1 = _mm256_srli_epi16(f0, 4);
    f0 = _mm256_and_si256(f0, mask);
    f1 = _mm256_and_si256(f1, mask);

    f2 = _mm256_unpacklo_epi8(f0,f1);
    f3 = _mm256_unpackhi_epi8(f0, f1);

    f0 = _mm256_sub_epi8(eta, f2); //a
    f1 = _mm256_sub_epi8(eta, f3);
    f2 = _mm256_sub_epi8(f2,eta); //-a
    f3 = _mm256_sub_epi8(f3, eta);

    g0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(f0));
    g1 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(f1));
    g2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(f0,1));
    g3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(f1,1));

    h0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(f2));
    h1 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(f3));
    h2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(f2,1));
    h3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(f3,1));

    _mm256_store_si256(slist + 64 * i, h0);
    _mm256_store_si256(slist + 64 * i + 16, h1);
    _mm256_store_si256(slist + 64 * i + 32, h2);
    _mm256_store_si256(slist + 64 * i + 48, h3);

    _mm256_store_si256(slist + N + 64 * i, g0);
    _mm256_store_si256(slist + N + 64 * i + 16, g1);
    _mm256_store_si256(slist + N + 64 * i + 32, g2);
    _mm256_store_si256(slist + N + 64 * i + 48, g3);

    _mm256_store_si256(slist + 2 * N + 64 * i, h0);
    _mm256_store_si256(slist + 2 * N + 64 * i + 16, h1);
    _mm256_store_si256(slist + 2 * N + 64 * i + 32, h2);
    _mm256_store_si256(slist + 2 * N + 64 * i + 48, h3);

  }
#endif
}

void unpack_sk(uint8_t rho[SEEDBYTES],
               uint8_t tr[CRHBYTES],
               uint8_t key[SEEDBYTES],
               polyveck *t0,
               sword s1list[L][N * 3],
               sword s2list[K][N * 3],
               const uint8_t sk[CRYPTO_SECRETKEYBYTES])
{
  unsigned int i;

  for(i = 0; i < SEEDBYTES; ++i)
    rho[i] = sk[i];
  sk += SEEDBYTES;

  for(i = 0; i < SEEDBYTES; ++i)
    key[i] = sk[i];
  sk += SEEDBYTES;

  for(i = 0; i < CRHBYTES; ++i)
    tr[i] = sk[i];
  sk += CRHBYTES;

  for(i=0; i < L; ++i)
    gen_slist(s1list[i], sk + i*POLYETA_PACKEDBYTES);
  sk += L*POLYETA_PACKEDBYTES;

  for(i=0; i < K; ++i)
    gen_slist(s2list[i], sk + i*POLYETA_PACKEDBYTES);
  sk += K*POLYETA_PACKEDBYTES;

  for(i=0; i < K; ++i)
    polyt0_unpack(&t0->vec[i], sk + i*POLYT0_PACKEDBYTES);
}

/*************************************************
* Name:        pack_sig
*
* Description: Bit-pack signature sig = (c, z, h).
*
* Arguments:   - uint8_t sig[]: output byte array
*              - const uint8_t *c: pointer to challenge hash length SEEDBYTES
*              - const polyvecl *z: pointer to vector z
*              - const polyveck *h: pointer to hint vector h
**************************************************/
void pack_sig(uint8_t sig[CRYPTO_BYTES],
              const uint8_t c[SEEDBYTES],
              const polyvecl *z,
              const polyveck *h)
{
  unsigned int i, j, k;

  for(i=0; i < SEEDBYTES; ++i)
    sig[i] = c[i];
  sig += SEEDBYTES;

  for(i = 0; i < L; ++i)
    polyz_pack(sig + i*POLYZ_PACKEDBYTES, &z->vec[i]);
  sig += L*POLYZ_PACKEDBYTES;

  /* Encode h */
  for(i = 0; i < OMEGA + K; ++i)
    sig[i] = 0;

  k = 0;
  for(i = 0; i < K; ++i) {
    for(j = 0; j < N; ++j)
      if(h->vec[i].coeffs[j] != 0)
        sig[k++] = j;

    sig[OMEGA + i] = k;
  }
}

/*************************************************
* Name:        unpack_sig
*
* Description: Unpack signature sig = (c, z, h).
*
* Arguments:   - uint8_t *c: pointer to output challenge hash
*              - polyvecl *z: pointer to output vector z
*              - polyveck *h: pointer to output hint vector h
*              - const uint8_t sig[]: byte array containing
*                bit-packed signature
*
* Returns 1 in case of malformed signature; otherwise 0.
**************************************************/
int unpack_sig(uint8_t c[SEEDBYTES],
               polyvecl *z,
               polyveck *h,
               const uint8_t sig[CRYPTO_BYTES])
{
  unsigned int i, j, k;

  for(i = 0; i < SEEDBYTES; ++i)
    c[i] = sig[i];
  sig += SEEDBYTES;

  for(i = 0; i < L; ++i)
    polyz_unpack(&z->vec[i], sig + i*POLYZ_PACKEDBYTES);
  sig += L*POLYZ_PACKEDBYTES;

  /* Decode h */
  k = 0;
  for(i = 0; i < K; ++i) {
    for(j = 0; j < N; ++j)
      h->vec[i].coeffs[j] = 0;

    if(sig[OMEGA + i] < k || sig[OMEGA + i] > OMEGA)
      return 1;

    for(j = k; j < sig[OMEGA + i]; ++j) {
      /* Coefficients are ordered for strong unforgeability */
      if(j > k && sig[j] <= sig[j-1]) return 1;
      h->vec[i].coeffs[sig[j]] = 1;
    }

    k = sig[OMEGA + i];
  }

  /* Extra indices are zero for strong unforgeability */
  for(j = k; j < OMEGA; ++j)
    if(sig[j])
      return 1;

  return 0;
}


/*************************************************
* Name:        polyt1_pack
*
* Description: Bit-pack polynomial t1 with coefficients fitting in 10 bits.
*              Input coefficients are assumed to be standard representatives.
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYT1_PACKED_BYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
ALIGN(64) static const uint16_t idx_t_p0[32] = {
  0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

ALIGN(64) static const uint16_t idx_t_p1[32] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
};


void polyt1_pack(uint8_t *r, const poly *a) {
  int pos = 0;
  int ctr = 0;
  __m512i f0, f1;
  __m512i p0;

  const __m512i idx0 = _mm512_load_si512(idx_t_p0);
  const __m512i idx1 = _mm512_load_si512(idx_t_p1);
  const __m512i mask0 = _mm512_set1_epi32(0xffff);
  const __m512i mask1 = _mm512_set1_epi64(0xffffffff);


  for (int i = 0; i < 8; ++i) {
    f0 = _mm512_load_si512(a->coeffs + pos);
    f1 = _mm512_load_si512(a->coeffs + pos + 16);

    f1 = _mm512_maskz_permutexvar_epi16(0xffff0000, idx1, f1);
    f0 = _mm512_mask_permutexvar_epi16(f1, 0xffff, idx0, f0);

    p0 = _mm512_srli_epi32(_mm512_andnot_epi32(mask0, f0), 6);
    f0 = _mm512_ternarylogic_epi32(p0, mask0, f0, 0x78);
    p0 = _mm512_srli_epi64(_mm512_andnot_epi64(mask1, f0), 12);
    f0 = _mm512_ternarylogic_epi32(p0, mask1, f0, 0x78);

    _mm512_mask_compressstoreu_epi8(r + ctr, 0x1f1f1f1f1f1f1f1f, f0);

    pos += 32;
    ctr += 40;
  }
}


/*************************************************
* Name:        polyt1_unpack
*
* Description: Unpack polynomial t1 with 10-bit coefficients.
*              Output coefficients are standard representatives.
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
ALIGN(64) static const uint8_t idx_t1_u1[64] = {
  0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9,
  2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 8, 9, 9, 10, 10, 11,
  0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9,
  2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 8, 9, 9, 10, 10, 11,
};

ALIGN(64) static const uint16_t idx_t1_u2[32] = {
  0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6,
  0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6
};

void polyt1_unpack(poly *r, const uint8_t *a) {
  int pos = 0;

  __m512i b;
  const __m512i mask = _mm512_set1_epi16(0x3FF);
  const __m512i index0 = _mm512_setr_epi32(0, 1, 2, 0,
                                           2, 3, 4, 0,
                                           5, 6, 7, 0,
                                           7, 8, 9, 0);
  const __m512i index1 = _mm512_load_si512(idx_t1_u1);
  const __m512i index2 = _mm512_load_si512(idx_t1_u2);

  for (int i = 0; i < 16; i += 2) {
    b = _mm512_loadu_si512(a + pos);
    b = _mm512_maskz_permutexvar_epi32(0x7777, index0, b);
    b = _mm512_shuffle_epi8(b, index1);
    b = _mm512_srlv_epi16(b, index2);
    // b &= mask;
    b = _mm512_and_si512(b, mask);
    _mm512_store_si512(&r->vec2[i], _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b, 0)));
    _mm512_store_si512(&r->vec2[i + 1], _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b, 1)));
    pos += 40;
  }
}

/*************************************************
* Name:        polyt0_pack
*
* Description: Bit-pack polynomial t0 with coefficients in ]-2^{D-1}, 2^{D-1}].
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYT0_PACKED_BYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
ALIGN(64) static const uint16_t idx_t0_p3[32] = {
  0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0,
  0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 0,
};

void polyt0_pack(uint8_t *r, const poly *a) {
  int pos = 0;
  int ctr = 0;
  __m512i f0, f1;
  __m512i p0;

  const __m512i n = _mm512_set1_epi32((1 << (D - 1)));
  const __m512i mask0 = _mm512_set1_epi64(0xffffffff);
  const __m512i mask1 = _mm512_set1_epi32(0xffff);
  const __m512i idx0 = _mm512_load_si512(idx_t_p0);
  const __m512i idx1 = _mm512_load_si512(idx_t_p1);
  const __m512i idx3 = _mm512_load_si512(idx_t0_p3);

  for (int i = 0; i < 8; ++i) {
    f0 = _mm512_loadu_si512(a->coeffs + pos);
    f1 = _mm512_loadu_si512(a->coeffs + pos + 16);

    f0 = _mm512_sub_epi32(n, f0);
    f1 = _mm512_sub_epi32(n, f1);

    f1 = _mm512_maskz_permutexvar_epi16(0xffff0000, idx1, f1);
    f0 = _mm512_mask_permutexvar_epi16(f1, 0xffff, idx0, f0);

    // f0 = (f0 & mask1) ^ _mm512_srli_epi32(_mm512_andnot_epi32(mask1,f0),3);
    // f0 = (f0 & mask0) ^ _mm512_srli_epi64(_mm512_andnot_epi64(mask0,f0),6);
    p0 = _mm512_srli_epi32(_mm512_andnot_epi32(mask1, f0), 3);
    f0 = _mm512_ternarylogic_epi32(p0, mask1, f0, 0x78);
    p0 = _mm512_srli_epi64(_mm512_andnot_epi64(mask0, f0), 6);
    f0 = _mm512_ternarylogic_epi32(p0, mask0, f0, 0x78);

    p0 = _mm512_maskz_permutexvar_epi16(0x08080808, idx3, f0);
    p0 = _mm512_slli_epi16(p0, 4);
    // f0 |= p0;
    f0 = _mm512_or_si512(f0,p0);
    f0 = _mm512_mask_srli_epi64(f0,0xaa,f0,12);

    _mm512_mask_compressstoreu_epi8(r + ctr, 0x1fff1fff1fff1fff, f0);

    ctr += 52;
    pos += 32;
  }
}

/*************************************************
* Name:        polyt0_unpack
*
* Description: Unpack polynomial t0 with coefficients in ]-2^{D-1}, 2^{D-1}].
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
ALIGN(64) static const uint8_t idx_t0_u[64] = {
  0, 1, 0xff, 0xff,
  1, 2, 3, 0xff,
  3, 4, 0xff, 0xff,
  4, 5, 6, 0xff,
  6, 7, 8, 0xff,
  8, 9, 0xff, 0xff,
  9, 10, 11, 0xff,
  11, 12, 0xff, 0xff,

  13, 14, 0xff, 0xff,
  14, 15, 16, 0xff,
  16, 17, 0xff, 0xff,
  17, 18, 19, 0xff,
  19, 20, 21, 0xff,
  21, 22, 0xff, 0xff,
  22, 23, 24, 0xff,
  24, 25, 0xff, 0xff
};

void polyt0_unpack(poly *r, const uint8_t *a) {
  int pos = 0;
  int ctr = 0;
  __m512i f0, f1, f2, f3;
  const __m512i idx0 = _mm512_load_si512(idx_t0_u);
  const __m512i idx1 = _mm512_setr_epi32(0, 5, 2, 7,
                                         4, 1, 6, 3,
                                         0, 5, 2, 7,
                                         4, 1, 6, 3);
  const __m512i mask = _mm512_set1_epi32(0x1FFF);
  const __m512i n = _mm512_set1_epi32((1 << (D - 1)));


  for (int i = 0; i < 4; ++i) {
    f0 = _mm512_load_si512(a + pos);
    f1 = _mm512_load_si512(a + pos + 26);
    f2 = _mm512_load_si512(a + pos + 52);
    f3 = _mm512_load_si512(a + pos + 78);

    f0 = _mm512_permutexvar_epi8(idx0, f0);
    f1 = _mm512_permutexvar_epi8(idx0, f1);
    f2 = _mm512_permutexvar_epi8(idx0, f2);
    f3 = _mm512_permutexvar_epi8(idx0, f3);

    f0 = _mm512_srlv_epi32(f0, idx1);
    f1 = _mm512_srlv_epi32(f1, idx1);
    f2 = _mm512_srlv_epi32(f2, idx1);
    f3 = _mm512_srlv_epi32(f3, idx1);

    f0 = _mm512_and_si512(f0, mask);
    f1 = _mm512_and_si512(f1, mask);
    f2 = _mm512_and_si512(f2, mask);
    f3 = _mm512_and_si512(f3, mask);

    f0 = _mm512_sub_epi32(n, f0);
    f1 = _mm512_sub_epi32(n, f1);
    f2 = _mm512_sub_epi32(n, f2);
    f3 = _mm512_sub_epi32(n, f3);

    _mm512_storeu_epi32(r->coeffs + ctr, f0);
    _mm512_storeu_epi32(r->coeffs + ctr + 16, f1);
    _mm512_storeu_epi32(r->coeffs + ctr + 32, f2);
    _mm512_storeu_epi32(r->coeffs + ctr + 48, f3);

    pos += 104;
    ctr += 64;
  }
}

/*************************************************
* Name:        polyz_pack
*               reuqire reduancy bytes at end
* Description: Bit-pack polynomial with coefficients
*              in [-(GAMMA1 - 1), GAMMA1].
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYZ_PACKED_BYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
#if GAMMA1 == (1 << 17)
ALIGN(64) static const uint32_t idx_z_p[16] = {
        0, 2, 0, 0,
        0, 6, 0, 0,
        0, 10, 0, 0,
        0, 14, 0, 0
};

void polyz_pack(uint8_t *r, const poly *restrict a) {
    int pos = 0;
    int ctr = 0;
    __m512i f0, f1, f2, f3;
    __m512i p0, p1, p2, p3;

    const __m512i mask0 = _mm512_set1_epi64(0xffffffff);
    const __m512i gamma = _mm512_set1_epi32(GAMMA1);
    const __m512i idx0 = _mm512_load_si512(idx_z_p);

    for (int i = 0; i < 4; ++i) {
        f0 = _mm512_loadu_si512(a->coeffs + pos);
        f1 = _mm512_loadu_si512(a->coeffs + pos + 16);
        f2 = _mm512_loadu_si512(a->coeffs + pos + 32);
        f3 = _mm512_loadu_si512(a->coeffs + pos + 48);

        f0 = _mm512_sub_epi32(gamma, f0);
        f1 = _mm512_sub_epi32(gamma, f1);
        f2 = _mm512_sub_epi32(gamma, f2);
        f3 = _mm512_sub_epi32(gamma, f3);

        p0 = _mm512_srli_epi64(_mm512_andnot_epi64(mask0, f0), 14);
        p1 = _mm512_srli_epi64(_mm512_andnot_epi64(mask0, f1), 14);
        p2 = _mm512_srli_epi64(_mm512_andnot_epi64(mask0, f2), 14);
        p3 = _mm512_srli_epi64(_mm512_andnot_epi64(mask0, f3), 14);

        f0 = _mm512_ternarylogic_epi32(p0, f0, mask0, 0x78);
        f1 = _mm512_ternarylogic_epi32(p1, f1, mask0, 0x78);
        f2 = _mm512_ternarylogic_epi32(p2, f2, mask0, 0x78);
        f3 = _mm512_ternarylogic_epi32(p3, f3, mask0, 0x78);

        p0 = _mm512_maskz_permutexvar_epi32(0x2222, idx0, f0);
        p1 = _mm512_maskz_permutexvar_epi32(0x2222, idx0, f1);
        p2 = _mm512_maskz_permutexvar_epi32(0x2222, idx0, f2);
        p3 = _mm512_maskz_permutexvar_epi32(0x2222, idx0, f3);

        p0 = _mm512_slli_epi32(p0, 4);
        p1 = _mm512_slli_epi32(p1, 4);
        p2 = _mm512_slli_epi32(p2, 4);
        p3 = _mm512_slli_epi32(p3, 4);

        f0 = _mm512_or_si512(f0,p0);
        f1 = _mm512_or_si512(f1,p1);
        f2 = _mm512_or_si512(f2,p2);
        f3 = _mm512_or_si512(f3,p3);

        f0 = _mm512_mask_srli_epi64(f0, 0xaa, f0, 28);
        f1 = _mm512_mask_srli_epi64(f1, 0xaa, f1, 28);
        f2 = _mm512_mask_srli_epi64(f2, 0xaa, f2, 28);
        f3 = _mm512_mask_srli_epi64(f3, 0xaa, f3, 28);

        _mm512_mask_compressstoreu_epi8(r + ctr, 0x01ff01ff01ff01ff, f0);
        _mm512_mask_compressstoreu_epi8(r + ctr + 36, 0x01ff01ff01ff01ff, f1);
        _mm512_mask_compressstoreu_epi8(r + ctr + 72, 0x01ff01ff01ff01ff, f2);
        _mm512_mask_compressstoreu_epi8(r + ctr + 108, 0x01ff01ff01ff01ff, f3);

        pos += 64;
        ctr += 144;

    }

}
#elif GAMMA1 == (1 << 19)
void polyz_pack(uint8_t *r, const poly *a) {
  int pos = 0;
  int ctr = 0;
  __m512i f0, f1, f2, f3;
  __m512i p0, p1, p2, p3;

  const __m512i mask0 = _mm512_set1_epi64(0xffffffff);
  const __m512i gamma = _mm512_set1_epi32(GAMMA1);

  for (int i = 0; i < 4; ++i) {
    f0 = _mm512_loadu_si512(a->coeffs + pos);
    f1 = _mm512_loadu_si512(a->coeffs + pos + 16);
    f2 = _mm512_loadu_si512(a->coeffs + pos + 32);
    f3 = _mm512_loadu_si512(a->coeffs + pos + 48);

    f0 = _mm512_sub_epi32(gamma, f0);
    f1 = _mm512_sub_epi32(gamma, f1);
    f2 = _mm512_sub_epi32(gamma, f2);
    f3 = _mm512_sub_epi32(gamma, f3);

    p0 = _mm512_srli_epi64(_mm512_andnot_epi64(mask0, f0), 12);
    p1 = _mm512_srli_epi64(_mm512_andnot_epi64(mask0, f1), 12);
    p2 = _mm512_srli_epi64(_mm512_andnot_epi64(mask0, f2), 12);
    p3 = _mm512_srli_epi64(_mm512_andnot_epi64(mask0, f3), 12);

    f0 = _mm512_ternarylogic_epi32(p0, f0, mask0, 0x78);
    f1 = _mm512_ternarylogic_epi32(p1, f1, mask0, 0x78);
    f2 = _mm512_ternarylogic_epi32(p2, f2, mask0, 0x78);
    f3 = _mm512_ternarylogic_epi32(p3, f3, mask0, 0x78);

    _mm512_mask_compressstoreu_epi8(r + ctr, 0x1f1f1f1f1f1f1f1f, f0);
    _mm512_mask_compressstoreu_epi8(r + ctr + 40, 0x1f1f1f1f1f1f1f1f, f1);
    _mm512_mask_compressstoreu_epi8(r + ctr + 80, 0x1f1f1f1f1f1f1f1f, f2);
    _mm512_mask_compressstoreu_epi8(r + ctr + 120, 0x1f1f1f1f1f1f1f1f, f3);

    pos += 64;
    ctr += 160;

  }

}
#endif

/*************************************************
* Name:        polyz_unpack
*
* Description: Unpack polynomial z with coefficients
*              in [-(GAMMA1 - 1), GAMMA1].
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
#if GAMMA1 == (1 << 19)

ALIGN(64) static const uint8_t z_up[64] = {
  0, 1, 2, 0,
  2, 3, 4, 0,
  5, 6, 7, 0,
  7, 8, 9, 0,
  10, 11, 12, 0,
  12, 13, 14, 0,
  15, 16, 17, 0,
  17, 18, 19, 0,
  20, 21, 22, 0,
  22, 23, 24, 0,
  25, 26, 27, 0,
  27, 28, 29, 0,
  30, 31, 32, 0,
  32, 33, 34, 0,
  35, 36, 37, 0,
  37, 38, 39, 0
};

void polyz_unpack(poly *r, const uint8_t *a) {
  __m512i a0, a1, a2, a3;
  int pos = 0;
  int ctr = 0;
  const __m512i mask = _mm512_set1_epi32(0xFFFFF);
  const __m512i gamma = _mm512_set1_epi32(GAMMA1);
  const __m512i index = _mm512_load_si512(z_up);
  const __m512i index5 = _mm512_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4);


  for (int i = 0; i < 4; ++i) {
    a0 = _mm512_loadu_si512(a + pos);
    a1 = _mm512_loadu_si512(a + pos + 40);
    a2 = _mm512_loadu_si512(a + pos + 80);
    a3 = _mm512_loadu_si512(a + pos + 120);

    a0 = _mm512_permutexvar_epi8(index, a0);
    a1 = _mm512_permutexvar_epi8(index, a1);
    a2 = _mm512_permutexvar_epi8(index, a2);
    a3 = _mm512_permutexvar_epi8(index, a3);

    a0 = _mm512_srlv_epi32(a0, index5);
    a1 = _mm512_srlv_epi32(a1, index5);
    a2 = _mm512_srlv_epi32(a2, index5);
    a3 = _mm512_srlv_epi32(a3, index5);

    a0 = _mm512_and_si512(a0, mask);
    a1 = _mm512_and_si512(a1, mask);
    a2 = _mm512_and_si512(a2, mask);
    a3 = _mm512_and_si512(a3, mask);

    a0 = _mm512_sub_epi32(gamma, a0);
    a1 = _mm512_sub_epi32(gamma, a1);
    a2 = _mm512_sub_epi32(gamma, a2);
    a3 = _mm512_sub_epi32(gamma, a3);

    _mm512_storeu_epi32(r->coeffs + ctr, a0);
    _mm512_storeu_epi32(r->coeffs + ctr + 16, a1);
    _mm512_storeu_epi32(r->coeffs + ctr + 32, a2);
    _mm512_storeu_epi32(r->coeffs + ctr + 48, a3);

    pos += 160;
    ctr += 64;
  }

}

#else

ALIGN(64) static const uint8_t idx_gamma[64] = {
  0, 1, 2, 0x80, 2, 3, 4, 0x80, 4, 5, 6, 0x80, 6, 7, 8, 0x80,
  9, 10, 11, 0x80, 11, 12, 13, 0x80, 13, 14, 15, 0x80, 15, 16, 17, 0x80,
  18, 19, 20, 0x80, 20, 21, 22, 0x80, 22, 23, 24, 0x80, 24, 25, 26, 0x80,
  27, 28, 29, 0x80, 29, 30, 31, 0x80, 31, 32, 33, 0x80, 33, 34, 35, 0x80,
};

void polyz_unpack(poly *r, const uint8_t *a) {
    __m512i a0;
    int pos = 0;
    int ctr = 0;
    const __m512i mask = _mm512_set1_epi32(0x3FFFF);
    const __m512i gamma = _mm512_set1_epi32(GAMMA1);
    const __m512i index = _mm512_load_si512(idx_gamma);
    const __m512i index3 = _mm512_setr_epi32(0, 1, 2, 0,
                                             2, 3, 4, 0,
                                             4, 5, 6, 0,
                                             6, 7, 8, 0);
    const __m512i index5 = _mm512_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6);


    for (int i = 0; i < 16; ++i) {
        a0 = _mm512_loadu_si512(a + pos);
        a0 = _mm512_permutexvar_epi8(index, a0);
        a0 = _mm512_srlv_epi32(a0, index5);
         a0 = _mm512_and_si512(a0,mask);
        a0 = _mm512_sub_epi32(gamma, a0);
        _mm512_storeu_epi32(r->coeffs + ctr, a0);

        pos += 36;
        ctr += 16;
    }

}
#endif

/*************************************************
* Name:        polyw1_pack
*
* Description: Bit-pack polynomial w1 with coefficients in [0,15] or [0,43].
*              Input coefficients are assumed to be standard representatives.
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYW1_PACKED_BYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
#if GAMMA2 == (Q-1)/88
ALIGN(64) static const uint8_t idx_w1_p[64] = {
  0, 1, 2,
  16, 17, 18,
  32, 33, 34,
  48, 49, 50,

  4, 5, 6,
  20, 21, 22,
  36, 37, 38,
  52, 53, 54,

  8, 9, 10,
  24, 25, 26,
  40, 41, 42,
  56, 57, 58,

  12, 13, 14,
  28, 29, 30,
  44, 45, 46,
  60, 61, 62,

  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};

void polyw1_pack(uint8_t *r, const poly *a) {
  __m512i f0, f1, f2, f3;
  const __m512i shift1 = _mm512_set1_epi16((64 << 8) + 1);
  const __m512i shift2 = _mm512_set1_epi32((4096 << 16) + 1);
  const __m512i idx0 = _mm512_load_si512(idx_w1_p);

  for (int i = 0; i < 4; i++) {
    f0 = _mm512_load_si512(&a->vec2[4 * i + 0]);
    f1 = _mm512_load_si512(&a->vec2[4 * i + 1]);
    f2 = _mm512_load_si512(&a->vec2[4 * i + 2]);
    f3 = _mm512_load_si512(&a->vec2[4 * i + 3]);
    f0 = _mm512_packus_epi32(f0, f1);
    f1 = _mm512_packus_epi32(f2, f3);
    f0 = _mm512_packus_epi16(f0, f1);
    f0 = _mm512_maddubs_epi16(f0, shift1);
    f0 = _mm512_madd_epi16(f0, shift2);
    f0 = _mm512_permutexvar_epi8(idx0, f0);
    _mm512_storeu_si512(&r[48 * i], f0);
  }

}

#elif GAMMA2 == (Q-1)/32
ALIGN(64) static const uint8_t idx_w1_p[64] = {
  0, 1, 16, 17, 32, 33, 48, 49,
  2, 3, 18, 19, 34, 35, 50, 51,
  4, 5, 20, 21, 36, 37, 52, 53,
  6, 7, 22, 23, 38, 39, 54, 55,
  8, 9, 24, 25, 40, 41, 56, 57,
  10, 11, 26, 27, 42, 43, 58, 59,
  12, 13, 28, 29, 44, 45, 60, 61,
  14, 15, 30, 31, 46, 47, 62, 63
};

void polyw1_pack(uint8_t *r, const poly *restrict a) {
  __m512i f0, f1, f2, f3, f4, f5, f6, f7;
  const __m512i shift1 = _mm512_set1_epi16((16 << 8) + 1);
  const __m512i idx0 = _mm512_load_si512(idx_w1_p);

  for (int i = 0; i < 2; i++) {
    f0 = _mm512_load_si512(&a->vec2[8 * i + 0]);
    f1 = _mm512_load_si512(&a->vec2[8 * i + 1]);
    f2 = _mm512_load_si512(&a->vec2[8 * i + 2]);
    f3 = _mm512_load_si512(&a->vec2[8 * i + 3]);
    f4 = _mm512_load_si512(&a->vec2[8 * i + 4]);
    f5 = _mm512_load_si512(&a->vec2[8 * i + 5]);
    f6 = _mm512_load_si512(&a->vec2[8 * i + 6]);
    f7 = _mm512_load_si512(&a->vec2[8 * i + 7]);
    f0 = _mm512_packus_epi32(f0, f1);
    f1 = _mm512_packus_epi32(f2, f3);
    f2 = _mm512_packus_epi32(f4, f5);
    f3 = _mm512_packus_epi32(f6, f7);
    f0 = _mm512_packus_epi16(f0, f1);
    f1 = _mm512_packus_epi16(f2, f3);
    f0 = _mm512_maddubs_epi16(f0, shift1);
    f1 = _mm512_maddubs_epi16(f1, shift1);
    f0 = _mm512_packus_epi16(f0, f1);

    f0 = _mm512_permutexvar_epi8(idx0, f0);
    _mm512_storeu_si512(r + 64 * i, f0);
  }

}
#endif

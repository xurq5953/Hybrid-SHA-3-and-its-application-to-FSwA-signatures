//
// Created by xurq on 2023/3/8.
//
#include "ntt.h"
#include "consts.h"
#include <x86intrin.h>





#define LOADB \
b0  = _mm512_load_epi32(a      );\
b1  = _mm512_load_epi32(a + 16 );\
b2  = _mm512_load_epi32(a + 32 );\
b3  = _mm512_load_epi32(a + 48 );\
b4  = _mm512_load_epi32(a + 64 );\
b5  = _mm512_load_epi32(a + 80 );\
b6  = _mm512_load_epi32(a + 96 );\
b7  = _mm512_load_epi32(a + 112);\
b8  = _mm512_load_epi32(a + 128);\
b9  = _mm512_load_epi32(a + 144);\
b10 = _mm512_load_epi32(a + 160);\
b11 = _mm512_load_epi32(a + 176);\
b12 = _mm512_load_epi32(a + 192);\
b13 = _mm512_load_epi32(a + 208);\
b14 = _mm512_load_epi32(a + 224);\
b15 = _mm512_load_epi32(a + 240);\


#define STORE \
_mm512_store_si512(a      , z0 );\
_mm512_store_si512(a + 16 , z1 );\
_mm512_store_si512(a + 32 , z2 );\
_mm512_store_si512(a + 48 , z3 );\
_mm512_store_si512(a + 64 , z4 );\
_mm512_store_si512(a + 80 , z5 );\
_mm512_store_si512(a + 96 , z6 );\
_mm512_store_si512(a + 112, z7 );\
_mm512_store_si512(a + 128, z8 );\
_mm512_store_si512(a + 144, z9 );\
_mm512_store_si512(a + 160, z10);\
_mm512_store_si512(a + 176, z11);\
_mm512_store_si512(a + 192, z12);\
_mm512_store_si512(a + 208, z13);\
_mm512_store_si512(a + 224, z14);\
_mm512_store_si512(a + 240, z15);\


#define STOREB \
_mm512_store_si512(a      , b0 );\
_mm512_store_si512(a + 16 , b1 );\
_mm512_store_si512(a + 32 , b2 );\
_mm512_store_si512(a + 48 , b3 );\
_mm512_store_si512(a + 64 , b4 );\
_mm512_store_si512(a + 80 , b5 );\
_mm512_store_si512(a + 96 , b6 );\
_mm512_store_si512(a + 112, b7 );\
_mm512_store_si512(a + 128, b8 );\
_mm512_store_si512(a + 144, b9 );\
_mm512_store_si512(a + 160, b10);\
_mm512_store_si512(a + 176, b11);\
_mm512_store_si512(a + 192, b12);\
_mm512_store_si512(a + 208, b13);\
_mm512_store_si512(a + 224, b14);\
_mm512_store_si512(a + 240, b15);\




#define shuffle16(a,b) \
    t = (a);                    \
    (a) = _mm512_shuffle_i32x4(a, b, 0x44);\
    (b) =  _mm512_shuffle_i32x4(t, b, 0xee); \
\

#define shuffle8(a,b) \
    t = _mm512_permutex_epi64((a), 0x4e);  \
    (a) = _mm512_mask_blend_epi64(0xcc, (a), _mm512_permutex_epi64((b), 0x4e));\
    (b) =  _mm512_mask_blend_epi64(0x33, (b), t);\
\


#define shuffle4(a,b) \
    t = (a);\
    (a) = _mm512_mask_shuffle_epi32((a), 0xcccc, (b), _MM_SHUFFLE(1, 0, 1, 0));\
    (b) = _mm512_mask_shuffle_epi32((b), 0x3333, t, _MM_SHUFFLE(3, 2, 3, 2));\
\

#define shuffle2(a,b) \
    t = (a);\
    (a) = _mm512_mask_shuffle_epi32((a), 0xaaaa, (b), _MM_SHUFFLE(2, 2, 0, 0));\
    (b) = _mm512_mask_shuffle_epi32((b), 0x5555, t, _MM_SHUFFLE(3, 3, 1, 1));  \
\


#define shuffle_inv(a,b) \
    t = (a);             \
    (a) =  _mm512_shuffle_i32x4((a), (b), 0x44); \
    (b) = _mm512_shuffle_i32x4(t, (b), 0xee);    \
    (a) = _mm512_permutexvar_epi32(idx, (a));\
    (b) = _mm512_permutexvar_epi32(idx, (b));\
  \



#define shuffle_forward(a,b) \
     t = (a);             \
     (a) = _mm512_maskz_permutexvar_epi32(0xff, idx0, t) | _mm512_maskz_permutexvar_epi32(0xff00, idx1, (b));\
     (b) = _mm512_maskz_permutexvar_epi32(0xff, idx3, t) | _mm512_maskz_permutexvar_epi32(0xff00, idx4, (b));\
\


#define  butterfly(a, b) \
t = _mm512_sub_epi32(a,b); \
a = _mm512_add_epi32(a,b); \
r0 = _mm512_mul_epi32(z1, t); \
b = _mm512_srli_epi64(t,32);\
r1 = _mm512_mul_epi32(b,z1); \
t = _mm512_mul_epi32(t,z0);    \
b = _mm512_mul_epi32(b,z0);\
r1 = _mm512_mul_epi32(q, r1);\
r0 = _mm512_mul_epi32(q, r0);\
t = _mm512_sub_epi32(t, r0);\
b = _mm512_sub_epi32(b, r1);\
b = _mm512_mask_blend_epi32(0xaaaa,_mm512_srli_epi64(t,32),b);\
\


#define butterfly0(a, b) \
t = _mm512_sub_epi32(a,b); \
a = _mm512_add_epi32(a,b); \
r0 = _mm512_mul_epi32(z1, t); \
b = _mm512_srli_epi64(t,32);\
r1 = _mm512_mul_epi32(b,_mm512_srli_epi64(z1,32)); \
t = _mm512_mul_epi32(t,z0);    \
b = _mm512_mul_epi32(b,_mm512_srli_epi64(z0,32));\
r1 = _mm512_mul_epi32(q,r1);\
r0 = _mm512_mul_epi32(q,r0);\
t = _mm512_sub_epi32(t, r0);\
b = _mm512_sub_epi32(b, r1);\
b =  _mm512_mask_blend_epi32(0xaaaa,_mm512_srli_epi64(t,32),b);\
\


#define reduce(b) \
r0 = _mm512_mul_epi32(dqiv, b); \
t = _mm512_srli_epi64(b,32);     \
r1 = _mm512_mul_epi32(t,dqiv);  \
t = _mm512_mul_epi32(t,div);    \
b = _mm512_mul_epi32(b,div);    \
r1 = _mm512_mul_epi32(q, r1);\
r0 = _mm512_mul_epi32(q, r0);    \
t = _mm512_sub_epi32(t, r1);\
b = _mm512_sub_epi32(b, r0);\
b = _mm512_mask_blend_epi32(0xaaaa,_mm512_srli_epi64(b,32),t);\
\



void intt_bo_avx512(int32_t a[256]) {
    __m512i b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15;

    const __m512i q = _mm512_set1_epi32(Q);
    const __m512i qinv = _mm512_set1_epi32(QINV);
    const __m512i div = _mm512_set1_epi32(DIV);
    const __m512i dqiv = _mm512_set1_epi32(DIV_QINV);

    const __m512i idx = _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);

    const __m512i idx0 = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m512i idx1 = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i idx3 = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m512i idx4 = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 7, 9, 11, 13, 15);

    __m512i z0, z1;
    __m512i r0, r1, t, f;
    __m512i *vpsi = (__m512i *) ipsi;
    int ptr = 0;

    LOADB

    shuffle_forward(b0,b1)
    shuffle_forward(b2,b3)
    shuffle_forward(b4,b5)
    shuffle_forward(b6,b7)
    shuffle_forward(b8,b9)
    shuffle_forward(b10,b11)
    shuffle_forward(b12,b13)
    shuffle_forward(b14,b15)


    //level 0
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly0(b0,b1)
    shuffle2(b0,b1)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly0(b2,b3)
    shuffle2(b2,b3)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly0(b4,b5)
    shuffle2(b4,b5)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly0(b6,b7)
    shuffle2(b6,b7)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly0(b8,b9)
    shuffle2(b8,b9)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly0(b10,b11)
    shuffle2(b10,b11)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly0(b12,b13)
    shuffle2(b12,b13)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly0(b14,b15)
    shuffle2(b14,b15)

    //level 1
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b0,b1)
    shuffle4(b0,b1)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b2,b3)
    shuffle4(b2,b3)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b4,b5)
    shuffle4(b4,b5)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b6,b7)
    shuffle4(b6,b7)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b8,b9)
    shuffle4(b8,b9)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b10,b11)
    shuffle4(b10,b11)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b12,b13)
    shuffle4(b12,b13)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b14,b15)
    shuffle4(b14,b15)


    //level 2
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b0,b1)
    shuffle8(b0,b1)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b2,b3)
    shuffle8(b2,b3)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b4,b5)
    shuffle8(b4,b5)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b6,b7)
    shuffle8(b6,b7)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b8,b9)
    shuffle8(b8,b9)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b10,b11)
    shuffle8(b10,b11)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b12,b13)
    shuffle8(b12,b13)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b14,b15)
    shuffle8(b14,b15)

    //level 3
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b0,b1)
    shuffle16(b0,b1)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b2,b3)
    shuffle16(b2,b3)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b4,b5)
    shuffle16(b4,b5)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b6,b7)
    shuffle16(b6,b7)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b8,b9)
    shuffle16(b8,b9)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b10,b11)
    shuffle16(b10,b11)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b12,b13)
    shuffle16(b12,b13)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b14,b15)
    shuffle16(b14,b15)

    ptr = 0;


    //level 4
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b0,b1)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b2,b3)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b4,b5)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b6,b7)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b8,b9)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b10,b11)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b12,b13)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b14,b15)



    //level 5
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b0,b2)
    butterfly(b1,b3)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b4,b6)
    butterfly(b5,b7)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b8,b10)
    butterfly(b9,b11)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b12,b14)
    butterfly(b13,b15)

    //level 6
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b0,b4)
    butterfly(b1,b5)
    butterfly(b2,b6)
    butterfly(b3,b7)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b8 ,b12)
    butterfly(b9 ,b13)
    butterfly(b10,b14)
    butterfly(b11,b15)



    //level 7
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b0,b8 )
    butterfly(b1,b9 )
    butterfly(b2,b10)
    butterfly(b3,b11)
    butterfly(b4,b12)
    butterfly(b5,b13)
    butterfly(b6,b14)
    butterfly(b7,b15)

    reduce(b0)
    reduce(b1)
    reduce(b2)
    reduce(b3)
    reduce(b4)
    reduce(b5)
    reduce(b6)
    reduce(b7)


    STOREB
}


void intt_so_avx512(int32_t a[N]) {
    __m512i b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15;

    const __m512i q = _mm512_set1_epi32(Q);
    const __m512i qinv = _mm512_set1_epi32(QINV);
    const __m512i div = _mm512_set1_epi32(DIV);
    const __m512i dqiv = _mm512_set1_epi32(DIV_QINV);

    const __m512i idx = _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);

    const __m512i idx0 = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m512i idx1 = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i idx3 = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m512i idx4 = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 7, 9, 11, 13, 15);

    __m512i z0, z1;
    __m512i r0, r1, t, f;
    __m512i *vpsi = (__m512i *) ipsi;
    int ptr = 0;

    LOADB


    //level 0
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly0(b0,b1)
    shuffle2(b0,b1)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly0(b2,b3)
    shuffle2(b2,b3)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly0(b4,b5)
    shuffle2(b4,b5)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly0(b6,b7)
    shuffle2(b6,b7)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly0(b8,b9)
    shuffle2(b8,b9)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly0(b10,b11)
    shuffle2(b10,b11)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly0(b12,b13)
    shuffle2(b12,b13)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly0(b14,b15)
    shuffle2(b14,b15)

    //level 1
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b0,b1)
    shuffle4(b0,b1)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b2,b3)
    shuffle4(b2,b3)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b4,b5)
    shuffle4(b4,b5)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b6,b7)
    shuffle4(b6,b7)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b8,b9)
    shuffle4(b8,b9)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b10,b11)
    shuffle4(b10,b11)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b12,b13)
    shuffle4(b12,b13)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b14,b15)
    shuffle4(b14,b15)


    //level 2
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b0,b1)
    shuffle8(b0,b1)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b2,b3)
    shuffle8(b2,b3)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b4,b5)
    shuffle8(b4,b5)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b6,b7)
    shuffle8(b6,b7)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b8,b9)
    shuffle8(b8,b9)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b10,b11)
    shuffle8(b10,b11)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b12,b13)
    shuffle8(b12,b13)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b14,b15)
    shuffle8(b14,b15)

    //level 3
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b0,b1)
    shuffle16(b0,b1)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b2,b3)
    shuffle16(b2,b3)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b4,b5)
    shuffle16(b4,b5)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b6,b7)
    shuffle16(b6,b7)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b8,b9)
    shuffle16(b8,b9)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b10,b11)
    shuffle16(b10,b11)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b12,b13)
    shuffle16(b12,b13)
    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b14,b15)
    shuffle16(b14,b15)

    ptr = 0;


    //level 4
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b0,b1)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b2,b3)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b4,b5)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b6,b7)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b8,b9)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b10,b11)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b12,b13)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b14,b15)



    //level 5
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b0,b2)
    butterfly(b1,b3)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b4,b6)
    butterfly(b5,b7)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b8,b10)
    butterfly(b9,b11)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b12,b14)
    butterfly(b13,b15)

    //level 6
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b0,b4)
    butterfly(b1,b5)
    butterfly(b2,b6)
    butterfly(b3,b7)
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b8 ,b12)
    butterfly(b9 ,b13)
    butterfly(b10,b14)
    butterfly(b11,b15)



    //level 7
    z0 = _mm512_set1_epi64(ipsi2[ptr++]);
    z1 = _mm512_srli_epi64(z0,32);
    butterfly(b0,b8 )
    butterfly(b1,b9 )
    butterfly(b2,b10)
    butterfly(b3,b11)
    butterfly(b4,b12)
    butterfly(b5,b13)
    butterfly(b6,b14)
    butterfly(b7,b15)

    reduce(b0)
    reduce(b1)
    reduce(b2)
    reduce(b3)
    reduce(b4)
    reduce(b5)
    reduce(b6)
    reduce(b7)


    STOREB
}
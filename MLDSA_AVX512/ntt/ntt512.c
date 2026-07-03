//
// Created by xurq on 2022/11/17.
//
#include "ntt.h"
#include "consts.h"
#include <x86intrin.h>


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



#define shuffle16a(a,b,c) \
    (c) = _mm512_shuffle_i32x4(a, b, 0x44);\
    (b) =  _mm512_shuffle_i32x4(a, b, 0xee); \
\



#define shuffle8a(a,b,c) \
    (c) = _mm512_mask_blend_epi64(0xcc, (a), _mm512_permutex_epi64((b), 0x4e)); \
    (b) =  _mm512_mask_blend_epi64(0x33, (b), _mm512_permutex_epi64((a), 0x4e));\
\



#define shuffle4a(a,b) \
    t = (a);\
    (a) = _mm512_mask_shuffle_epi32((a), 0xcccc, (b), _MM_SHUFFLE(1, 0, 1, 0));\
    (b) = _mm512_mask_shuffle_epi32((b), 0x3333, t, _MM_SHUFFLE(3, 2, 3, 2));\
\

#define shuffle2a(a,b) \
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




#define butterfly(a,b) \
r0 = _mm512_mul_epi32(z1,b);\
h = _mm512_srli_epi64((b),32);     \
r1 = _mm512_mul_epi32(z1,h);\
(b) = _mm512_mul_epi32(z0,(b));   \
h = _mm512_mul_epi32(z0,h);   \
r0 = _mm512_mul_epi32(r0,q); \
r1 = _mm512_mul_epi32(r1,q);    \
r1 = _mm512_sub_epi32(r1, h); \
r0 = _mm512_sub_epi32(r0, (b));  \
r1 = _mm512_mask_blend_epi32(0xAAAA,_mm512_srli_epi64(r0,32),r1); \
(b) = _mm512_add_epi32((a),r1);\
(a) = _mm512_sub_epi32((a),r1);\
\


#define butterfly7(a,b) \
r0 = _mm512_mul_epi32(z1,b);\
h = _mm512_srli_epi64((b),32);     \
r1 = _mm512_mul_epi32(_mm512_srli_epi64(z1,32),h);\
(b) = _mm512_mul_epi32(z0,(b));   \
h = _mm512_mul_epi32(_mm512_srli_epi64(z0,32),h);   \
r0 = _mm512_mul_epi32(r0,q); \
r1 = _mm512_mul_epi32(r1,q);    \
r1 = _mm512_sub_epi32(r1, h); \
r0 = _mm512_sub_epi32(r0, (b));  \
r1 = _mm512_mask_blend_epi32(0xAAAA,_mm512_srli_epi64(r0,32),r1); \
(b) = _mm512_add_epi32((a),r1);\
(a) = _mm512_sub_epi32((a),r1);\
\





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


void ntt_bo_avx512(int32_t a[N]) {
    __m512i b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15;

    const __m512i q = _mm512_set1_epi32(Q);

    const __m512i idx = _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);

    __m512i z0, z1;
    __m512i r0,r1,t,f,t0,t1,h;
    __m512i *vpsi = (__m512i*)psi;
    int ptr = 0;

    LOADB
    //level 0
    z0 = _mm512_set1_epi32(25847);
    z1 = _mm512_set1_epi32(1830765815);
    butterfly(b0,b8 );
    butterfly(b1,b9 );
    butterfly(b2,b10);
    butterfly(b3,b11);
    butterfly(b4,b12);
    butterfly(b5,b13);
    butterfly(b6,b14);
    butterfly(b7,b15);



    //level 1
    z0 = _mm512_set1_epi32(-2608894);
    z1 = _mm512_set1_epi32(-1929875198);
    butterfly(b0,b4);
    butterfly(b1,b5);
    butterfly(b2,b6);
    butterfly(b3,b7);
    z0 = _mm512_set1_epi32(-518909);
    z1 = _mm512_set1_epi32(-1927777021);
    butterfly(b8 ,b12);
    butterfly(b9 ,b13);
    butterfly(b10,b14);
    butterfly(b11,b15);



    //level 2
    z0 = _mm512_set1_epi32(237124);
    z1 = _mm512_set1_epi32(1640767044);
    butterfly(b0 ,b2 );
    butterfly(b1 ,b3 );
    z0 = _mm512_set1_epi32(-777960);
    z1 = _mm512_set1_epi32(1477910808);
    butterfly(b4 ,b6 );
    butterfly(b5 ,b7 );
    z0 = _mm512_set1_epi32(-876248);
    z1 = _mm512_set1_epi32(1612161320);
    butterfly(b8 ,b10);
    butterfly(b9 ,b11);
    z0 = _mm512_set1_epi32(466468);
    z1 = _mm512_set1_epi32(1640734244);
    butterfly(b12,b14);
    butterfly(b13,b15);

    //level 3
    z0 = _mm512_set1_epi32(1826347);
    z1 = _mm512_set1_epi32(308362795);
    butterfly(b0,b1 );
    z0 = _mm512_set1_epi32(2353451);
    z1 = _mm512_set1_epi32(-1815525077);
    butterfly(b2,b3 );
    z0 = _mm512_set1_epi32(-359251);
    z1 = _mm512_set1_epi32(-1374673747);
    butterfly(b4,b5);
    z0 = _mm512_set1_epi32(-2091905);
    z1 = _mm512_set1_epi32(-1091570561);
    butterfly(b6,b7);
    z0 = _mm512_set1_epi32(3119733);
    z1 = _mm512_set1_epi32(-1929495947);
    butterfly(b8,b9);
    z0 = _mm512_set1_epi32(-2884855);
    z1 = _mm512_set1_epi32(515185417);
    butterfly(b10,b11);
    z0 = _mm512_set1_epi32(3111497);
    z1 = _mm512_set1_epi32(-285697463);
    butterfly(b12,b13);
    z0 = _mm512_set1_epi32(2680103);
    z1 = _mm512_set1_epi32(625853735);
    butterfly(b14,b15);



    //level 4
    shuffle16(b0,b1)
    shuffle16(b2,b3)
    shuffle16(b4,b5)
    shuffle16(b6,b7)
    shuffle16(b8,b9)
    shuffle16(b10,b11)
    shuffle16(b12,b13)
    shuffle16(b14,b15)


    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b0,b1)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b2,b3)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b4,b5)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b6,b7)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b8,b9)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b10,b11)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b12,b13)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b14,b15)


    //level 5
    shuffle8(b0,b1)
    shuffle8(b2,b3)
    shuffle8(b4,b5)
    shuffle8(b6,b7)
    shuffle8(b8,b9)
    shuffle8(b10,b11)
    shuffle8(b12,b13)
    shuffle8(b14,b15)


    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b0,b1)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b2,b3)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b4,b5)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b6,b7)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b8,b9)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b10,b11)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b12,b13)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b14,b15)



    //level 6
    shuffle4(b0,b1)
    shuffle4(b2,b3)
    shuffle4(b4,b5)
    shuffle4(b6,b7)
    shuffle4(b8,b9)
    shuffle4(b10,b11)
    shuffle4(b12,b13)
    shuffle4(b14,b15)


    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b0,b1)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b2,b3)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b4,b5)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b6,b7)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b8,b9)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b10,b11)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b12,b13)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b14,b15)

    //level 7
    shuffle2(b0,b1)
    shuffle2(b2,b3)
    shuffle2(b4,b5)
    shuffle2(b6,b7)
    shuffle2(b8,b9)
    shuffle2(b10,b11)
    shuffle2(b12,b13)
    shuffle2(b14,b15)


    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly7(b0,b1)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly7(b2,b3)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly7(b4,b5)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly7(b6,b7)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly7(b8,b9)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly7(b10,b11)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly7(b12,b13)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly7(b14,b15)


    shuffle_inv(b0,b1)
    shuffle_inv(b2,b3)
    shuffle_inv(b4,b5)
    shuffle_inv(b6,b7)
    shuffle_inv(b8,b9)
    shuffle_inv(b10,b11)
    shuffle_inv(b12,b13)
    shuffle_inv(b14,b15)

    STOREB
}


void ntt_so_avx512(int32_t a[N]) {
    __m512i b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15;
    __m512i c0,c2,c4,c6,c8,c10,c12,c14;
    const __m512i q = _mm512_set1_epi32(Q);


    __m512i z0, z1;
    __m512i r0,r1,t,h;
    __m512i *vpsi = (__m512i*)psi;
    int ptr = 0;

    LOADB
    //level 0
    z0 = _mm512_set1_epi32(25847);
    z1 = _mm512_set1_epi32(1830765815);
    butterfly(b0,b8 );
    butterfly(b1,b9 );
    butterfly(b2,b10);
    butterfly(b3,b11);
    butterfly(b4,b12);
    butterfly(b5,b13);
    butterfly(b6,b14);
    butterfly(b7,b15);



    //level 1
    z0 = _mm512_set1_epi32(-2608894);
    z1 = _mm512_set1_epi32(-1929875198);
    butterfly(b0,b4);
    butterfly(b1,b5);
    butterfly(b2,b6);
    butterfly(b3,b7);
    z0 = _mm512_set1_epi32(-518909);
    z1 = _mm512_set1_epi32(-1927777021);
    butterfly(b8 ,b12);
    butterfly(b9 ,b13);
    butterfly(b10,b14);
    butterfly(b11,b15);



    //level 2
    z0 = _mm512_set1_epi32(237124);
    z1 = _mm512_set1_epi32(1640767044);
    butterfly(b0 ,b2 );
    butterfly(b1 ,b3 );
    z0 = _mm512_set1_epi32(-777960);
    z1 = _mm512_set1_epi32(1477910808);
    butterfly(b4 ,b6 );
    butterfly(b5 ,b7 );
    z0 = _mm512_set1_epi32(-876248);
    z1 = _mm512_set1_epi32(1612161320);
    butterfly(b8 ,b10);
    butterfly(b9 ,b11);
    z0 = _mm512_set1_epi32(466468);
    z1 = _mm512_set1_epi32(1640734244);
    butterfly(b12,b14);
    butterfly(b13,b15);

    //level 3
    z0 = _mm512_set1_epi32(1826347);
    z1 = _mm512_set1_epi32(308362795);
    butterfly(b0,b1 );
    z0 = _mm512_set1_epi32(2353451);
    z1 = _mm512_set1_epi32(-1815525077);
    butterfly(b2,b3 );
    z0 = _mm512_set1_epi32(-359251);
    z1 = _mm512_set1_epi32(-1374673747);
    butterfly(b4,b5);
    z0 = _mm512_set1_epi32(-2091905);
    z1 = _mm512_set1_epi32(-1091570561);
    butterfly(b6,b7);
    z0 = _mm512_set1_epi32(3119733);
    z1 = _mm512_set1_epi32(-1929495947);
    butterfly(b8,b9);
    z0 = _mm512_set1_epi32(-2884855);
    z1 = _mm512_set1_epi32(515185417);
    butterfly(b10,b11);
    z0 = _mm512_set1_epi32(3111497);
    z1 = _mm512_set1_epi32(-285697463);
    butterfly(b12,b13);
    z0 = _mm512_set1_epi32(2680103);
    z1 = _mm512_set1_epi32(625853735);
    butterfly(b14,b15);



    //level 4


    shuffle16a(b0 ,b1 ,c0 )
    shuffle16a(b2 ,b3 ,c2 )
    shuffle16a(b4 ,b5 ,c4 )
    shuffle16a(b6 ,b7 ,c6 )
    shuffle16a(b8 ,b9 ,c8 )
    shuffle16a(b10,b11,c10)
    shuffle16a(b12,b13,c12)
    shuffle16a(b14,b15,c14)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(c0,b1)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(c2,b3)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(c4,b5)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(c6,b7)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(c8,b9)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(c10,b11)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(c12,b13)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(c14,b15)


    //level 5


    shuffle8a(c0 ,b1 ,b0 )
    shuffle8a(c2 ,b3 ,b2 )
    shuffle8a(c4 ,b5 ,b4 )
    shuffle8a(c6 ,b7 ,b6 )
    shuffle8a(c8 ,b9 ,b8 )
    shuffle8a(c10,b11,b10)
    shuffle8a(c12,b13,b12)
    shuffle8a(c14,b15,b14)


    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b0,b1)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b2,b3)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b4,b5)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b6,b7)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b8,b9)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b10,b11)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b12,b13)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b14,b15)



    //level 6

    shuffle4(b0,b1)
    shuffle4(b2,b3)
    shuffle4(b4,b5)
    shuffle4(b6,b7)
    shuffle4(b8,b9)
    shuffle4(b10,b11)
    shuffle4(b12,b13)
    shuffle4(b14,b15)


    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b0,b1)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b2,b3)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b4,b5)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b6,b7)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b8,b9)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b10,b11)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b12,b13)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 =_mm512_srli_epi64(z0,32);
    butterfly(b14,b15)

    //level 7

    shuffle2(b0,b1)
    shuffle2(b2,b3)
    shuffle2(b4,b5)
    shuffle2(b6,b7)
    shuffle2(b8,b9)
    shuffle2(b10,b11)
    shuffle2(b12,b13)
    shuffle2(b14,b15)


    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly7(b0,b1)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly7(b2,b3)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly7(b4,b5)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly7(b6,b7)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly7(b8,b9)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly7(b10,b11)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly7(b12,b13)

    z0 = _mm512_load_si512(&vpsi[ptr++]);
    z1 = _mm512_load_si512(&vpsi[ptr++]);
    butterfly7(b14,b15)


    STOREB
}


#define LOAD \
z0  = _mm512_load_epi32(a      );\
z1  = _mm512_load_epi32(a + 16 );\
z2  = _mm512_load_epi32(a + 32 );\
z3  = _mm512_load_epi32(a + 48 );\
z4  = _mm512_load_epi32(a + 64 );\
z5  = _mm512_load_epi32(a + 80 );\
z6  = _mm512_load_epi32(a + 96 );\
z7  = _mm512_load_epi32(a + 112);\
z8  = _mm512_load_epi32(a + 128);\
z9  = _mm512_load_epi32(a + 144);\
z10 = _mm512_load_epi32(a + 160);\
z11 = _mm512_load_epi32(a + 176);\
z12 = _mm512_load_epi32(a + 192);\
z13 = _mm512_load_epi32(a + 208);\
z14 = _mm512_load_epi32(a + 224);\
z15 = _mm512_load_epi32(a + 240);\


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

void shuffle(int32_t a[N]) {
    __m512i z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15;
    __m512i t0, t1, t2, t3;

    const __m512i idx0 = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m512i idx1 = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i idx3 = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m512i idx4 = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 7, 9, 11, 13, 15);


    LOAD


    t0 = z0;
    t1 = z2;
    t2 = z4;
    t3 = z6;
    z0 = _mm512_maskz_permutexvar_epi32(0xff, idx0, t0) | _mm512_maskz_permutexvar_epi32(0xff00, idx1, z1);
    z1 = _mm512_maskz_permutexvar_epi32(0xff, idx3, t0) | _mm512_maskz_permutexvar_epi32(0xff00, idx4, z1);
    z2 = _mm512_maskz_permutexvar_epi32(0xff, idx0, t1) | _mm512_maskz_permutexvar_epi32(0xff00, idx1, z3);
    z3 = _mm512_maskz_permutexvar_epi32(0xff, idx3, t1) | _mm512_maskz_permutexvar_epi32(0xff00, idx4, z3);
    z4 = _mm512_maskz_permutexvar_epi32(0xff, idx0, t2) | _mm512_maskz_permutexvar_epi32(0xff00, idx1, z5);
    z5 = _mm512_maskz_permutexvar_epi32(0xff, idx3, t2) | _mm512_maskz_permutexvar_epi32(0xff00, idx4, z5);
    z6 = _mm512_maskz_permutexvar_epi32(0xff, idx0, t3) | _mm512_maskz_permutexvar_epi32(0xff00, idx1, z7);
    z7 = _mm512_maskz_permutexvar_epi32(0xff, idx3, t3) | _mm512_maskz_permutexvar_epi32(0xff00, idx4, z7);

    t0 = z8;
    t1 = z10;
    t2 = z12;
    t3 = z14;
    z8 = _mm512_maskz_permutexvar_epi32(0xff, idx0, t0) | _mm512_maskz_permutexvar_epi32(0xff00, idx1, z9);
    z9 = _mm512_maskz_permutexvar_epi32(0xff, idx3, t0) | _mm512_maskz_permutexvar_epi32(0xff00, idx4, z9);
    z10 = _mm512_maskz_permutexvar_epi32(0xff, idx0, t1) | _mm512_maskz_permutexvar_epi32(0xff00, idx1, z11);
    z11 = _mm512_maskz_permutexvar_epi32(0xff, idx3, t1) | _mm512_maskz_permutexvar_epi32(0xff00, idx4, z11);
    z12 = _mm512_maskz_permutexvar_epi32(0xff, idx0, t2) | _mm512_maskz_permutexvar_epi32(0xff00, idx1, z13);
    z13 = _mm512_maskz_permutexvar_epi32(0xff, idx3, t2) | _mm512_maskz_permutexvar_epi32(0xff00, idx4, z13);
    z14 = _mm512_maskz_permutexvar_epi32(0xff, idx0, t3) | _mm512_maskz_permutexvar_epi32(0xff00, idx1, z15);
    z15 = _mm512_maskz_permutexvar_epi32(0xff, idx3, t3) | _mm512_maskz_permutexvar_epi32(0xff00, idx4, z15);


    STORE

}





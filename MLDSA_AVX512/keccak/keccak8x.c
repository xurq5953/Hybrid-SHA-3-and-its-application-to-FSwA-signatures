//
// Created by xurq on 2022/10/18.
//
#include <x86intrin.h>
#include <stdint.h>
#include "../align.h"
#include "keccak8x.h"


ALIGN(64) static const uint64_t RC[24] = {
        0x0000000000000001ULL,
        0x0000000000008082ULL,
        0x800000000000808aULL,
        0x8000000080008000ULL,
        0x000000000000808bULL,
        0x0000000080000001ULL,
        0x8000000080008081ULL,
        0x8000000000008009ULL,
        0x000000000000008aULL,
        0x0000000000000088ULL,
        0x0000000080008009ULL,
        0x000000008000000aULL,
        0x000000008000808bULL,
        0x800000000000008bULL,
        0x8000000000008089ULL,
        0x8000000000008003ULL,
        0x8000000000008002ULL,
        0x8000000000000080ULL,
        0x000000000000800aULL,
        0x800000008000000aULL,
        0x8000000080008081ULL,
        0x8000000000008080ULL,
        0x0000000080000001ULL,
        0x8000000080008008ULL};

#define V(a, b, c, d) _mm512_ternarylogic_epi64((a), (b), (c), (d))

#define declareRegister \
__m512i c, c0, c1, c2, c3, c4;\
__m512i state00,state01,state02,state03,state04,\
        state10,state11,state12,state13,state14,\
        state20,state21,state22,state23,state24,\
        state30,state31,state32,state33,state34,\
        state40,state41,state42,state43,state44;

#define assignState \
state00 = state[ 0];\
state10 = state[ 1];\
state20 = state[ 2];\
state30 = state[ 3];\
state40 = state[ 4];\
state01 = state[ 5];\
state11 = state[ 6];\
state21 = state[ 7];\
state31 = state[ 8];\
state41 = state[ 9];\
state02 = state[10];\
state12 = state[11];\
state22 = state[12];\
state32 = state[13];\
state42 = state[14];\
state03 = state[15];\
state13 = state[16];\
state23 = state[17];\
state33 = state[18];\
state43 = state[19];\
state04 = state[20];\
state14 = state[21];\
state24 = state[22];\
state34 = state[23];\
state44 = state[24];

#define storeState \
state[ 0] = state00;\
state[ 1] = state10;\
state[ 2] = state20;\
state[ 3] = state30;\
state[ 4] = state40;\
state[ 5] = state01;\
state[ 6] = state11;\
state[ 7] = state21;\
state[ 8] = state31;\
state[ 9] = state41;\
state[10] = state02;\
state[11] = state12;\
state[12] = state22;\
state[13] = state32;\
state[14] = state42;\
state[15] = state03;\
state[16] = state13;\
state[17] = state23;\
state[18] = state33;\
state[19] = state43;\
state[20] = state04;\
state[21] = state14;\
state[22] = state24;\
state[23] = state34;\
state[24] = state44;


#define theta(a, e, o, i, u, z, r) \
c = _mm512_rol_epi64(c##z, 1); \
state##a = V(state##a, c##r, c, 0x96);\
state##e = V(state##e, c##r, c, 0x96);\
state##o = V(state##o, c##r, c, 0x96);\
state##i = V(state##i, c##r, c, 0x96);\
state##u = V(state##u, c##r, c, 0x96);\


#define pho_chi_y0(a, e, o, i, u, r) \
c1 = _mm512_rol_epi64(state##e, 44);\
c2 = _mm512_rol_epi64(state##o, 43); \
state##i = _mm512_rol_epi64(state##i, 21); \
state##u = _mm512_rol_epi64(state##u, 14); \
state##e = V(c1, c2, state##i, 0xd2); \
state##o = V(c2, state##i, state##u, 0xd2);\
state##i = V(state##i, state##u, state##a, 0xd2);  \
state##u = V(state##u, state##a, c1, 0xd2);\
state##a = V(state##a, c1, c2, 0xd2) ^ _mm512_set1_epi64(RC[r]);\


#define pho_chi_y1(a, e, o, i, u) \
c1 = _mm512_rol_epi64(state##e, 20);\
c2 = _mm512_rol_epi64(state##o, 3); \
state##i = _mm512_rol_epi64(state##i, 45); \
state##u = _mm512_rol_epi64(state##u, 61); \
state##a = _mm512_rol_epi64(state##a, 28);\
state##e = V(c1, c2, state##i, 0xd2); \
state##o = V(c2, state##i, state##u, 0xd2);\
state##i = V(state##i, state##u, state##a, 0xd2);  \
state##u = V(state##u, state##a, c1, 0xd2);\
state##a = V(state##a, c1, c2, 0xd2);

#define pho_chi_y2(a, e, o, i, u) \
c1 = _mm512_rol_epi64(state##e, 6);\
c2 = _mm512_rol_epi64(state##o, 25); \
state##i = _mm512_rol_epi64(state##i, 8); \
state##u = _mm512_rol_epi64(state##u, 18); \
state##a = _mm512_rol_epi64(state##a, 1);\
state##e = V(c1, c2, state##i, 0xd2); \
state##o = V(c2, state##i, state##u, 0xd2);\
state##i = V(state##i, state##u, state##a, 0xd2);  \
state##u = V(state##u, state##a, c1, 0xd2);\
state##a = V(state##a, c1, c2, 0xd2);

#define pho_chi_y3(a, e, o, i, u) \
c1 = _mm512_rol_epi64(state##e, 36);\
c2 = _mm512_rol_epi64(state##o, 10); \
state##i = _mm512_rol_epi64(state##i, 15); \
state##u = _mm512_rol_epi64(state##u, 56); \
state##a = _mm512_rol_epi64(state##a, 27);\
state##e = V(c1, c2, state##i, 0xd2); \
state##o = V(c2, state##i, state##u, 0xd2);\
state##i = V(state##i, state##u, state##a, 0xd2);  \
state##u = V(state##u, state##a, c1, 0xd2);\
state##a = V(state##a, c1, c2, 0xd2);

#define pho_chi_y4(a, e, o, i, u) \
c1 = _mm512_rol_epi64(state##e, 55);\
c2 = _mm512_rol_epi64(state##o, 39); \
state##i = _mm512_rol_epi64(state##i, 41); \
state##u = _mm512_rol_epi64(state##u, 2); \
state##a = _mm512_rol_epi64(state##a, 62);\
state##e = V(c1, c2, state##i, 0xd2); \
state##o = V(c2, state##i, state##u, 0xd2);\
state##i = V(state##i, state##u, state##a, 0xd2);  \
state##u = V(state##u, state##a, c1, 0xd2);\
state##a = V(state##a, c1, c2, 0xd2);


void XURQ_keccak8x_permute(__m512i *state) {
    declareRegister

    assignState

//round 0
    c0 = V(state00, state01, state02, 0x96);
    c0 = V(c0, state03, state04, 0x96);
    c1 = V(state10, state11, state12, 0x96);
    c1 = V(c1, state13, state14, 0x96);
    c2 = V(state20, state21, state22, 0x96);
    c2 = V(c2, state23, state24, 0x96);
    c3 = V(state30, state31, state32, 0x96);
    c3 = V(c3, state33, state34, 0x96);
    c4 = V(state40, state41, state42, 0x96);
    c4 = V(c4, state43, state44, 0x96);
    theta(00, 01, 02, 03, 04, 1, 4)
    theta(10, 11, 12, 13, 14, 2, 0)
    theta(20, 21, 22, 23, 24, 3, 1)
    theta(30, 31, 32, 33, 34, 4, 2)
    theta(40, 41, 42, 43, 44, 0, 3)
    pho_chi_y0(00, 11, 22, 33, 44, 0)
    pho_chi_y1(30, 41, 02, 13, 24)
    pho_chi_y2(10, 21, 32, 43, 04)
    pho_chi_y3(40, 01, 12, 23, 34)
    pho_chi_y4(20, 31, 42, 03, 14)
//round 1
    c0 = V(state00, state30, state10, 0x96);
    c0 = V(c0, state40, state20, 0x96);
    c1 = V(state11, state41, state21, 0x96);
    c1 = V(c1, state01, state31, 0x96);
    c2 = V(state22, state02, state32, 0x96);
    c2 = V(c2, state12, state42, 0x96);
    c3 = V(state33, state13, state43, 0x96);
    c3 = V(c3, state23, state03, 0x96);
    c4 = V(state44, state24, state04, 0x96);
    c4 = V(c4, state34, state14, 0x96);
    theta(00, 30, 10, 40, 20, 1, 4)
    theta(11, 41, 21, 01, 31, 2, 0)
    theta(22, 02, 32, 12, 42, 3, 1)
    theta(33, 13, 43, 23, 03, 4, 2)
    theta(44, 24, 04, 34, 14, 0, 3)
    pho_chi_y0(00, 41, 32, 23, 14, 1)
    pho_chi_y1(33, 24, 10, 01, 42)
    pho_chi_y2(11, 02, 43, 34, 20)
    pho_chi_y3(44, 30, 21, 12, 03)
    pho_chi_y4(22, 13, 04, 40, 31)
//round 2
    c0 = V(state00, state33, state11, 0x96);
    c0 = V(c0, state44, state22, 0x96);
    c1 = V(state41, state24, state02, 0x96);
    c1 = V(c1, state30, state13, 0x96);
    c2 = V(state32, state10, state43, 0x96);
    c2 = V(c2, state21, state04, 0x96);
    c3 = V(state23, state01, state34, 0x96);
    c3 = V(c3, state12, state40, 0x96);
    c4 = V(state14, state42, state20, 0x96);
    c4 = V(c4, state03, state31, 0x96);
    theta(00, 33, 11, 44, 22, 1, 4)
    theta(41, 24, 02, 30, 13, 2, 0)
    theta(32, 10, 43, 21, 04, 3, 1)
    theta(23, 01, 34, 12, 40, 4, 2)
    theta(14, 42, 20, 03, 31, 0, 3)
    pho_chi_y0(00, 24, 43, 12, 31, 2)
    pho_chi_y1(23, 42, 11, 30, 04)
    pho_chi_y2(41, 10, 34, 03, 22)
    pho_chi_y3(14, 33, 02, 21, 40)
    pho_chi_y4(32, 01, 20, 44, 13)
//round 3
    c0 = V(state00, state23, state41, 0x96);
    c0 = V(c0, state14, state32, 0x96);
    c1 = V(state24, state42, state10, 0x96);
    c1 = V(c1, state33, state01, 0x96);
    c2 = V(state43, state11, state34, 0x96);
    c2 = V(c2, state02, state20, 0x96);
    c3 = V(state12, state30, state03, 0x96);
    c3 = V(c3, state21, state44, 0x96);
    c4 = V(state31, state04, state22, 0x96);
    c4 = V(c4, state40, state13, 0x96);
    theta(00, 23, 41, 14, 32, 1, 4)
    theta(24, 42, 10, 33, 01, 2, 0)
    theta(43, 11, 34, 02, 20, 3, 1)
    theta(12, 30, 03, 21, 44, 4, 2)
    theta(31, 04, 22, 40, 13, 0, 3)
    pho_chi_y0(00, 42, 34, 21, 13, 3)
    pho_chi_y1(12, 04, 41, 33, 20)
    pho_chi_y2(24, 11, 03, 40, 32)
    pho_chi_y3(31, 23, 10, 02, 44)
    pho_chi_y4(43, 30, 22, 14, 01)
//round 4
    c0 = V(state00, state12, state24, 0x96);
    c0 = V(c0, state31, state43, 0x96);
    c1 = V(state42, state04, state11, 0x96);
    c1 = V(c1, state23, state30, 0x96);
    c2 = V(state34, state41, state03, 0x96);
    c2 = V(c2, state10, state22, 0x96);
    c3 = V(state21, state33, state40, 0x96);
    c3 = V(c3, state02, state14, 0x96);
    c4 = V(state13, state20, state32, 0x96);
    c4 = V(c4, state44, state01, 0x96);
    theta(00, 12, 24, 31, 43, 1, 4)
    theta(42, 04, 11, 23, 30, 2, 0)
    theta(34, 41, 03, 10, 22, 3, 1)
    theta(21, 33, 40, 02, 14, 4, 2)
    theta(13, 20, 32, 44, 01, 0, 3)
    pho_chi_y0(00, 04, 03, 02, 01, 4)
    pho_chi_y1(21, 20, 24, 23, 22)
    pho_chi_y2(42, 41, 40, 44, 43)
    pho_chi_y3(13, 12, 11, 10, 14)
    pho_chi_y4(34, 33, 32, 31, 30)
//round 5
    c0 = V(state00, state21, state42, 0x96);
    c0 = V(c0, state13, state34, 0x96);
    c1 = V(state04, state20, state41, 0x96);
    c1 = V(c1, state12, state33, 0x96);
    c2 = V(state03, state24, state40, 0x96);
    c2 = V(c2, state11, state32, 0x96);
    c3 = V(state02, state23, state44, 0x96);
    c3 = V(c3, state10, state31, 0x96);
    c4 = V(state01, state22, state43, 0x96);
    c4 = V(c4, state14, state30, 0x96);
    theta(00, 21, 42, 13, 34, 1, 4)
    theta(04, 20, 41, 12, 33, 2, 0)
    theta(03, 24, 40, 11, 32, 3, 1)
    theta(02, 23, 44, 10, 31, 4, 2)
    theta(01, 22, 43, 14, 30, 0, 3)
    pho_chi_y0(00, 20, 40, 10, 30, 5)
    pho_chi_y1(02, 22, 42, 12, 32)
    pho_chi_y2(04, 24, 44, 14, 34)
    pho_chi_y3(01, 21, 41, 11, 31)
    pho_chi_y4(03, 23, 43, 13, 33)
//round 6
    c0 = V(state00, state02, state04, 0x96);
    c0 = V(c0, state01, state03, 0x96);
    c1 = V(state20, state22, state24, 0x96);
    c1 = V(c1, state21, state23, 0x96);
    c2 = V(state40, state42, state44, 0x96);
    c2 = V(c2, state41, state43, 0x96);
    c3 = V(state10, state12, state14, 0x96);
    c3 = V(c3, state11, state13, 0x96);
    c4 = V(state30, state32, state34, 0x96);
    c4 = V(c4, state31, state33, 0x96);
    theta(00, 02, 04, 01, 03, 1, 4)
    theta(20, 22, 24, 21, 23, 2, 0)
    theta(40, 42, 44, 41, 43, 3, 1)
    theta(10, 12, 14, 11, 13, 4, 2)
    theta(30, 32, 34, 31, 33, 0, 3)
    pho_chi_y0(00, 22, 44, 11, 33, 6)
    pho_chi_y1(10, 32, 04, 21, 43)
    pho_chi_y2(20, 42, 14, 31, 03)
    pho_chi_y3(30, 02, 24, 41, 13)
    pho_chi_y4(40, 12, 34, 01, 23)
//round 7
    c0 = V(state00, state10, state20, 0x96);
    c0 = V(c0, state30, state40, 0x96);
    c1 = V(state22, state32, state42, 0x96);
    c1 = V(c1, state02, state12, 0x96);
    c2 = V(state44, state04, state14, 0x96);
    c2 = V(c2, state24, state34, 0x96);
    c3 = V(state11, state21, state31, 0x96);
    c3 = V(c3, state41, state01, 0x96);
    c4 = V(state33, state43, state03, 0x96);
    c4 = V(c4, state13, state23, 0x96);
    theta(00, 10, 20, 30, 40, 1, 4)
    theta(22, 32, 42, 02, 12, 2, 0)
    theta(44, 04, 14, 24, 34, 3, 1)
    theta(11, 21, 31, 41, 01, 4, 2)
    theta(33, 43, 03, 13, 23, 0, 3)
    pho_chi_y0(00, 32, 14, 41, 23, 7)
    pho_chi_y1(11, 43, 20, 02, 34)
    pho_chi_y2(22, 04, 31, 13, 40)
    pho_chi_y3(33, 10, 42, 24, 01)
    pho_chi_y4(44, 21, 03, 30, 12)
//round 8
    c0 = V(state00, state11, state22, 0x96);
    c0 = V(c0, state33, state44, 0x96);
    c1 = V(state32, state43, state04, 0x96);
    c1 = V(c1, state10, state21, 0x96);
    c2 = V(state14, state20, state31, 0x96);
    c2 = V(c2, state42, state03, 0x96);
    c3 = V(state41, state02, state13, 0x96);
    c3 = V(c3, state24, state30, 0x96);
    c4 = V(state23, state34, state40, 0x96);
    c4 = V(c4, state01, state12, 0x96);
    theta(00, 11, 22, 33, 44, 1, 4)
    theta(32, 43, 04, 10, 21, 2, 0)
    theta(14, 20, 31, 42, 03, 3, 1)
    theta(41, 02, 13, 24, 30, 4, 2)
    theta(23, 34, 40, 01, 12, 0, 3)
    pho_chi_y0(00, 43, 31, 24, 12, 8)
    pho_chi_y1(41, 34, 22, 10, 03)
    pho_chi_y2(32, 20, 13, 01, 44)
    pho_chi_y3(23, 11, 04, 42, 30)
    pho_chi_y4(14, 02, 40, 33, 21)
//round 9
    c0 = V(state00, state41, state32, 0x96);
    c0 = V(c0, state23, state14, 0x96);
    c1 = V(state43, state34, state20, 0x96);
    c1 = V(c1, state11, state02, 0x96);
    c2 = V(state31, state22, state13, 0x96);
    c2 = V(c2, state04, state40, 0x96);
    c3 = V(state24, state10, state01, 0x96);
    c3 = V(c3, state42, state33, 0x96);
    c4 = V(state12, state03, state44, 0x96);
    c4 = V(c4, state30, state21, 0x96);
    theta(00, 41, 32, 23, 14, 1, 4)
    theta(43, 34, 20, 11, 02, 2, 0)
    theta(31, 22, 13, 04, 40, 3, 1)
    theta(24, 10, 01, 42, 33, 4, 2)
    theta(12, 03, 44, 30, 21, 0, 3)
    pho_chi_y0(00, 34, 13, 42, 21, 9)
    pho_chi_y1(24, 03, 32, 11, 40)
    pho_chi_y2(43, 22, 01, 30, 14)
    pho_chi_y3(12, 41, 20, 04, 33)
    pho_chi_y4(31, 10, 44, 23, 02)
//round 10
    c0 = V(state00, state24, state43, 0x96);
    c0 = V(c0, state12, state31, 0x96);
    c1 = V(state34, state03, state22, 0x96);
    c1 = V(c1, state41, state10, 0x96);
    c2 = V(state13, state32, state01, 0x96);
    c2 = V(c2, state20, state44, 0x96);
    c3 = V(state42, state11, state30, 0x96);
    c3 = V(c3, state04, state23, 0x96);
    c4 = V(state21, state40, state14, 0x96);
    c4 = V(c4, state33, state02, 0x96);
    theta(00, 24, 43, 12, 31, 1, 4)
    theta(34, 03, 22, 41, 10, 2, 0)
    theta(13, 32, 01, 20, 44, 3, 1)
    theta(42, 11, 30, 04, 23, 4, 2)
    theta(21, 40, 14, 33, 02, 0, 3)
    pho_chi_y0(00, 03, 01, 04, 02, 10)
    pho_chi_y1(42, 40, 43, 41, 44)
    pho_chi_y2(34, 32, 30, 33, 31)
    pho_chi_y3(21, 24, 22, 20, 23)
    pho_chi_y4(13, 11, 14, 12, 10)
//round 11
    c0 = V(state00, state42, state34, 0x96);
    c0 = V(c0, state21, state13, 0x96);
    c1 = V(state03, state40, state32, 0x96);
    c1 = V(c1, state24, state11, 0x96);
    c2 = V(state01, state43, state30, 0x96);
    c2 = V(c2, state22, state14, 0x96);
    c3 = V(state04, state41, state33, 0x96);
    c3 = V(c3, state20, state12, 0x96);
    c4 = V(state02, state44, state31, 0x96);
    c4 = V(c4, state23, state10, 0x96);
    theta(00, 42, 34, 21, 13, 1, 4)
    theta(03, 40, 32, 24, 11, 2, 0)
    theta(01, 43, 30, 22, 14, 3, 1)
    theta(04, 41, 33, 20, 12, 4, 2)
    theta(02, 44, 31, 23, 10, 0, 3)
    pho_chi_y0(00, 40, 30, 20, 10, 11)
    pho_chi_y1(04, 44, 34, 24, 14)
    pho_chi_y2(03, 43, 33, 23, 13)
    pho_chi_y3(02, 42, 32, 22, 12)
    pho_chi_y4(01, 41, 31, 21, 11)
//round 12
    c0 = V(state00, state04, state03, 0x96);
    c0 = V(c0, state02, state01, 0x96);
    c1 = V(state40, state44, state43, 0x96);
    c1 = V(c1, state42, state41, 0x96);
    c2 = V(state30, state34, state33, 0x96);
    c2 = V(c2, state32, state31, 0x96);
    c3 = V(state20, state24, state23, 0x96);
    c3 = V(c3, state22, state21, 0x96);
    c4 = V(state10, state14, state13, 0x96);
    c4 = V(c4, state12, state11, 0x96);
    theta(00, 04, 03, 02, 01, 1, 4)
    theta(40, 44, 43, 42, 41, 2, 0)
    theta(30, 34, 33, 32, 31, 3, 1)
    theta(20, 24, 23, 22, 21, 4, 2)
    theta(10, 14, 13, 12, 11, 0, 3)
    pho_chi_y0(00, 44, 33, 22, 11, 12)
    pho_chi_y1(20, 14, 03, 42, 31)
    pho_chi_y2(40, 34, 23, 12, 01)
    pho_chi_y3(10, 04, 43, 32, 21)
    pho_chi_y4(30, 24, 13, 02, 41)
//round 13
    c0 = V(state00, state20, state40, 0x96);
    c0 = V(c0, state10, state30, 0x96);
    c1 = V(state44, state14, state34, 0x96);
    c1 = V(c1, state04, state24, 0x96);
    c2 = V(state33, state03, state23, 0x96);
    c2 = V(c2, state43, state13, 0x96);
    c3 = V(state22, state42, state12, 0x96);
    c3 = V(c3, state32, state02, 0x96);
    c4 = V(state11, state31, state01, 0x96);
    c4 = V(c4, state21, state41, 0x96);
    theta(00, 20, 40, 10, 30, 1, 4)
    theta(44, 14, 34, 04, 24, 2, 0)
    theta(33, 03, 23, 43, 13, 3, 1)
    theta(22, 42, 12, 32, 02, 4, 2)
    theta(11, 31, 01, 21, 41, 0, 3)
    pho_chi_y0(00, 14, 23, 32, 41, 13)
    pho_chi_y1(22, 31, 40, 04, 13)
    pho_chi_y2(44, 03, 12, 21, 30)
    pho_chi_y3(11, 20, 34, 43, 02)
    pho_chi_y4(33, 42, 01, 10, 24)
//round 14
    c0 = V(state00, state22, state44, 0x96);
    c0 = V(c0, state11, state33, 0x96);
    c1 = V(state14, state31, state03, 0x96);
    c1 = V(c1, state20, state42, 0x96);
    c2 = V(state23, state40, state12, 0x96);
    c2 = V(c2, state34, state01, 0x96);
    c3 = V(state32, state04, state21, 0x96);
    c3 = V(c3, state43, state10, 0x96);
    c4 = V(state41, state13, state30, 0x96);
    c4 = V(c4, state02, state24, 0x96);
    theta(00, 22, 44, 11, 33, 1, 4)
    theta(14, 31, 03, 20, 42, 2, 0)
    theta(23, 40, 12, 34, 01, 3, 1)
    theta(32, 04, 21, 43, 10, 4, 2)
    theta(41, 13, 30, 02, 24, 0, 3)
    pho_chi_y0(00, 31, 12, 43, 24, 14)
    pho_chi_y1(32, 13, 44, 20, 01)
    pho_chi_y2(14, 40, 21, 02, 33)
    pho_chi_y3(41, 22, 03, 34, 10)
    pho_chi_y4(23, 04, 30, 11, 42)
//round 15
    c0 = V(state00, state32, state14, 0x96);
    c0 = V(c0, state41, state23, 0x96);
    c1 = V(state31, state13, state40, 0x96);
    c1 = V(c1, state22, state04, 0x96);
    c2 = V(state12, state44, state21, 0x96);
    c2 = V(c2, state03, state30, 0x96);
    c3 = V(state43, state20, state02, 0x96);
    c3 = V(c3, state34, state11, 0x96);
    c4 = V(state24, state01, state33, 0x96);
    c4 = V(c4, state10, state42, 0x96);
    theta(00, 32, 14, 41, 23, 1, 4)
    theta(31, 13, 40, 22, 04, 2, 0)
    theta(12, 44, 21, 03, 30, 3, 1)
    theta(43, 20, 02, 34, 11, 4, 2)
    theta(24, 01, 33, 10, 42, 0, 3)
    pho_chi_y0(00, 13, 21, 34, 42, 15)
    pho_chi_y1(43, 01, 14, 22, 30)
    pho_chi_y2(31, 44, 02, 10, 23)
    pho_chi_y3(24, 32, 40, 03, 11)
    pho_chi_y4(12, 20, 33, 41, 04)
//round 16
    c0 = V(state00, state43, state31, 0x96);
    c0 = V(c0, state24, state12, 0x96);
    c1 = V(state13, state01, state44, 0x96);
    c1 = V(c1, state32, state20, 0x96);
    c2 = V(state21, state14, state02, 0x96);
    c2 = V(c2, state40, state33, 0x96);
    c3 = V(state34, state22, state10, 0x96);
    c3 = V(c3, state03, state41, 0x96);
    c4 = V(state42, state30, state23, 0x96);
    c4 = V(c4, state11, state04, 0x96);
    theta(00, 43, 31, 24, 12, 1, 4)
    theta(13, 01, 44, 32, 20, 2, 0)
    theta(21, 14, 02, 40, 33, 3, 1)
    theta(34, 22, 10, 03, 41, 4, 2)
    theta(42, 30, 23, 11, 04, 0, 3)
    pho_chi_y0(00, 01, 02, 03, 04, 16)
    pho_chi_y1(34, 30, 31, 32, 33)
    pho_chi_y2(13, 14, 10, 11, 12)
    pho_chi_y3(42, 43, 44, 40, 41)
    pho_chi_y4(21, 22, 23, 24, 20)
//round 17
    c0 = V(state00, state34, state13, 0x96);
    c0 = V(c0, state42, state21, 0x96);
    c1 = V(state01, state30, state14, 0x96);
    c1 = V(c1, state43, state22, 0x96);
    c2 = V(state02, state31, state10, 0x96);
    c2 = V(c2, state44, state23, 0x96);
    c3 = V(state03, state32, state11, 0x96);
    c3 = V(c3, state40, state24, 0x96);
    c4 = V(state04, state33, state12, 0x96);
    c4 = V(c4, state41, state20, 0x96);
    theta(00, 34, 13, 42, 21, 1, 4)
    theta(01, 30, 14, 43, 22, 2, 0)
    theta(02, 31, 10, 44, 23, 3, 1)
    theta(03, 32, 11, 40, 24, 4, 2)
    theta(04, 33, 12, 41, 20, 0, 3)
    pho_chi_y0(00, 30, 10, 40, 20, 17)
    pho_chi_y1(03, 33, 13, 43, 23)
    pho_chi_y2(01, 31, 11, 41, 21)
    pho_chi_y3(04, 34, 14, 44, 24)
    pho_chi_y4(02, 32, 12, 42, 22)
//round 18
    c0 = V(state00, state03, state01, 0x96);
    c0 = V(c0, state04, state02, 0x96);
    c1 = V(state30, state33, state31, 0x96);
    c1 = V(c1, state34, state32, 0x96);
    c2 = V(state10, state13, state11, 0x96);
    c2 = V(c2, state14, state12, 0x96);
    c3 = V(state40, state43, state41, 0x96);
    c3 = V(c3, state44, state42, 0x96);
    c4 = V(state20, state23, state21, 0x96);
    c4 = V(c4, state24, state22, 0x96);
    theta(00, 03, 01, 04, 02, 1, 4)
    theta(30, 33, 31, 34, 32, 2, 0)
    theta(10, 13, 11, 14, 12, 3, 1)
    theta(40, 43, 41, 44, 42, 4, 2)
    theta(20, 23, 21, 24, 22, 0, 3)
    pho_chi_y0(00, 33, 11, 44, 22, 18)
    pho_chi_y1(40, 23, 01, 34, 12)
    pho_chi_y2(30, 13, 41, 24, 02)
    pho_chi_y3(20, 03, 31, 14, 42)
    pho_chi_y4(10, 43, 21, 04, 32)
//round 19
    c0 = V(state00, state40, state30, 0x96);
    c0 = V(c0, state20, state10, 0x96);
    c1 = V(state33, state23, state13, 0x96);
    c1 = V(c1, state03, state43, 0x96);
    c2 = V(state11, state01, state41, 0x96);
    c2 = V(c2, state31, state21, 0x96);
    c3 = V(state44, state34, state24, 0x96);
    c3 = V(c3, state14, state04, 0x96);
    c4 = V(state22, state12, state02, 0x96);
    c4 = V(c4, state42, state32, 0x96);
    theta(00, 40, 30, 20, 10, 1, 4)
    theta(33, 23, 13, 03, 43, 2, 0)
    theta(11, 01, 41, 31, 21, 3, 1)
    theta(44, 34, 24, 14, 04, 4, 2)
    theta(22, 12, 02, 42, 32, 0, 3)
    pho_chi_y0(00, 23, 41, 14, 32, 19)
    pho_chi_y1(44, 12, 30, 03, 21)
    pho_chi_y2(33, 01, 24, 42, 10)
    pho_chi_y3(22, 40, 13, 31, 04)
    pho_chi_y4(11, 34, 02, 20, 43)
//round 20
    c0 = V(state00, state44, state33, 0x96);
    c0 = V(c0, state22, state11, 0x96);
    c1 = V(state23, state12, state01, 0x96);
    c1 = V(c1, state40, state34, 0x96);
    c2 = V(state41, state30, state24, 0x96);
    c2 = V(c2, state13, state02, 0x96);
    c3 = V(state14, state03, state42, 0x96);
    c3 = V(c3, state31, state20, 0x96);
    c4 = V(state32, state21, state10, 0x96);
    c4 = V(c4, state04, state43, 0x96);
    theta(00, 44, 33, 22, 11, 1, 4)
    theta(23, 12, 01, 40, 34, 2, 0)
    theta(41, 30, 24, 13, 02, 3, 1)
    theta(14, 03, 42, 31, 20, 4, 2)
    theta(32, 21, 10, 04, 43, 0, 3)
    pho_chi_y0(00, 12, 24, 31, 43, 20)
    pho_chi_y1(14, 21, 33, 40, 02)
    pho_chi_y2(23, 30, 42, 04, 11)
    pho_chi_y3(32, 44, 01, 13, 20)
    pho_chi_y4(41, 03, 10, 22, 34)
//round 21
    c0 = V(state00, state14, state23, 0x96);
    c0 = V(c0, state32, state41, 0x96);
    c1 = V(state12, state21, state30, 0x96);
    c1 = V(c1, state44, state03, 0x96);
    c2 = V(state24, state33, state42, 0x96);
    c2 = V(c2, state01, state10, 0x96);
    c3 = V(state31, state40, state04, 0x96);
    c3 = V(c3, state13, state22, 0x96);
    c4 = V(state43, state02, state11, 0x96);
    c4 = V(c4, state20, state34, 0x96);
    theta(00, 14, 23, 32, 41, 1, 4)
    theta(12, 21, 30, 44, 03, 2, 0)
    theta(24, 33, 42, 01, 10, 3, 1)
    theta(31, 40, 04, 13, 22, 4, 2)
    theta(43, 02, 11, 20, 34, 0, 3)
    pho_chi_y0(00, 21, 42, 13, 34, 21)
    pho_chi_y1(31, 02, 23, 44, 10)
    pho_chi_y2(12, 33, 04, 20, 41)
    pho_chi_y3(43, 14, 30, 01, 22)
    pho_chi_y4(24, 40, 11, 32, 03)
//round 22
    c0 = V(state00, state31, state12, 0x96);
    c0 = V(c0, state43, state24, 0x96);
    c1 = V(state21, state02, state33, 0x96);
    c1 = V(c1, state14, state40, 0x96);
    c2 = V(state42, state23, state04, 0x96);
    c2 = V(c2, state30, state11, 0x96);
    c3 = V(state13, state44, state20, 0x96);
    c3 = V(c3, state01, state32, 0x96);
    c4 = V(state34, state10, state41, 0x96);
    c4 = V(c4, state22, state03, 0x96);
    theta(00, 31, 12, 43, 24, 1, 4)
    theta(21, 02, 33, 14, 40, 2, 0)
    theta(42, 23, 04, 30, 11, 3, 1)
    theta(13, 44, 20, 01, 32, 4, 2)
    theta(34, 10, 41, 22, 03, 0, 3)
    pho_chi_y0(00, 02, 04, 01, 03, 22)
    pho_chi_y1(13, 10, 12, 14, 11)
    pho_chi_y2(21, 23, 20, 22, 24)
    pho_chi_y3(34, 31, 33, 30, 32)
    pho_chi_y4(42, 44, 41, 43, 40)
//round 23
    c0 = V(state00, state13, state21, 0x96);
    c0 = V(c0, state34, state42, 0x96);
    c1 = V(state02, state10, state23, 0x96);
    c1 = V(c1, state31, state44, 0x96);
    c2 = V(state04, state12, state20, 0x96);
    c2 = V(c2, state33, state41, 0x96);
    c3 = V(state01, state14, state22, 0x96);
    c3 = V(c3, state30, state43, 0x96);
    c4 = V(state03, state11, state24, 0x96);
    c4 = V(c4, state32, state40, 0x96);
    theta(00, 13, 21, 34, 42, 1, 4)
    theta(02, 10, 23, 31, 44, 2, 0)
    theta(04, 12, 20, 33, 41, 3, 1)
    theta(01, 14, 22, 30, 43, 4, 2)
    theta(03, 11, 24, 32, 40, 0, 3)
    pho_chi_y0(00, 10, 20, 30, 40, 23)
    pho_chi_y1(01, 11, 21, 31, 41)
    pho_chi_y2(02, 12, 22, 32, 42)
    pho_chi_y3(03, 13, 23, 33, 43)
    pho_chi_y4(04, 14, 24, 34, 44)


    storeState
}

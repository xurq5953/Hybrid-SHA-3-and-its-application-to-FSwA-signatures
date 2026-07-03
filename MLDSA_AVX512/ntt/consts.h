#ifndef PQCLEAN_DILITHIUM2_AVX2_CONSTS_H
#define PQCLEAN_DILITHIUM2_AVX2_CONSTS_H
#include "align.h"
#include "cdecl.h"


typedef ALIGNED_INT32(624) qdata_t;
extern const qdata_t PQCLEAN_DILITHIUM2_AVX2_qdata;

extern const int32_t qdata2[];

extern const int32_t psi[];

extern const int32_t ipsi[];

extern const int64_t ipsi2[];

extern const int32_t qdata[];

#define QINV16X 0
#define Q16X 16
#define DIV16X 32
#define ZETAS 48
#define ZETAS_INV 560

#endif

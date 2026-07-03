//
// Created by xurq on 2022/10/18.
//

#ifndef DILITHIUM2AVX_KECCAK8X_H
#define DILITHIUM2AVX_KECCAK8X_H

#include <x86intrin.h>


void XURQ_keccak8x_permute(__m512i *state);

#endif //DILITHIUM2AVX_KECCAK8X_H

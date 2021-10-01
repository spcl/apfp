#pragma once

#include <gmp.h>

#include "Config.h"

constexpr int kMantissaBytes = kMantissaBits / 8;

using Limb = mp_limb_t;
using Exponent = mp_exp_t;
using Sign = int;

struct Mantissa {
    Limb limbs[kMantissaBytes / sizeof(Limb)];
};
static_assert(sizeof(Mantissa) == kMantissaBytes, "Mantissa must be tightly packed.");

extern "C" void MatrixMultiplication(Mantissa *mantissa_a, Exponent *exponent_a, Sign *sign_a, Mantissa *mantissa_b,
                                     Exponent *exponent_b, Sign *sign_b, Mantissa *mantissa_c, Exponent *exponent_c,
                                     Sign *sign_c, int n, int m, int k);

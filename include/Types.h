#pragma once

#include <ap_int.h>
#include <gmp.h>
#include <mpfr.h>

#include <cstdint>

#include "Config.h"

using Limb = mp_limb_t;
#ifdef APFP_USE_GMP_SEMANTICS
using Exponent = mp_exp_t;
#else
using Exponent = mpfr_exp_t;
#endif
/// Sign convention: 0 is positive, anything else is negative
using Sign = Exponent;  // This will make sure that we are always limb-aligned
constexpr int kMantissaBytes = kBytes - sizeof(Exponent) - sizeof(Sign);
constexpr int kMantissaBits = 8 * kMantissaBytes;
using Mantissa = uint8_t[kMantissaBytes];
static_assert(sizeof(Mantissa) == kMantissaBytes, "Mantissa must be tightly packed.");
static_assert(kMantissaBytes % sizeof(Limb) == 0, "Mantissa size must be a multiple of the GMP/MPFR limb size.");

// This is the only MPFR rounding mode supported, so use it throughout
constexpr auto kRoundingMode = MPFR_RNDZ;

using DramLine = ap_uint<512>;
static_assert(sizeof(DramLine) == 64, "DRAM lines must be tightly packed.");

constexpr int kLinesPerNumber = kBytes / sizeof(DramLine);
static_assert(kBytes % sizeof(DramLine) == 0, "Numbers must be a multiple of DRAM lines.");

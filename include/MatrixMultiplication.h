#pragma once

#include <ap_int.h>
#include <gmp.h>

#include <cstdint>  // uint8_t
#include <cstring>  // std::memcpy
#include <sstream>

#include "Config.h"

constexpr int kBytes = kBits / 8;

using Limb = mp_limb_t;
using Exponent = mp_exp_t;
/// Sign convention: 0 is positive, anything else is negative
using Sign = int;  // Word-aligned because this probably makes copies faster on the CPU

using DramLine = ap_uint<512>;
static_assert(sizeof(DramLine) == 64, "DRAM lines must be tightly packed.");

/// Full floating point number densely packed to fit into a 512-bit DRAM line.
#pragma pack(push, 1)
class PackedFloat {
   public:
    PackedFloat() {
        // Leave stuff uninitialized by default
    }

    // Use default copy/move constructors and assignments
    PackedFloat(PackedFloat const &) = default;
    PackedFloat(PackedFloat &&) = default;
    PackedFloat &operator=(PackedFloat const &) = default;
    PackedFloat &operator=(PackedFloat &&) = default;

#ifndef HLSLIB_SYNTHESIS  // Interoperability with GMP, but only on the host side
    PackedFloat(mpf_t num) {
        const auto gmp_bits = mpf_get_prec(num);
#ifndef NDEBUG
        if (gmp_bits > kBits) {
            std::stringstream ss;
            ss << "Cannot fit GMP number with " << gmp_bits << " bits (maximum supported is " << kBits << " bits).\n";
            throw std::runtime_error(ss.str());
        }
#endif
        std::memcpy(mantissa, num->_mp_d, gmp_bits / 8);
        exponent = num->_mp_exp;
        sign = num->_mp_size < 0;  // 1 if negative, 0 otherwise
    }

    void ToGmp(mpf_t num) {
#ifndef NDEBUG
        const auto gmp_bits = mpf_get_prec(num);
        if (gmp_bits < kBits) {
            std::stringstream ss;
            ss << "GMP precision of " << gmp_bits << " bits is too low to fit " << kBits << "-bit APFP numbers.\n";
            throw std::runtime_error(ss.str());
        }
#endif
        num->_mp_exp = exponent;
        num->_mp_size = (sign != 0) ? -kBytes / int(sizeof(Limb)) : kBytes / sizeof(Limb);
        std::memcpy(num->_mp_d, mantissa, kBytes);
    }

    PackedFloat &operator=(mpf_t num) {
        *this = PackedFloat(num);
        return *this;
    }

    __mpf_struct ToGmp() {
        __mpf_struct num;
        mpf_init2(&num, kBits);
        ToGmp(&num);
        return num;
    }
#endif  // End interoperability with GMP

    // Fields are left public
    uint8_t mantissa[kBytes - sizeof(Exponent) - sizeof(Sign)];
    Exponent exponent;
    Sign sign;
};
#pragma pack(pop)
static_assert(sizeof(PackedFloat) == kBytes, "Numbers must be tightly packed.");

constexpr int kLinesPerNumber = sizeof(PackedFloat) / sizeof(DramLine);
static_assert(sizeof(PackedFloat) % sizeof(DramLine) == 0, "Numbers must be a multiple of DRAM lines.");

extern "C" void MatrixMultiplication(DramLine const *a, DramLine const *b, DramLine const *c_read, DramLine *c_write,
                                     int n, int m, int k);

#pragma once

#include <gmp.h>
#include <mpfr.h>

#include <algorithm>  // std::min
#include <cassert>
#include <cstdint>  // uint8_t
#include <cstring>  // std::memcpy
#include <iomanip>  // std::setfill, std::setw
#include <iostream>
#include <sstream>

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

/// Full floating point number densely packed to fit into a 512-bit DRAM line.
#pragma pack(push, 1)
class PackedFloat {
   public:
    inline PackedFloat() {
        // Leave stuff uninitialized by default
    }

    // Use default copy/move constructors and assignments
    inline PackedFloat(PackedFloat const &) = default;
    inline PackedFloat(PackedFloat &&) = default;
    inline PackedFloat &operator=(PackedFloat const &) = default;
    inline PackedFloat &operator=(PackedFloat &&) = default;

    inline PackedFloat(Sign const &_sign, Exponent const &_exponent, void const *const _mantissa)
        : exponent(_exponent), sign(_sign) {
        std::memcpy(mantissa, _mantissa, kMantissaBytes);
    }

    inline Limb operator[](const size_t i) const {
        return reinterpret_cast<mp_limb_t const *>(mantissa)[i];
    }

#ifndef HLSLIB_SYNTHESIS  // Interoperability with GMP/MPFR, but only on the host side
    inline PackedFloat(mpf_srcptr num) {
        // Copy the most significant bytes, padding zeros if necessary
        const auto num_limbs = std::min(size_t(std::abs(num->_mp_size)),
                                        (mpf_get_prec(num) + 8 * sizeof(mp_limb_t) - 1) / (8 * sizeof(mp_limb_t)));
        const int gmp_bytes = num_limbs * sizeof(mp_limb_t);
        const size_t bytes_to_copy = std::min(gmp_bytes, kMantissaBytes);
        const size_t copy_from = gmp_bytes - bytes_to_copy;
        std::memset(mantissa + bytes_to_copy, 0x0, kMantissaBytes - bytes_to_copy);
        std::memcpy(mantissa, reinterpret_cast<uint8_t const *>(num->_mp_d) + copy_from, bytes_to_copy);
        exponent = num->_mp_exp - num_limbs + 1;
        sign = num->_mp_size < 0;  // 1 if negative, 0 otherwise
    }

    inline PackedFloat(const mpfr_srcptr num) {
        // Copy the most significant bytes, padding zeros if necessary
        const auto mpfr_limbs = (mpfr_get_prec(num) + 8 * sizeof(mp_limb_t) - 1) / (8 * sizeof(mp_limb_t));
        const size_t mpfr_bytes = mpfr_limbs * sizeof(mp_limb_t);
        const size_t bytes_to_copy = std::min(mpfr_bytes, size_t(kMantissaBytes));
        const size_t copy_from = mpfr_bytes - bytes_to_copy;
        const size_t copy_to = kMantissaBytes - bytes_to_copy;
        std::memset(mantissa, 0x0, copy_to);
        std::memcpy(mantissa + copy_to, reinterpret_cast<uint8_t const *>(num->_mpfr_d) + copy_from, bytes_to_copy);
        exponent = num->_mpfr_exp;
        sign = num->_mpfr_sign < 0;  // 1 if negative, 0 otherwise
    }

    inline PackedFloat &operator=(mpf_srcptr num) {
        *this = PackedFloat(num);
        return *this;
    }

    inline void ToGmp(mpf_ptr num) const {
        const size_t gmp_limbs = (mpf_get_prec(num) + 8 * sizeof(mp_limb_t) - 1) / (8 * sizeof(mp_limb_t));
        constexpr size_t kNumLimbs = kMantissaBytes / sizeof(Limb);
        // GMP does not allow graceful rounding, so we cannot handle having insufficient bits in the target GMP number
        assert(gmp_limbs >= kNumLimbs);
        num->_mp_size = 0;
        for (size_t i = 0; i < kNumLimbs; ++i) {
            const auto limb = (*this)[i];
            num->_mp_d[i] = limb;
            if (limb > 0) {
                num->_mp_size = i + 1;
            }
        }
        num->_mp_exp = exponent + num->_mp_size - 1;
        if (sign) {
            num->_mp_size = -num->_mp_size;
        }
    }

    inline void ToMpfr(mpfr_ptr num) const {
        // Copy the most significant bytes, padding zeros if necessary
        const auto mpfr_limbs = (mpfr_get_prec(num) + 8 * sizeof(mp_limb_t) - 1) / (8 * sizeof(mp_limb_t));
        const size_t mpfr_bytes = mpfr_limbs * sizeof(mp_limb_t);
        const size_t bytes_to_copy = std::min(mpfr_bytes, size_t(kMantissaBytes));
        const auto copy_to = mpfr_bytes - bytes_to_copy;
        const auto copy_from = kMantissaBytes - bytes_to_copy;
        std::memset(num->_mpfr_d, 0x0, copy_to);
        std::memcpy(reinterpret_cast<uint8_t *>(num->_mpfr_d) + copy_to, mantissa + copy_from, bytes_to_copy);
        num->_mpfr_exp = exponent;
        num->_mpfr_sign = sign ? mpfr_sign_t(-1) : mpfr_sign_t(1);  // 1 if negative, 0 otherwise
    }
#endif  // End interoperability with GMP

    inline std::string ToString() const {
        std::stringstream ss;
        ss << ((Sign() < 0) ? "-" : "+") << std::hex;
        constexpr auto i_end = (kMantissaBytes + sizeof(Limb) + 1) / sizeof(Limb);
        for (size_t i = 0; i < i_end; ++i) {
            if (i < i_end - 1) {
                ss << std::setfill('0') << std::setw(2 * sizeof(Limb)) << (*this)[i] << "|";
            } else {
                constexpr auto kLimbModulo = kMantissaBytes % sizeof(Limb);
                ss << std::setfill('0') << std::setw(kLimbModulo > 0 ? (2 * kLimbModulo) : (2 * sizeof(Limb)))
                   << (*this)[i];
            }
        }
        ss << "e" << std::dec << exponent;
        return ss.str();
    }

    inline bool operator==(PackedFloat const &rhs) const {
        if ((sign == 0) != (rhs.sign == 0)) {
            return false;
        }
        if (exponent != rhs.exponent) {
            return false;
        }
        return std::memcmp(mantissa, rhs.mantissa, kMantissaBytes) == 0;
    }

    inline bool operator!=(PackedFloat const &rhs) const {
        return !(*this == rhs);
    }

    static PackedFloat Zero() {
        PackedFloat x;
        x.exponent = 0;
        x.sign = 0;
        std::memset(x.mantissa, 0, kMantissaBytes);
        return x;
    }

    // Fields are left public
    Mantissa mantissa;
    Exponent exponent;
    Sign sign;
};
#pragma pack(pop)
static_assert(sizeof(PackedFloat) == kBytes, "Numbers must be tightly packed.");

inline std::ostream &operator<<(std::ostream &os, PackedFloat const &val) {
    os << val.ToString();
    return os;
}

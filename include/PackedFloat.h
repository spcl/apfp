#pragma once

#include <gmp.h>

#include <algorithm>  // std::min
#include <cstdint>    // uint8_t
#include <cstring>    // std::memcpy
#include <iomanip>    // std::setfill, std::setw
#include <iostream>
#include <sstream>

#include "Config.h"

using Limb = mp_limb_t;
using Exponent = mp_exp_t;
/// Sign convention: 0 is positive, anything else is negative
using Sign = int;  // Word-aligned because this probably makes copies faster on the CPU
constexpr int kMantissaBytes = kBytes - sizeof(Exponent) - sizeof(Sign);
constexpr int kMantissaBits = 8 * kMantissaBytes;
using Mantissa = uint8_t[kMantissaBytes];

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

#ifndef HLSLIB_SYNTHESIS  // Interoperability with GMP, but only on the host side
    inline PackedFloat(mpf_t num) {
        const int num_limbs = std::abs(num->_mp_size);
        const auto bytes_to_copy = std::min(size_t(kMantissaBytes), sizeof(Limb) * num_limbs);
        std::memcpy(mantissa, num->_mp_d, bytes_to_copy);
        std::memset(mantissa + sizeof(Limb) * num_limbs, 0x0, kMantissaBytes - bytes_to_copy);
        exponent = num->_mp_exp - num_limbs + 1;
        sign = num->_mp_size < 0;  // 1 if negative, 0 otherwise
    }

    inline void ToGmp(mpf_t num);

    inline PackedFloat &operator=(mpf_t num) {
        *this = PackedFloat(num);
        return *this;
    }

    inline __mpf_struct ToGmp() {
        __mpf_struct num;
        mpf_init2(&num, kBits);
        ToGmp(&num);
        return num;
    }
#endif  // End interoperability with GMP

    inline std::string ToString() const {
        std::stringstream ss;
        ss << ((Sign() < 0) ? "-" : "+") << std::hex;
        constexpr auto i_end = (kMantissaBytes + sizeof(Limb) + 1) / sizeof(Limb);
        for (size_t i = 0; i < i_end; ++i) {
            ss << std::setfill('0') << std::setw(2 * sizeof(Limb)) << (*this)[i];
            if (i < i_end - 1) {
                ss << "|";
            }
        }
        ss << "e" << std::dec << exponent;
        return ss.str();
    }

    inline Limb operator[](const size_t i) const {
        if (i >= kMantissaBytes / sizeof(Limb)) {
            constexpr int kRemainder = kMantissaBytes % sizeof(Limb);
            constexpr int kOffset = kMantissaBytes - kMantissaBytes % sizeof(Limb);
            if (kRemainder == 1) {
                return Limb(mantissa[kOffset]);
            } else if (kRemainder == 2) {
                return Limb(*reinterpret_cast<uint16_t const *>(&mantissa[kOffset]));
            } else if (kRemainder == 4) {
                return Limb(*reinterpret_cast<uint32_t const *>(&mantissa[kOffset]));
            }
            static_assert(kRemainder == 1 || kRemainder == 2 || kRemainder == 4,
                          "Mantissa non-limb aligned tail must have a size that is a power of two.");
        }
        return reinterpret_cast<mp_limb_t const *>(mantissa)[i];
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

inline void PackedFloat::ToGmp(mpf_t num) {
    const auto bytes_to_copy = std::min(mpf_get_prec(num), mp_bitcnt_t(kMantissaBits)) / 8;
    const int limbs_to_copy = (bytes_to_copy + sizeof(Limb) - 1) / sizeof(Limb);
    num->_mp_exp = exponent;
    // Consider all limbs to be active
    num->_mp_size = 0;
    for (int i = 0; i < limbs_to_copy; ++i) {
        const auto limb = (*this)[i];
        if (limb != 0) {
            num->_mp_d[i] = limb;
            num->_mp_size = i + 1;
        }
    }
    num->_mp_exp = exponent + num->_mp_size - 1;
    if (sign) {
        num->_mp_size = -num->_mp_size;
    }
}

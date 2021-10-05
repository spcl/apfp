#pragma once

#include <gmp.h>

#include <algorithm>  // std::min
#include <cstdint>    // uint8_t
#include <cstring>    // std::memcpy
#include <iomanip>    // std::setfill, std::setw
#include <sstream>

#include "Config.h"

using Limb = mp_limb_t;
using Exponent = mp_exp_t;
/// Sign convention: 0 is positive, anything else is negative
using Sign = int;  // Word-aligned because this probably makes copies faster on the CPU
constexpr int kMantissaBytes = kBytes - sizeof(Exponent) - sizeof(Sign);
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

#ifndef HLSLIB_SYNTHESIS  // Interoperability with GMP, but only on the host side
    inline PackedFloat(mpf_t num) {
        const auto bytes_to_copy = std::min(sizeof(Limb) * std::abs(num->_mp_size), size_t(kBytes));
        // Copy only the limbs actually used by GMP, as the remainder is uninitialized
        std::memcpy(mantissa, num->_mp_d, bytes_to_copy);
        // Zero out unused bytes
        std::memset(mantissa + bytes_to_copy, 0x0, kBytes - bytes_to_copy);
        exponent = num->_mp_exp;
        sign = num->_mp_size < 0;  // 1 if negative, 0 otherwise
    }

    inline void ToGmp(mpf_t num) {
        // Round the supported precision up to the numb
        const auto bytes_to_copy = std::min(8 * mpf_get_prec(num), mp_bitcnt_t(kBytes));
        const int limbs_to_copy = (bytes_to_copy + sizeof(Limb) - 1) / sizeof(Limb);
        num->_mp_exp = exponent;
        // Consider all limbs to be active
        num->_mp_size = (sign != 0) ? -limbs_to_copy : limbs_to_copy;
        std::memcpy(num->_mp_d, mantissa, bytes_to_copy);
    }

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
        constexpr auto kNumLimbs = kMantissaBytes / sizeof(Limb);
        for (size_t i = 0; i < kNumLimbs; ++i) {
            ss << std::setfill('0') << std::setw(2 * sizeof(Limb)) << reinterpret_cast<Limb const *>(mantissa)[i]
               << "|";
        }
        int tail = 0;
        for (size_t i = 0; i < kBytes % sizeof(Limb); ++i) {
            tail += mantissa[kBytes - (kBytes % sizeof(Limb)) + i] << i;
        }
        ss << tail << "e" << std::dec << exponent;
        return ss.str();
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

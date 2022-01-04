#pragma once

#include <algorithm>  // std::min
#include <cassert>
#include <cstdint>  // uint8_t
#include <cstring>  // std::memcpy
#include <iomanip>  // std::setfill, std::setw
#include <iostream>
#include <sstream>

#include "Config.h"
#include "Types.h"

using MantissaFlat = ap_uint<kMantissaBits>;

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

    inline PackedFloat(Sign const &sign, Exponent const &exponent, void const *const mantissa) {
#pragma HLS INLINE
        SetSign(sign);
        SetExponent(exponent);
        SetMantissa(mantissa);
    }

    inline PackedFloat(const DramLine flits[kLinesPerNumber]) {
#pragma HLS INLINE
        for (int i = 0; i < kLinesPerNumber; ++i) {
#pragma HLS UNROLL
            SetFlit(i, flits[i]);
        }
    }

    MantissaFlat GetMantissa() const {
#pragma HLS INLINE
        return data_.range(kMantissaBits - 1, 0);
    }

    void SetMantissa(MantissaFlat const &mantissa) {
#pragma HLS INLINE
        data_.range(kMantissaBits - 1, 0) = mantissa;
    }

    void SetMantissa(void const *const mantissa) {
#pragma HLS INLINE
        std::memcpy(&data_, mantissa, kMantissaBytes);
    }

    Exponent GetExponent() const {
#pragma HLS INLINE
        return data_.range(kBits - 8 * sizeof(Sign) - 1, kBits - 8 * sizeof(Sign) - 8 * sizeof(Exponent));
    }

    void SetExponent(Exponent const &exponent) {
#pragma HLS INLINE
        data_.range(kBits - 8 * sizeof(Sign) - 1, kBits - 8 * sizeof(Sign) - 8 * sizeof(Exponent)) = exponent;
    }

    Sign GetSign() const {
#pragma HLS INLINE
        return data_.range(kBits - 1, kBits - 8 * sizeof(Sign));
    }

    void SetSign(Sign const &sign) {
#pragma HLS INLINE
        data_.range(kBits - 1, kBits - 8 * sizeof(Sign)) = sign;
    }

    DramLine GetFlit(const size_t i) const {
#pragma HLS INLINE
        return data_.range((i + 1) * 512 - 1, i * 512);
    }

    void SetFlit(const size_t i, DramLine const &flit) {
#pragma HLS INLINE
        data_.range((i + 1) * 512 - 1, i * 512) = flit;
    }

    void operator>>(DramLine flits[kLinesPerNumber]) const {
#pragma HLS INLINE
        for (int i = 0; i < kLinesPerNumber; ++i) {
#pragma HLS UNROLL
            flits[i] = GetFlit(i);
        }
    }

    static PackedFloat Zero() {
#pragma HLS INLINE
        PackedFloat x;
        x.data_ = 0;
        return x;
    }

    Limb GetLimb(const size_t i) const {
        return data_.range((i + 1) * 8 * sizeof(Limb) - 1, i * 8 * sizeof(Limb));
    }

#ifndef HLSLIB_SYNTHESIS  // Interoperability with GMP/MPFR, but only on the host side
    inline PackedFloat(mpf_srcptr num) {
        // Copy the most significant bytes, padding zeros if necessary
        const auto num_limbs = std::min(size_t(std::abs(num->_mp_size)),
                                        (mpf_get_prec(num) + 8 * sizeof(mp_limb_t) - 1) / (8 * sizeof(mp_limb_t)));
        const int gmp_bytes = num_limbs * sizeof(mp_limb_t);
        const size_t bytes_to_copy = std::min(gmp_bytes, kMantissaBytes);
        const size_t copy_from = gmp_bytes - bytes_to_copy;
        std::memset(reinterpret_cast<char *>(&data_) + bytes_to_copy, 0x0, kMantissaBytes - bytes_to_copy);
        std::memcpy(reinterpret_cast<char *>(&data_), reinterpret_cast<uint8_t const *>(num->_mp_d) + copy_from,
                    bytes_to_copy);
        SetExponent(num->_mp_exp - num_limbs + 1);
        SetSign(num->_mp_size < 0);  // 1 if negative, 0 otherwise
    }

    inline PackedFloat(const mpfr_srcptr num) {
        // Copy the most significant bytes, padding zeros if necessary
        const auto mpfr_limbs = (mpfr_get_prec(num) + 8 * sizeof(mp_limb_t) - 1) / (8 * sizeof(mp_limb_t));
        const size_t mpfr_bytes = mpfr_limbs * sizeof(mp_limb_t);
        const size_t bytes_to_copy = std::min(mpfr_bytes, size_t(kMantissaBytes));
        const size_t copy_from = mpfr_bytes - bytes_to_copy;
        const size_t copy_to = kMantissaBytes - bytes_to_copy;
        std::memset(reinterpret_cast<char *>(&data_), 0x0, copy_to);
        std::memcpy(reinterpret_cast<char *>(&data_) + copy_to,
                    reinterpret_cast<uint8_t const *>(num->_mpfr_d) + copy_from, bytes_to_copy);
        SetExponent(num->_mpfr_exp);
        SetSign(num->_mpfr_sign < 0);  // 1 if negative, 0 otherwise
    }

    inline PackedFloat &operator=(mpf_srcptr num) {
        *this = PackedFloat(num);
        return *this;
    }

    inline void ToGmp(mpf_ptr num) {
        const size_t gmp_limbs = (mpf_get_prec(num) + 8 * sizeof(mp_limb_t) - 1) / (8 * sizeof(mp_limb_t));
        constexpr size_t kNumLimbs = kMantissaBytes / sizeof(Limb);
        // GMP does not allow graceful rounding, so we cannot handle having insufficient bits in the target GMP number
        assert(gmp_limbs >= kNumLimbs);
        num->_mp_size = 0;
        for (size_t i = 0; i < kNumLimbs; ++i) {
            const auto limb = GetLimb(i);
            num->_mp_d[i] = limb;
            if (limb > 0) {
                num->_mp_size = i + 1;
            }
        }
        num->_mp_exp = GetExponent() + num->_mp_size - 1;
        if (GetSign()) {
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
        std::memcpy(reinterpret_cast<char *>(num->_mpfr_d) + copy_to,
                    reinterpret_cast<char const *>(&data_) + copy_from, bytes_to_copy);
        num->_mpfr_exp = GetExponent();
        num->_mpfr_sign = GetSign() ? mpfr_sign_t(-1) : mpfr_sign_t(1);  // 1 if negative, 0 otherwise
    }
#endif  // End interoperability with GMP

    inline std::string ToString() const {
        std::stringstream ss;
        ss << ((Sign() < 0) ? "-" : "+") << std::hex;
        constexpr auto i_end = kMantissaBytes / sizeof(Limb);
        static_assert(kMantissaBytes % sizeof(Limb) == 0);
        for (size_t i = 0; i < i_end; ++i) {
            ss << std::setfill('0') << std::setw(2 * sizeof(Limb)) << GetLimb(i) << "|";
        }
        ss << "e" << std::dec << GetExponent();
        return ss.str();
    }

    inline bool operator==(PackedFloat const &rhs) const {
        if ((GetSign() == 0) != (rhs.GetSign() == 0)) {
            return false;
        }
        if (GetExponent() != rhs.GetExponent()) {
            return false;
        }
        return std::memcmp(&data_, &rhs.data_, kMantissaBytes) == 0;
    }

    inline bool operator!=(PackedFloat const &rhs) const {
        return !(*this == rhs);
    }

   private:
    ap_uint<kBits> data_;
};
#pragma pack(pop)
static_assert(sizeof(PackedFloat) == kBytes, "Numbers must be tightly packed.");

inline std::ostream &operator<<(std::ostream &os, PackedFloat const &val) {
    os << val.ToString();
    return os;
}

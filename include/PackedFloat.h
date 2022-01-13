#pragma once

#include <algorithm>  // std::min
#include <cassert>
#include <cstring>  // std::memcpy
#include <iomanip>  // std::setfill, std::setw
#include <sstream>

#include "Config.h"
#include "DeviceTypes.h"

using MantissaFlat = ap_uint<kMantissaBits>;

#pragma pack(push, 1)
struct PackedSignExponent {
    Exponent exponent : 8 * sizeof(Exponent) - 1;
    bool sign : 1;
};
static_assert(sizeof(PackedSignExponent) == sizeof(Exponent), "Sign must be tightly packed into exponent.");

/// Full floating point number densely packed to fit into a 512-bit DRAM line.
class PackedFloat {
   public:
    inline PackedFloat() {
        // Leave stuff uninitialized by default
#pragma HLS INLINE
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
        const uint64_t sign_exponent = data_.range(kBits - 1, kBits - 8 * sizeof(Exponent));
        return reinterpret_cast<PackedSignExponent const *>(&sign_exponent)->exponent;
    }

    void SetExponent(Exponent const &exponent) {
#pragma HLS INLINE
        PackedSignExponent sign_exponent;
        sign_exponent.exponent = exponent;
        data_.range(kBits - 2, kBits - 8 * sizeof(Exponent)) = *reinterpret_cast<Exponent const *>(&sign_exponent);
    }

    Sign GetSign() const {
#pragma HLS INLINE
        return data_.get_bit(kBits - 1);
    }

    bool GetSignBit() const {
#pragma HLS INLINE
        return data_.get_bit(kBits - 1);
    }

    void SetSign(Sign const &sign) {
#pragma HLS INLINE
        data_.set_bit(kBits - 1, sign != 0);
    }

    void SetSign(bool sign) {
#pragma HLS INLINE
        data_.set_bit(kBits - 1, sign);
    }

    DramLine GetFlit(const size_t i) const {
#pragma HLS INLINE
        return data_.range((i + 1) * 512 - 1, i * 512);
    }

    void SetFlit(const size_t i, DramLine const &flit) {
#pragma HLS INLINE
        data_.range((i + 1) * 512 - 1, i * 512) = flit;
    }

    void UnpackFlits(DramLine flits[kLinesPerNumber]) const {
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

    inline bool IsZero() const {
        return GetMantissa() == 0;
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
        // Should we just assume nan/inf can't appear?
        if (mpfr_regular_p(num)) {
            // Copy the most significant bytes, padding zeros if necessary
            const auto mpfr_limbs = (mpfr_get_prec(num) + 8 * sizeof(mp_limb_t) - 1) / (8 * sizeof(mp_limb_t));
            const size_t mpfr_bytes = mpfr_limbs * sizeof(mp_limb_t);
            const size_t bytes_to_copy = std::min(mpfr_bytes, size_t(kMantissaBytes));
            const size_t copy_from = mpfr_bytes - bytes_to_copy;
            const size_t copy_to = kMantissaBytes - bytes_to_copy;
            // Doesn't this copy to the LSB of the mantissa?
            std::memset(reinterpret_cast<char *>(&data_), 0x0, copy_to);
            std::memcpy(reinterpret_cast<char *>(&data_) + copy_to,
                        reinterpret_cast<uint8_t const *>(num->_mpfr_d) + copy_from, bytes_to_copy);
            // Section 5.16 of the MPFR manual suggests the exponent might take on special values
            SetExponent(mpfr_get_exp(num));
            SetSign(mpfr_signbit(num) ? 1 : 0);  // 1 if negative, 0 otherwise
        } else {
            *this = PackedFloat::Zero();
        }
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
        // Initialize to 1
        // Otherwise set_exp explodes when it sees num has the special exponent
        // [Section 5.16 of the MPFR manual]
        mpfr_set_ui(num, 1, kRoundingMode);
        // Scan mantissa to check if it's zero since MPFR has a special zero value
        if (IsZero()) {
            mpfr_set_ui(num, 0, kRoundingMode);
        } else {
            // Copy the most significant bytes, padding zeros if necessary
            const auto mpfr_limbs = (mpfr_get_prec(num) + 8 * sizeof(mp_limb_t) - 1) / (8 * sizeof(mp_limb_t));
            const size_t mpfr_bytes = mpfr_limbs * sizeof(mp_limb_t);
            const size_t bytes_to_copy = std::min(mpfr_bytes, size_t(kMantissaBytes));
            const auto copy_to = mpfr_bytes - bytes_to_copy;
            const auto copy_from = kMantissaBytes - bytes_to_copy;
            std::memset(num->_mpfr_d, 0x0, copy_to);
            std::memcpy(reinterpret_cast<char *>(num->_mpfr_d) + copy_to,
                        reinterpret_cast<char const *>(&data_) + copy_from, bytes_to_copy);
            // Returns nonzero is exponent is not in range
            if (mpfr_set_exp(num, GetExponent())) {
                // The only way this can happen is if we hit the magic exponent for NaN/Zero/Inf
                // So flush to zero for now
                mpfr_set_ui(num, 0, kRoundingMode);
            }
            // 1 if negative, 0 otherwise
            mpfr_setsign(num, num, GetSign(), kRoundingMode);
        }
    }
#endif  // End interoperability with GMP

    inline std::string ToString() const {
        std::stringstream ss;
        ss << ((GetSign() != 0) ? "-" : "+") << std::hex;
        constexpr auto i_end = kMantissaBytes / sizeof(Limb);
        static_assert(kMantissaBytes % sizeof(Limb) == 0, "Mantissa size must be a multiple of the limb size.");
        for (size_t i = 0; i < i_end; ++i) {
            ss << std::setfill('0') << std::setw(2 * sizeof(Limb)) << GetLimb(i) << "|";
        }
        ss << "e" << std::dec << GetExponent();
        return ss.str();
    }

    inline bool operator==(PackedFloat const &rhs) const {
        // This passes -0 == 0 but right now we blow away the sign bit in the conversions of singular values anyway
        if (IsZero() && rhs.IsZero()) {
            return true;
        }
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

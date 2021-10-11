#include "ArithmeticOperations.h"

#include <ap_int.h>
#include <hlslib/xilinx/Utility.h>  // hlslib::CeilDivide, hlslib::ConstLog2

#include "Karatsuba.h"

template <int bits>
inline bool IsLastBitSet(ap_uint<bits> const &num) {
    return num(bits - 1, bits - 1) == 1;
}

PackedFloat Multiply(PackedFloat const &a, PackedFloat const &b) {
    // Pad mantissas to avoid passing awkward sizes to Karatsuba
    const ap_uint<kBits> a_mantissa_padded(*reinterpret_cast<ap_uint<kMantissaBits> const *>(a.mantissa));
    const ap_uint<kBits> b_mantissa_padded(*reinterpret_cast<ap_uint<kMantissaBits> const *>(b.mantissa));
    // Meat of the computation. Only keep the top bits of the computation and throw away the rest
#ifdef APFP_MPFR_SEMANTICS
    const ap_uint<kMantissaBits + 1> _m_mantissa =
        Karatsuba(a_mantissa_padded, b_mantissa_padded) >> (kMantissaBits - 1);
    // We need to shift the mantissa forward if the most significant bit is not set
    const bool should_be_shifted = !IsLastBitSet(_m_mantissa);
    ap_uint<kMantissaBits + 1> m_mantissa = should_be_shifted ? _m_mantissa : (_m_mantissa >> 1);
    // Add up exponents. If the most significant bit was 1, we're done. Otherwise subtract 1 due to
    // the shift.
    const Exponent m_exponent = a.exponent + b.exponent - should_be_shifted;
#else  // Use GMP semantics
    constexpr auto kLimbBits = 8 * sizeof(mp_limb_t);
    ap_uint<(2 * kMantissaBits)> _m_mantissa = Karatsuba(a_mantissa_padded, b_mantissa_padded);
    const bool limb_zero = _m_mantissa.range(kMantissaBits + kLimbBits - 1, kMantissaBits) == 0;
    ap_uint<kMantissaBits + kLimbBits> m_mantissa = _m_mantissa;  // Truncate
    const Exponent m_exponent = a.exponent + b.exponent - limb_zero;
#endif
    // The sign is just the XOR of the existing signs
    const bool m_sign = a.sign != b.sign;
    return {m_sign, m_exponent, &m_mantissa};
}

PackedFloat Add(PackedFloat const &a, PackedFloat const &b) {
#pragma HLS INLINE

    // Figure out how much we need to shift by
    ap_uint<kMantissaBits> a_mantissa(*reinterpret_cast<ap_uint<kMantissaBits> const *>(a.mantissa));
    ap_uint<kMantissaBits> b_mantissa(*reinterpret_cast<ap_uint<kMantissaBits> const *>(b.mantissa));
    const bool a_is_larger = a.exponent > b.exponent;
    const bool b_is_larger = !a_is_larger;
    const bool a_is_zero = a_mantissa == 0;
    const bool b_is_zero = b_mantissa == 0;
    const Exponent res_exponent = ((a_is_larger && !a_is_zero) || b_is_zero) ? a.exponent : b.exponent;
    const Exponent shift_c = (b_is_larger && !b_is_zero) ? (b.exponent - a.exponent) : 0;
    const Exponent shift_m = (a_is_larger && !a_is_zero) ? (a.exponent - b.exponent) : 0;
    // Optionally shift by 1, 2, 4, 8, 16... log2(B), such that all bits have eventually traveled to
    // their designated position.
    const int kNumStages = hlslib::ConstLog2(kBits);
    for (int i = 0; i < kNumStages; ++i) {
#pragma HLS UNROLL
        a_mantissa = ((shift_c & (1 << i)) == 0) ? a_mantissa : (a_mantissa >> (1 << i));
        b_mantissa = ((shift_m & (1 << i)) == 0) ? b_mantissa : (b_mantissa >> (1 << i));
    }
    // Finally zero out the mantissas if they are shifted by more than the precision
    a_mantissa = shift_c >= kBits ? decltype(a_mantissa)(0) : a_mantissa;
    b_mantissa = shift_m >= kBits ? decltype(b_mantissa)(0) : b_mantissa;
    // Now we can add up the aligned mantissas
    const ap_uint<kMantissaBits + 1> _res_mantissa = a_mantissa + b_mantissa;
    // If the addition overflowed, we need to shift and increment the exponent
    const bool addition_overflowed = IsLastBitSet(_res_mantissa);
    ap_uint<kMantissaBits> res_mantissa = addition_overflowed ? (_res_mantissa >> 1) : _res_mantissa;
    PackedFloat result;
    std::memcpy(result.mantissa, &res_mantissa, kMantissaBytes);
    result.exponent = res_exponent + addition_overflowed;
    // Sign will be the same as whatever is the largest number
    result.sign = a_is_larger ? a.sign : b.sign;

    return result;
}

PackedFloat MultiplyAccumulate(PackedFloat const &a, PackedFloat const &b, PackedFloat const &c) {
    return Add(c, Multiply(a, b));
}

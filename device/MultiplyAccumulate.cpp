#include "MultiplyAccumulate.h"

#include <ap_int.h>
#include <hlslib/xilinx/Utility.h>  // hlslib::CeilDivide, hlslib::ConstLog2

#include "Karatsuba.h"

template <int bits>
inline bool IsLastBitSet(ap_uint<bits> const &num) {
    return num(bits - 1, bits - 1) == 1;
}

PackedFloat MultiplyAccumulate(PackedFloat const &a, PackedFloat const &b, PackedFloat const &c) {
#pragma HLS INLINE

    ////////////////////////////////////////////////////////////////////////////
    // Multiplication
    ////////////////////////////////////////////////////////////////////////////

    // Pad mantissas to avoid passing awkward sizes to Karatsuba
    const ap_uint<kBits> a_mantissa_padded = *reinterpret_cast<ap_uint<kMantissaBits> const *>(a.mantissa);
    const ap_uint<kBits> b_mantissa_padded = *reinterpret_cast<ap_uint<kMantissaBits> const *>(b.mantissa);
    // Meat of the computation. Only keep the top bits of the computation and throw away the rest
    const ap_uint<kMantissaBits + 1> _m_mantissa =
        Karatsuba(a_mantissa_padded, b_mantissa_padded) >> (kMantissaBits - 1);
    // We need to shift the mantissa forward if the most significant bit is not set
    const bool should_be_shifted = !IsLastBitSet(_m_mantissa);
    ap_uint<kMantissaBits> m_mantissa = should_be_shifted ? _m_mantissa : (_m_mantissa >> 1);
    // The sign is just the XOR of the existing signs
    const bool m_sign = a.sign != b.sign;
    // Add up exponents. If the most significant bit was 1, we're done. Otherwise subtract 1 due to
    // the shift.
    const Exponent m_exponent = a.exponent + b.exponent - should_be_shifted;

    ////////////////////////////////////////////////////////////////////////////
    // Addition
    ////////////////////////////////////////////////////////////////////////////

    // Figure out how much we need to shift by
    ap_uint<kMantissaBits> c_mantissa = *reinterpret_cast<ap_uint<kMantissaBits> const *>(c.mantissa);
    const bool c_is_larger = c.exponent > m_exponent;
    const bool m_is_larger = !c_is_larger;
    const bool c_is_zero = c_mantissa == 0;
    const bool m_is_zero = m_mantissa == 0;
    const Exponent res_exponent = ((c_is_larger && !c_is_zero) || m_is_zero) ? c.exponent : m_exponent;
    const Exponent shift_c = (m_is_larger && !m_is_zero) ? (m_exponent - c.exponent) : 0;
    const Exponent shift_m = (c_is_larger && !c_is_zero) ? (c.exponent - m_exponent) : 0;
    // Optionally shift by 1, 2, 4, 8, 16... log2(B), such that all bits have eventually traveled to
    // their designated position.
    const int kNumStages = hlslib::ConstLog2(kBits);
    for (int i = 0; i < kNumStages; ++i) {
#pragma HLS UNROLL
        c_mantissa = ((shift_c & (1 << i)) == 0) ? c_mantissa : (c_mantissa >> (1 << i));
        m_mantissa = ((shift_m & (1 << i)) == 0) ? m_mantissa : (m_mantissa >> (1 << i));
    }
    // Finally zero out the mantissas if they are shifted by more than the precision
    c_mantissa = shift_c >= kBits ? decltype(c_mantissa)(0) : c_mantissa;
    m_mantissa = shift_m >= kBits ? decltype(m_mantissa)(0) : m_mantissa;
    // Now we can add up the aligned mantissas
    ap_uint<kMantissaBits + 1> _res_mantissa = m_mantissa + m_mantissa;
    // If the addition overflowed, we need to shift and increment the exponent
    const bool addition_overflowed = IsLastBitSet(_res_mantissa);
    ap_uint<kMantissaBits> res_mantissa = addition_overflowed ? (_res_mantissa >> 1) : _res_mantissa;
    PackedFloat result;
    std::memcpy(result.mantissa, &res_mantissa, kMantissaBytes);
    result.exponent = res_exponent + addition_overflowed;
    // Sign will be the same as whatever is the largest number
    result.sign = c_is_larger ? c.sign : m_sign;

    return result;
}

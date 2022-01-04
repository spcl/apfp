#include "ArithmeticOperations.h"

#include <ap_int.h>
#include <hlslib/xilinx/Utility.h>  // hlslib::CeilDivide, hlslib::ConstLog2

#include "Karatsuba.h"

template <int bits>
inline bool IsMostSignificantBitSet(ap_uint<bits> const &num) {
#pragma HLS INLINE
    return num.test(bits - 1);
}

template <int bits>
inline int CountLeadingZeros(ap_uint<bits> const &num) {
#pragma HLS INLINE
    int leading_zeros = 0;
    for(int i = 0; i < bits; ++i) {
        if(num.test(bits - (i + 1))) {
            break;
        }
        leading_zeros = i;
    }
    return leading_zeros;
}

PackedFloat Multiply(PackedFloat const &a, PackedFloat const &b) {
#pragma HLS INLINE
    // Pad mantissas to avoid passing awkward sizes to Karatsuba
    const ap_uint<kBits> a_mantissa_padded(a.GetMantissa());
    const ap_uint<kBits> b_mantissa_padded(b.GetMantissa());
#ifdef APFP_GMP_SEMANTICS  // Use GMP semantics
    constexpr auto kLimbBits = 8 * sizeof(mp_limb_t);
    // Meat of the computation. Only keep the top bits of the computation and throw away the rest
    const ap_uint<(2 * kMantissaBits)> _m_mantissa = Karatsuba(a_mantissa_padded, b_mantissa_padded);
    const bool limb_zero = _m_mantissa.range(kMantissaBits + kLimbBits - 1, kMantissaBits) == 0;
    ap_uint<kMantissaBits + kLimbBits> m_mantissa = _m_mantissa;  // Truncate
    const Exponent m_exponent = a.GetExponent() + b.GetExponent() - limb_zero;
#else  // Otherwise use MPFR semantics
    const ap_uint<kMantissaBits + 1> _m_mantissa =
        Karatsuba(a_mantissa_padded, b_mantissa_padded) >> (kMantissaBits - 1);
    // We need to shift the mantissa forward if the most significant bit is not set
    const bool should_be_shifted = !IsMostSignificantBitSet(_m_mantissa);
    ap_uint<kMantissaBits + 1> m_mantissa = should_be_shifted ? _m_mantissa : (_m_mantissa >> 1);
    // Add up exponents. If the most significant bit was 1, we're done. Otherwise subtract 1 due to
    // the shift.
    const Exponent m_exponent = a.GetExponent() + b.GetExponent() - (should_be_shifted ? 1 : 0);
#endif
    // The sign is just the XOR of the existing signs
    PackedFloat result;
    result.SetMantissa(m_mantissa);
    result.SetExponent(m_exponent);
    result.SetSign(a.GetSign() != b.GetSign());
    return result;
}

// Does this correctly output the result if a and b are different signs?
// The mantissa of the result should depend on the sign bits of a and b

PackedFloat Add(PackedFloat const &a, PackedFloat const &b) {
#pragma HLS INLINE

    // Figure out how much we need to shift by
    ap_uint<kMantissaBits> a_mantissa(a.GetMantissa());
    ap_uint<kMantissaBits> b_mantissa(b.GetMantissa());

#ifndef HLSLIB_SYNTHESIS
    // We better not be getting subnormal inputs
    assert(a.IsZero() || IsMostSignificantBitSet(a_mantissa));
    assert(a.IsZero() || IsMostSignificantBitSet(b_mantissa));
#endif

    const bool exp_are_equal = (a.GetExponent() == b.GetExponent());
    const bool a_exp_is_larger = (a.GetExponent() > b.GetExponent());
    const bool a_mant_is_zero = a_mantissa == 0;
    const bool b_mant_is_zero = b_mantissa == 0;
    const bool a_is_larger = (!a_mant_is_zero) && (a_exp_is_larger || (exp_are_equal && a.GetMantissa() > b.GetMantissa()));
    const bool subtraction = a.GetSign() != b.GetSign();
    Exponent res_exponent = ((a_exp_is_larger && !a_mant_is_zero) || b_mant_is_zero) ? a.GetExponent() : b.GetExponent();
    const Exponent shift_c = (!a_exp_is_larger && !b_mant_is_zero) ? (b.GetExponent() - a.GetExponent()) : 0;
    const Exponent shift_m = (a_exp_is_larger && !a_mant_is_zero) ? (a.GetExponent() - b.GetExponent()) : 0;

#ifndef HLSLIB_SYNTHESIS
    // Turns out Xilinx allows signed shifts!
    assert(shift_m >= 0);
    assert(shift_c >= 0);
#endif

    // Optionally shift right by 1, 2, 4, 8, 16... log2(B), such that all bits have eventually traveled to
    // their designated position.
    // This is unsigned
    a_mantissa = a_mantissa >> shift_c;
    b_mantissa = b_mantissa >> shift_m;
//     const int kNumStages = hlslib::ConstLog2(kMantissaBits);
// ShiftStages:
//     for (int i = 0; i < kNumStages; ++i) {
// #pragma HLS UNROLL
//         a_mantissa = ((shift_c & (1 << i)) == 0) ? a_mantissa : (a_mantissa >> (1 << i));
//         b_mantissa = ((shift_m & (1 << i)) == 0) ? b_mantissa : (b_mantissa >> (1 << i));
//     }
    // Finally zero out the mantissas if they are shifted by more than the precision
    // Is this necessary?
    a_mantissa = shift_c >= kMantissaBits ? decltype(a_mantissa)(0) : a_mantissa;
    b_mantissa = shift_m >= kMantissaBits ? decltype(b_mantissa)(0) : b_mantissa;
    // Now we can add up the aligned mantissas
    // ==== Add/Sub mantissas and overflow check ====
    const ap_uint<kMantissaBits + 1> ab_sum = a_mantissa + b_mantissa;
#pragma HLS BIND_OP variable = ab_sum op = add impl = fabric latency = 4
    ap_uint<kMantissaBits + 1> larger_mantissa = a_is_larger ? a_mantissa : b_mantissa;
    ap_uint<kMantissaBits + 1> smaller_mantissa = a_is_larger ? b_mantissa : a_mantissa;
    // This returns an ap_int but the answer is always positive so the MSB is never set
    // Xilinx manual states signed <-> unsigned ignores the sign and converts bit for bit
    const ap_uint<kMantissaBits + 1> ab_abs_diff = static_cast<ap_uint<kMantissaBits>>(larger_mantissa - smaller_mantissa);
#pragma HLS BIND_OP variable = ab_abs_diff op = sub impl = fabric latency = 4

    const ap_uint<kMantissaBits + 1> _res_mantissa = subtraction ? ab_abs_diff : ab_sum;
    
    // If the addition overflowed, we need to shift and increment the exponent
    const bool addition_overflowed = IsMostSignificantBitSet(_res_mantissa);
    ap_uint<kMantissaBits> res_mantissa = addition_overflowed ? (_res_mantissa >> 1) : _res_mantissa;
    res_exponent = res_exponent + (addition_overflowed ? 1 : 0);

    // ==== Renormalize / Underflow ====
    // Normalize the mantissa
    bool res_nonzero = res_mantissa != 0;
    const Exponent leading_zeros = res_nonzero ? CountLeadingZeros(res_mantissa) : kMantissaBits;

    // Left shift by the number of leading zeros
    res_mantissa = res_nonzero ? (res_mantissa << leading_zeros) : decltype(res_mantissa)(0);

    // We need to watch for underflow here
    const bool underflow = res_exponent < std::numeric_limits<Exponent>::min() + leading_zeros;
    res_exponent = res_exponent - leading_zeros;

    // Flush to zero if we underflow
    if (underflow || !res_nonzero) { 
        res_mantissa = 0;
        res_exponent = 0;
    }

#ifndef HLSLIB_SYNTHESIS
    // We cannot have an unnormalized mantissa by this point
    assert(underflow || !res_nonzero || IsMostSignificantBitSet(res_mantissa));
#endif

    PackedFloat result;
    result.SetMantissa(res_mantissa);
    result.SetExponent(res_exponent);
    // Sign will be the same as whatever is the largest number
    result.SetSign(a_is_larger ? a.GetSign() : b.GetSign());

    return result;
}

PackedFloat MultiplyAccumulate(PackedFloat const &a, PackedFloat const &b, PackedFloat const &c) {
#pragma HLS INLINE
    return Add(c, Multiply(a, b));
}

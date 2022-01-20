#include "ArithmeticOperations.h"

#include <ap_int.h>
#include <hlslib/xilinx/Utility.h>  // hlslib::CeilDivide, hlslib::ConstLog2

#include "Karatsuba.h"
#include "PipelinedAdd.h"

template <int bits>
bool IsMostSignificantBitSet(ap_uint<bits> const &num) {
#pragma HLS INLINE
    return num.test(bits - 1);
}

template <int bits>
ap_uint<bits> DynamicRightShift(ap_uint<bits> num, ap_uint<hlslib::ConstLog2(bits)> const &shift_by) {
#pragma HLS INLINE
    const auto kNumStages = hlslib::ConstLog2(bits);
    for (int i = 0; i < kNumStages; ++i) {
#pragma HLS UNROLL
        num = shift_by.test(i) ? (num >> (1 << i)) : num;
    }
    return num;
}

template <int bits>
ap_uint<bits> DynamicLeftShift(ap_uint<bits> num, ap_uint<hlslib::ConstLog2(bits)> const &shift_by) {
#pragma HLS INLINE
    const auto kNumStages = hlslib::ConstLog2(bits);
    for (int i = 0; i < kNumStages; ++i) {
#pragma HLS UNROLL
        num = shift_by.test(i) ? (num << (1 << i)) : num;
    }
    return num;
}

PackedFloat Multiply(PackedFloat const &a, PackedFloat const &b) {
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
    result.SetSign(a.GetSignBit() != b.GetSignBit());
    return result;
}

// Does this correctly output the result if a and b are different signs?
// The mantissa of the result should depend on the sign bits of a and b
PackedFloat Add(PackedFloat const &a_in, PackedFloat const &b_in) {

    // Retrieve once and for all to make sure there's no overhead from unpacking them
    const bool _a_sign = a_in.GetSign();
    const bool _b_sign = b_in.GetSign();
    const Exponent _a_exponent = a_in.GetExponent();
    const Exponent _b_exponent = b_in.GetExponent();
    const MantissaFlat _a_mantissa = a_in.GetMantissa();
    const MantissaFlat _b_mantissa = b_in.GetMantissa();

    const bool exp_are_equal = (_a_exponent == _b_exponent);
    const bool a_in_exp_strictly_larger = (_a_exponent > _b_exponent);
    const bool a_in_mant_is_zero = _a_mantissa == 0;
    const bool b_in_mant_is_zero = _b_mantissa == 0;
    const bool a_in_mantissa_larger = _a_mantissa >= _b_mantissa;

    // Plain comparison of exponent and mantissa
    const bool a_larger_if_both_nonzero = a_in_exp_strictly_larger || (exp_are_equal && a_in_mantissa_larger);
    // Handle a being zero
    const bool a_larger_if_b_nonzero = !a_in_mant_is_zero && a_larger_if_both_nonzero;
    // Handle b being zero
    const bool a_is_larger = b_in_mant_is_zero || a_larger_if_b_nonzero;

    // We always have a >= b to simplify the code
    // a is zero iff b is zero
    const Exponent a_exponent = a_is_larger ? _a_exponent : _b_exponent;
    const Exponent b_exponent = a_is_larger ? _b_exponent : _a_exponent;
    const MantissaFlat a_mantissa = a_is_larger ? _a_mantissa : _b_mantissa;
    const MantissaFlat b_mantissa = a_is_larger ? _b_mantissa : _a_mantissa;
    const bool a_sign = a_is_larger ? _a_sign : _b_sign;
    const bool b_sign = a_is_larger ? _b_sign : _a_sign;

    const bool a_is_zero = a_is_larger ? a_in_mant_is_zero : b_in_mant_is_zero;

#ifndef HLSLIB_SYNTHESIS
    // We better not be getting subnormal inputs
    assert(a_in.IsZero() || IsMostSignificantBitSet(a_in.GetMantissa()));
    assert(b_in.IsZero() || IsMostSignificantBitSet(b_in.GetMantissa()));
    // a is zero => b is zero
    assert(!a_is_zero || (a_is_zero && a_in_mant_is_zero && b_in_mant_is_zero));
#endif

    const bool subtraction = a_sign != b_sign;
    Exponent res_exponent = a_exponent;
    using Shift = ap_uint<8 * sizeof(Exponent) - 2>;
    const Shift shift_m = a_exponent - b_exponent;

    // Figure out how much we need to shift by
    // Xilinx permits signed shifts
    // We want to keep an extra bit of precision (LSB) to properly round the output
    // We also want an extra bit of range (MSB) to track overflow
    // The names in the following code segment have _msb/_lsb suffix if they have the extra msb/lsb respectively
    using MantissaExtended = ap_uint<2 * kMantissaBits + 1>;
    auto a_mantissa_shifted = MantissaExtended(a_mantissa) << kMantissaBits;
    auto b_mantissa_shifted = DynamicRightShift(MantissaExtended(b_mantissa) << kMantissaBits, shift_m);

    // Now we can add up the aligned mantissas
    // ==== Add/Sub mantissas ====
    // We cannot truncate yet because of the renormalization step
    const MantissaExtended ab_sum_lsb_msb = PipelinedAdd<2 * kMantissaBits>(a_mantissa_shifted, b_mantissa_shifted);

    // This returns an ap_int but the answer is always positive so the MSB is never set
    // Xilinx manual states signed <-> unsigned ignores the sign and converts bit for bit
    // Widening assignments and right shifts of ap_int are sign extended so we specify the casting route
    assert(a_mantissa_shifted >= b_mantissa_shifted);
    MantissaExtended ab_diff_lsb_msb = MantissaExtended(PipelinedSub(a_mantissa_shifted, b_mantissa_shifted));
    assert(!IsMostSignificantBitSet(ab_diff_lsb_msb));

    // ==== overflow check ====

    // If the addition overflowed, we need to shift and increment the exponent
    // We could just do the right shift and let the mantissa normalization step fix up the exponent
    const auto _res_mantissa_lsb_msb = subtraction ? ab_diff_lsb_msb : ab_sum_lsb_msb;

    const bool addition_overflowed = IsMostSignificantBitSet(_res_mantissa_lsb_msb);
    // We're still holding onto the extra lsb
    const MantissaExtended res_mantissa_lsb =
        addition_overflowed ? (_res_mantissa_lsb_msb >> 1) : _res_mantissa_lsb_msb;
    res_exponent = res_exponent + (addition_overflowed ? 1 : 0);

    // ==== Renormalize / Underflow ====
    // Normalize the mantissa
    bool res_nonzero = res_mantissa_lsb != 0;
    const ap_uint<hlslib::ConstLog2(kMantissaBits)> leading_zeros =
        MantissaFlat(res_mantissa_lsb >> kMantissaBits).countLeadingZeros();

    // Left shift by the number of leading zeros and truncate the lsb now
    ap_uint<kMantissaBits> res_mantissa = DynamicLeftShift(res_mantissa_lsb, leading_zeros) >> kMantissaBits;

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
    result.SetSign(a_sign);

    return a_is_zero ? PackedFloat::Zero() : result;
}

PackedFloat MultiplyAccumulate(PackedFloat const &a, PackedFloat const &b, PackedFloat const &c) {
#pragma HLS INLINE
    return Add(c, Multiply(a, b));
}

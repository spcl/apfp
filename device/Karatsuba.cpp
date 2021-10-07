#include "Karatsuba.h"

#include <type_traits>  // std::enable_if

template <int bits>
typename std::enable_if<(bits > kMultBaseBits), ap_uint<2 * bits>>::type _Karatsuba(ap_uint<bits> const &a,
                                                                                    ap_uint<bits> const &b) {
#pragma HLS INLINE
    static_assert(bits % 2 == 0, "Number of bits must be even.");
    using Full = ap_uint<bits>;
    using Half = ap_uint<bits / 2>;

    // Decompose input operands into halves for the recursive step
    Half a0 = a.range(bits / 2 - 1, 0);
    Half a1 = a.range(bits - 1, bits / 2);
    Half b0 = b.range(bits / 2 - 1, 0);
    Half b1 = b.range(bits - 1, bits / 2);

    // Recurse on a_0 * b_0 and a_1 * b_1
    Full z0 = _Karatsuba<bits / 2>(a0, b0);
    Full z2 = _Karatsuba<bits / 2>(a1, b1);

    // Compute |a_0 - a_1| and sign(a_0 - a_1)
    bool a0a1_is_neg = a0 < a1;
    Half a0a1 = a0a1_is_neg ? (a1 - a0) : (a0 - a1);
    // Compute |b_1 - b_0| and sign(b_1 - b_0)
    bool b0b1_is_neg = b1 < b0;
    Half b0b1 = b0b1_is_neg ? (b0 - b1) : (b1 - b0);

    // XOR the two signs to get the final sign
    bool a0a1b0b1_is_neg = a0a1_is_neg != b0b1_is_neg;
    // Recurse on |a_0 - a_1| * |b_0 - b_1|
    Full a0a1b0b1 = _Karatsuba<bits / 2>(a0a1, b0b1);
    ap_int<bits + 2> a0a1b0b1_signed = a0a1b0b1_is_neg ? -ap_int<bits + 1>(a0a1b0b1) : ap_int<bits + 2>(a0a1b0b1);
    ap_uint<bits + 2> z1 = ap_uint<bits + 2>(a0a1b0b1_signed) + z0 + z2;

    // Align everything and combine
    ap_uint<(2 * bits)> z0z2 = z0 + (ap_uint<(2 * bits)>(z2) << bits);
    ap_uint<(2 * bits)> z1_aligned = ap_uint<(2 * bits)>(z1) << (bits / 2);
    ap_uint<(2 * bits)> z = z1_aligned + z0z2;

    return z;
}

// Bottom out using SFINAE when the bit width is lower or equal to the specified base number of bits
template <int bits>
typename std::enable_if<(bits <= kMultBaseBits), ap_uint<2 * bits>>::type _Karatsuba(ap_uint<bits> const &a,
                                                                                     ap_uint<bits> const &b) {
#pragma HLS INLINE
    return a * b;
}

ap_uint<2 * kBits> Karatsuba(ap_uint<kBits> const &a, ap_uint<kBits> const &b) {
#pragma HLS INLINE
    return _Karatsuba<kBits>(a, b);
}

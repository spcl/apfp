#include "Karatsuba.h"

#include <type_traits>  // std::enable_if

constexpr int AddLatency(int bits) {
    // 4 is the maximum supported latency of integer adds using the BIND_OP pragma
    return (bits >= 1024) ? 4 : (bits >= 768) ? 3 : (bits >= 512) ? 2 : (bits >= 256) ? 1 : 0;
}

constexpr int kInlineCutoff = 256;


template <int bits>
auto _Karatsuba(ap_uint<bits> const &a, ap_uint<bits> const &b) ->
    typename std::enable_if<(bits==kInlineCutoff)&&(bits > kMultBaseBits), ap_uint<2 * bits>>::type {
#pragma HLS INLINE OFF
    #include "Karatsuba_impl.h"
}

template <int bits>
auto _Karatsuba(ap_uint<bits> const &a, ap_uint<bits> const &b) ->
    typename std::enable_if<(bits!=kInlineCutoff)&&(bits > kMultBaseBits), ap_uint<2 * bits>>::type {
#pragma HLS INLINE RECURSIVE
    #include "Karatsuba_impl.h"
}

// Bottom out using SFINAE when the bit width is lower or equal to the specified base number of bits
template <int bits>
auto _Karatsuba(ap_uint<bits> const &a, ap_uint<bits> const &b) ->
    typename std::enable_if<(bits <= kMultBaseBits), ap_uint<2 * bits>>::type {
#pragma HLS INLINE
    return a * b;
}

ap_uint<2 * kBits> Karatsuba(ap_uint<kBits> const &a, ap_uint<kBits> const &b) {
#pragma HLS INLINE
    return _Karatsuba<kBits>(a, b);
}

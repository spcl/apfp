#pragma once

#include <ap_int.h>
#include <hlslib/xilinx/Utility.h>

#include <limits>

#include "Config.h"

namespace {

#ifdef APFP_USE_PIPELINED_ADD
constexpr int kPipelinedAddBaseBits = kAddBaseBits;
#else
// Bottom out immediately, falling back on the Xilinx implementation regardless of the bit width, but continuing to use
// kAddBaseBits to inject some pipeline stages
constexpr int kPipelinedAddBaseBits = std::numeric_limits<int>::max();
#endif

constexpr int AddLatency(int bits) {
    // 4 is the maximum supported latency of integer adds using the BIND_OP pragma
    return (bits <= kAddBaseBits / 4)
               ? 0
               : (bits <= kAddBaseBits / 2) ? 1 : (bits <= 3 * (kAddBaseBits / 4)) ? 2 : (bits <= kAddBaseBits) ? 3 : 4;
}

template <int total_bits, int num_steps, int step>
struct _PipelinedAddImpl {
    static constexpr int bit_end = (step * total_bits) / num_steps;
    static constexpr int bit_begin = ((step - 1) * total_bits) / num_steps;
    static constexpr int bits = bit_end - bit_begin;

    static bool Apply(ap_uint<total_bits> const &a, ap_uint<total_bits> const &b, ap_uint<total_bits + 1> &result,
                      bool carry_in) {
#pragma HLS INLINE
        const auto carry_step = _PipelinedAddImpl<total_bits, num_steps, step - 1>::Apply(a, b, result, carry_in);
        const ap_uint<bits> a_chunk = a.range(bit_end - 1, bit_begin);
        const ap_uint<bits> b_chunk = b.range(bit_end - 1, bit_begin);
        const ap_uint<bits + 2> result_chunk = PipelinedAdd(a_chunk, b_chunk, carry_step);
        result.range(bit_end - 1, bit_begin) = result_chunk.range(bits - 1, 0);
        return result_chunk.get_bit(bits);
    }
};

template <int total_bits, int num_steps>
struct _PipelinedAddImpl<total_bits, num_steps, 1> {
    static constexpr int bit_end = total_bits / num_steps;
    static constexpr int bits = bit_end;

    static bool Apply(ap_uint<total_bits> const &a, ap_uint<total_bits> const &b, ap_uint<total_bits + 1> &result,
                      bool carry_in) {
#pragma HLS INLINE
        const ap_uint<bits> a_chunk = a.range(bit_end - 1, 0);
        const ap_uint<bits> b_chunk = b.range(bit_end - 1, 0);
        const ap_uint<bits + 1> result_chunk = PipelinedAdd(a_chunk, b_chunk, carry_in);
        result.range(bit_end - 1, 0) = result_chunk.range(bits - 1, 0);
        return result_chunk.get_bit(bits);
    }
};

}  // namespace

template <int bits>
auto PipelinedAdd(ap_uint<bits> const &a, ap_uint<bits> const &b, bool carry_in = false) ->
    typename std::enable_if<(bits > kPipelinedAddBaseBits), ap_uint<bits + 2>>::type {
#pragma HLS INLINE
    ap_uint<bits + 1> result;
    constexpr int num_steps = hlslib::CeilDivide(bits, kPipelinedAddBaseBits);
    const auto carry_out = _PipelinedAddImpl<bits, num_steps, num_steps>::Apply(a, b, result, carry_in);
    result.set_bit(bits, carry_out);
    return result;
}

template <int bits>
auto PipelinedAdd(ap_uint<bits> const &a, ap_uint<bits> const &b, bool carry_in = false) ->
    typename std::enable_if<(bits <= kPipelinedAddBaseBits), ap_uint<bits + 2>>::type {
#pragma HLS INLINE
    const auto result = a + b + ap_uint<1>(carry_in ? 1 : 0);
#pragma HLS BIND_OP variable = result op = add impl = fabric latency = AddLatency(bits)
    return result;
}

template <int bits>
auto PipelinedSub(ap_uint<bits> const &a, ap_uint<bits> const &b) -> ap_uint<bits + 2> {
#pragma HLS INLINE
    return PipelinedAdd<bits>(a, ~b, true);
}

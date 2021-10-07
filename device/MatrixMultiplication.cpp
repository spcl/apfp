#include "MatrixMultiplication.h"

#include <hlslib/xilinx/Simulation.h>
#include <hlslib/xilinx/Stream.h>
#include <hlslib/xilinx/Utility.h>

#include "Karatsuba.h"

void ReadA(DramLine const *const mem, hlslib::Stream<PackedFloat> &to_kernel, const int size_n, const int size_k) {
    DramLine num[kLinesPerNumber];
    for (int n = 0; n < size_n; ++n) {
        for (int k = 0; k < size_k; ++k) {
            for (int i = 0; i < kLinesPerNumber; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
                num[i] = mem[(n * size_k + k) * kLinesPerNumber + i];
                if (i == kLinesPerNumber - 1) {
                    to_kernel.Push(*reinterpret_cast<PackedFloat const *>(num));
                }
            }
        }
    }
}

void ReadB(DramLine const *const mem, hlslib::Stream<PackedFloat> &to_kernel, const int size_k, const int size_m) {
    DramLine num[kLinesPerNumber];
    for (int k = 0; k < size_k; ++k) {
        for (int m = 0; m < size_m; ++m) {
            for (int i = 0; i < kLinesPerNumber; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
                num[i] = mem[(k * size_m + m) * kLinesPerNumber + i];
                if (i == kLinesPerNumber - 1) {
                    to_kernel.Push(*reinterpret_cast<PackedFloat const *>(num));
                }
            }
        }
    }
}

void ReadC(DramLine const *const mem, hlslib::Stream<PackedFloat> &to_kernel, const int size_n, const int size_m) {
    DramLine num[kLinesPerNumber];
    for (int n = 0; n < size_n; ++n) {
        for (int m = 0; m < size_m; ++m) {
            for (int i = 0; i < kLinesPerNumber; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
                num[i] = mem[(n * size_m + m) * kLinesPerNumber + i];
                if (i == kLinesPerNumber - 1) {
                    to_kernel.Push(*reinterpret_cast<PackedFloat const *>(num));
                }
            }
        }
    }
}

void WriteC(hlslib::Stream<PackedFloat> &from_kernel, DramLine *const mem, const int size_n, int const size_m) {
    DramLine num[kLinesPerNumber];
    for (int n = 0; n < size_n; ++n) {
        for (int m = 0; m < size_m; ++m) {
            for (int i = 0; i < kLinesPerNumber; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
                if (i == 0) {
                    *reinterpret_cast<PackedFloat *>(num) = from_kernel.Pop();
                }
                mem[(n * size_m + m) * kLinesPerNumber + i] = num[i];
            }
        }
    }
}

template <int bits>
inline bool IsLastBitSet(ap_uint<bits> const &num) {
    return num(bits - 1, bits - 1) == 1;
}

void Compute(hlslib::Stream<PackedFloat> &a_in, hlslib::Stream<PackedFloat> &b_in, hlslib::Stream<PackedFloat> &c_in,
             hlslib::Stream<PackedFloat> &c_out, int const size_n, int const size_k, int const size_m) {
    PackedFloat a, b, c;
    for (int n = 0; n < size_n; ++n) {
        for (int k = 0; k < size_k; ++k) {
            for (int m = 0; m < size_m; ++m) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
                if (n == 0) {
                    b = b_in.Pop();
                }
                if (k == 0) {
                    c = c_in.Pop();
                }
                if (m == 0) {
                    a = a_in.Pop();
                }

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
                // Add up exponents. If the most significant bit was 1, we're done. Otherwise subtract 1 due to the
                // shift.
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
                if (k == size_k - 1) {
                    c_out.Push(result);
                }
            }
        }
    }
}

void MatrixMultiplication(DramLine const *const a, DramLine const *const b, DramLine const *const c_read,
                          DramLine *const c_write, const int size_n, const int size_m, int const size_k) {
#pragma HLS INTERFACE m_axi offset = slave port = a bundle = a
#pragma HLS INTERFACE m_axi offset = slave port = b bundle = b
// Even though they actually point to the same memory location, we use two separate interfaces for reading and writing
// C, to make sure that the compiler doesn't try to look for dependencies/conflicts
#pragma HLS INTERFACE m_axi offset = slave port = c_read bundle = c_read
#pragma HLS INTERFACE m_axi offset = slave port = c_write bundle = c_write
#pragma HLS interface mode = ap_ctrl_none port = return
#pragma HLS DATAFLOW
    hlslib::Stream<PackedFloat> a_to_kernel("a_to_kernel");
    hlslib::Stream<PackedFloat> b_to_kernel("b_to_kernel");
    hlslib::Stream<PackedFloat> c_to_kernel("c_to_kernel");
    hlslib::Stream<PackedFloat> kernel_to_c("kernel_to_c");
    HLSLIB_DATAFLOW_INIT();
    HLSLIB_DATAFLOW_FUNCTION(ReadA, a, a_to_kernel, size_n, size_k);
    HLSLIB_DATAFLOW_FUNCTION(ReadB, b, b_to_kernel, size_k, size_m);
    HLSLIB_DATAFLOW_FUNCTION(ReadC, c_read, c_to_kernel, size_n, size_m);
    HLSLIB_DATAFLOW_FUNCTION(Compute, a_to_kernel, b_to_kernel, c_to_kernel, kernel_to_c, size_n, size_k, size_m);
    HLSLIB_DATAFLOW_FUNCTION(WriteC, kernel_to_c, c_write, size_n, size_m);
    HLSLIB_DATAFLOW_FINALIZE();
}

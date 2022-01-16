#include "MatrixMultiplication.h"

#include <hlslib/xilinx/Simulation.h>
#include <hlslib/xilinx/Stream.h>
#include <hlslib/xilinx/Utility.h>  // hlslib::CeilDivide

#include "ArithmeticOperations.h"
#include "Karatsuba.h"
#include "PipelinedAdd.h"

// Annoyingly we have to specialize the innermost loop on whether multiple DRAM flits per number are required or not,
// because HLS otherwise gets confused by pragmas applied to a loop of size 1 in the latter case.
template <int lines_per_number>
void ReadAInner(DramLine const *const mem, hlslib::Stream<PackedFloat> &a_to_feeder, const int size_n,
                const int tiles_n, const int size_k, const int n0, const int k) {
#pragma HLS INLINE
    DramLine num[kLinesPerNumber];
ReadA_N:
    for (int n1 = 0; n1 < ((n0 < tiles_n - 1) ? kTileSizeN : (size_n - n0 * kTileSizeN)); ++n1) {
    ReadA_Flits:
        for (int i = 0; i < kLinesPerNumber; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
            num[i] = mem[((n0 * kTileSizeN + n1) * size_k + k) * kLinesPerNumber + i];
            if (i == kLinesPerNumber - 1) {
                a_to_feeder.Push(PackedFloat(num));
            }
        }
    }
}

template <>
void ReadAInner<1>(DramLine const *const mem, hlslib::Stream<PackedFloat> &a_to_feeder, const int size_n,
                   const int tiles_n, const int size_k, const int n0, const int k) {
#pragma HLS INLINE
ReadA_N:
    for (int n1 = 0; n1 < ((n0 < tiles_n - 1) ? kTileSizeN : (size_n - n0 * kTileSizeN)); ++n1) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
        DramLine num[1];
        num[0] = mem[(n0 * kTileSizeN + n1) * size_k + k];
        a_to_feeder.Push(PackedFloat(num));
    }
}

void ReadA(DramLine const *const mem, hlslib::Stream<PackedFloat> &a_to_feeder, const int size_n, const int size_k,
           const int size_m) {
    const auto tiles_n = hlslib::CeilDivide(size_n, kTileSizeN);
    const auto tiles_m = hlslib::CeilDivide(size_m, kTileSizeM);
ReadA_TilesN:
    for (int n0 = 0; n0 < tiles_n; ++n0) {
    ReadA_TilesM:
        for (int m0 = 0; m0 < tiles_m; ++m0) {
        ReadA_K:
            for (int k = 0; k < size_k; ++k) {
                ReadAInner<kLinesPerNumber>(mem, a_to_feeder, size_n, tiles_n, size_k, n0, k);
            }
        }
    }
}

// In order to eliminate control logic in the compute function, we introduce extra feeders that run in the iteration
// space of the computational module, but write to the kernel every iteration to absorb the conditional pipeline reads
void FeedA(hlslib::Stream<PackedFloat> &a_to_feeder, hlslib::Stream<PackedFloat> &a_to_kernel, const int size_n,
           const int size_k, const int size_m) {
    const auto tiles_n = hlslib::CeilDivide(size_n, kTileSizeN);
    const auto tiles_m = hlslib::CeilDivide(size_m, kTileSizeM);
    PackedFloat a;
FeedA_TilesN:
    for (int n0 = 0; n0 < tiles_n; ++n0) {
    FeedA_TilesM:
        for (int m0 = 0; m0 < tiles_m; ++m0) {
        FeedA_K:
            for (int k = 0; k < size_k; ++k) {
            FeedA_N:
                for (int n1 = 0; n1 < ((n0 < tiles_n - 1) ? kTileSizeN : (size_n - n0 * kTileSizeN)); ++n1) {
                FeedA_M:
                    for (int m1 = 0; m1 < kTileSizeM; ++m1) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
                        if (m1 == 0) {
                            a = a_to_feeder.Pop();
                        }
                        a_to_kernel.Push(a);
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

template <int lines_per_number>
void ReadBInner(DramLine const *const mem, hlslib::Stream<PackedFloat> &b_to_feeder, const int size_m, const int m0,
                const int k) {
#pragma HLS INLINE
    DramLine num[kLinesPerNumber];
ReadB_M:
    for (int m1 = 0; m1 < kTileSizeM; ++m1) {
    ReadB_Flits:
        for (int i = 0; i < kLinesPerNumber; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
            num[i] = mem[(k * size_m + m0 * kTileSizeM + m1) * kLinesPerNumber + i];
            if (i == kLinesPerNumber - 1) {
                b_to_feeder.Push(PackedFloat(num));
            }
        }
    }
}

template <>
void ReadBInner<1>(DramLine const *const mem, hlslib::Stream<PackedFloat> &b_to_feeder, const int size_m, const int m0,
                   const int k) {
#pragma HLS INLINE
ReadB_M:
    for (int m1 = 0; m1 < kTileSizeM; ++m1) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
        DramLine num[1];
        num[0] = mem[k * size_m + m0 * kTileSizeM + m1];
        b_to_feeder.Push(PackedFloat(num));
    }
}

void ReadB(DramLine const *const mem, hlslib::Stream<PackedFloat> &b_to_feeder, const int size_n, const int size_k,
           const int size_m) {
    const auto tiles_n = hlslib::CeilDivide(size_n, kTileSizeN);
    const auto tiles_m = hlslib::CeilDivide(size_m, kTileSizeM);
ReadB_TilesN:
    for (int n0 = 0; n0 < tiles_n; ++n0) {
    ReadB_TilesM:
        for (int m0 = 0; m0 < tiles_m; ++m0) {
        ReadB_K:
            for (int k = 0; k < size_k; ++k) {
                ReadBInner<kLinesPerNumber>(mem, b_to_feeder, size_m, m0, k);
            }
        }
    }
}

void FeedB(hlslib::Stream<PackedFloat> &b_to_feeder, hlslib::Stream<PackedFloat> &b_to_kernel, const int size_n,
           const int size_k, const int size_m) {
    const auto tiles_n = hlslib::CeilDivide(size_n, kTileSizeN);
    const auto tiles_m = hlslib::CeilDivide(size_m, kTileSizeM);
    PackedFloat b;
FeedB_TilesN:
    for (int n0 = 0; n0 < tiles_n; ++n0) {
    FeedB_TilesM:
        for (int m0 = 0; m0 < tiles_m; ++m0) {
        FeedB_K:
            for (int k = 0; k < size_k; ++k) {
            FeedB_N:
                for (int n1 = 0; n1 < ((n0 < tiles_n - 1) ? kTileSizeN : (size_n - n0 * kTileSizeN)); ++n1) {
                FeedB_M:
                    for (int m1 = 0; m1 < kTileSizeM; ++m1) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
                        if (n1 == 0) {
                            b = b_to_feeder.Pop();
                        }
                        b_to_kernel.Push(b);
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

template <int lines_per_number>
void ReadCInner(DramLine const *const mem, hlslib::Stream<PackedFloat> &c_to_feeder, const int size_m, const int n0,
                const int m0, const int n1) {
#pragma HLS INLINE
ReadC_M:
    for (int m1 = 0; m1 < kTileSizeM; ++m1) {
        DramLine num[kLinesPerNumber];
    ReadC_Flits:
        for (int i = 0; i < kLinesPerNumber; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
            num[i] = mem[((n0 * kTileSizeN + n1) * size_m + m0 * kTileSizeM + m1) * kLinesPerNumber + i];
            if (i == kLinesPerNumber - 1) {
                c_to_feeder.Push(PackedFloat(num));
            }
        }
    }
}

template <>
void ReadCInner<1>(DramLine const *const mem, hlslib::Stream<PackedFloat> &c_to_feeder, const int size_m, const int n0,
                   const int m0, const int n1) {
#pragma HLS INLINE
ReadC_M:
    for (int m1 = 0; m1 < kTileSizeM; ++m1) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
        DramLine num[1];
        num[0] = mem[(n0 * kTileSizeN + n1) * size_m + m0 * kTileSizeM + m1];
        c_to_feeder.Push(PackedFloat(num));
    }
}

void ReadC(DramLine const *const mem, hlslib::Stream<PackedFloat> &c_to_feeder, const int size_n, const int size_m) {
    const auto tiles_n = hlslib::CeilDivide(size_n, kTileSizeN);
    const auto tiles_m = hlslib::CeilDivide(size_m, kTileSizeM);
ReadC_TilesN:
    for (int n0 = 0; n0 < tiles_n; ++n0) {
    ReadC_TilesM:
        for (int m0 = 0; m0 < tiles_m; ++m0) {
        ReadC_N:
            for (int n1 = 0; n1 < ((n0 < tiles_n - 1) ? kTileSizeN : (size_n - n0 * kTileSizeN)); ++n1) {
                ReadCInner<kLinesPerNumber>(mem, c_to_feeder, size_m, n0, m0, n1);
            }
        }
    }
}

void FeedC(hlslib::Stream<PackedFloat> &c_to_feeder, hlslib::Stream<PackedFloat> &c_to_kernel, const int size_n,
           const int size_k, const int size_m) {
    const auto tiles_n = hlslib::CeilDivide(size_n, kTileSizeN);
    const auto tiles_m = hlslib::CeilDivide(size_m, kTileSizeM);
    PackedFloat c;
FeedC_TilesN:
    for (int n0 = 0; n0 < tiles_n; ++n0) {
    FeedC_TilesM:
        for (int m0 = 0; m0 < tiles_m; ++m0) {
        FeedC_K:
            for (int k = 0; k < size_k; ++k) {
            FeedC_N:
                for (int n1 = 0; n1 < ((n0 < tiles_n - 1) ? kTileSizeN : (size_n - n0 * kTileSizeN)); ++n1) {
                FeedC_M:
                    for (int m1 = 0; m1 < kTileSizeM; ++m1) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
                        if (k == 0) {
                            c = c_to_feeder.Pop();
                        }
                        c_to_kernel.Push(c);
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void DrainC(hlslib::Stream<PackedFloat> &c_to_drainer, hlslib::Stream<PackedFloat> &drainer_to_c, const int size_n,
            const int size_k, const int size_m) {
    const auto tiles_n = hlslib::CeilDivide(size_n, kTileSizeN);
    const auto tiles_m = hlslib::CeilDivide(size_m, kTileSizeM);
DrainC_TilesN:
    for (int n0 = 0; n0 < tiles_n; ++n0) {
    DrainC_TilesM:
        for (int m0 = 0; m0 < tiles_m; ++m0) {
        DrainC_K:
            for (int k = 0; k < size_k; ++k) {
            DrainC_N:
                for (int n1 = 0; n1 < ((n0 < tiles_n - 1) ? kTileSizeN : (size_n - n0 * kTileSizeN)); ++n1) {
                DrainC_M:
                    for (int m1 = 0; m1 < kTileSizeM; ++m1) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
                        const auto c = c_to_drainer.Pop();
                        if (k == size_k - 1) {
                            drainer_to_c.Push(c);
                        }
                    }
                }
            }
        }
    }
}

template <int lines_per_number>
void WriteCInner(hlslib::Stream<PackedFloat> &from_kernel, DramLine *const mem, const int size_n, const int size_m,
                 const int n0, const int m0, const int n1) {
#pragma HLS INLINE
WriteC_M:
    for (int m1 = 0; m1 < kTileSizeM; ++m1) {
        DramLine num[kLinesPerNumber];
#pragma HLS ARRAY_PARTITION variable = num complete
    WriteC_Flits:
        for (int i = 0; i < kLinesPerNumber; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
            if (i == 0) {
                from_kernel.Pop().UnpackFlits(num);
            }
            const bool in_bounds = (n0 * kTileSizeN + n1 < size_n) && (m0 * kTileSizeM + m1 < size_m);
            if (in_bounds) {
                mem[((n0 * kTileSizeN + n1) * size_m + m0 * kTileSizeM + m1) * kLinesPerNumber + i] = num[i];
            }
        }
    }
}

template <>
void WriteCInner<1>(hlslib::Stream<PackedFloat> &from_kernel, DramLine *const mem, const int size_n, const int size_m,
                    const int n0, const int m0, const int n1) {
#pragma HLS INLINE
WriteC_M:
    for (int m1 = 0; m1 < kTileSizeM; ++m1) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
        DramLine num[1];
        from_kernel.Pop().UnpackFlits(num);
        const bool in_bounds = (n0 * kTileSizeN + n1 < size_n) && (m0 * kTileSizeM + m1 < size_m);
        if (in_bounds) {
            mem[(n0 * kTileSizeN + n1) * size_m + m0 * kTileSizeM + m1] = num[0];
        }
    }
}

void WriteC(hlslib::Stream<PackedFloat> &from_kernel, DramLine *const mem, const int size_n, int const size_m) {
    const auto tiles_n = hlslib::CeilDivide(size_n, kTileSizeN);
    const auto tiles_m = hlslib::CeilDivide(size_m, kTileSizeM);
WriteC_TilesN:
    for (int n0 = 0; n0 < tiles_n; ++n0) {
    WriteC_TilesM:
        for (int m0 = 0; m0 < tiles_m; ++m0) {
        WriteC_N:
            for (int n1 = 0; n1 < ((n0 < tiles_n - 1) ? kTileSizeN : (size_n - n0 * kTileSizeN)); ++n1) {
                WriteCInner<kLinesPerNumber>(from_kernel, mem, size_n, size_m, n0, m0, n1);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void ComputeEntry(hlslib::Stream<PackedFloat> &a_in, hlslib::Stream<PackedFloat> &b_in,
                  hlslib::Stream<MantissaFlat> &a_out, hlslib::Stream<MantissaFlat> &b_out,
                  hlslib::Stream<ap_uint<8 * sizeof(Exponent)>> &ab_bypass, int const size_n, int const size_k,
                  int const size_m) {
    PackedFloat a_buffer;
    PackedFloat b_buffer[kTileSizeM];
    const int tiles_n = hlslib::CeilDivide(size_n, kTileSizeN);
    const int tiles_m = hlslib::CeilDivide(size_m, kTileSizeM);
ComputeEntry_TilesN:
    for (int n0 = 0; n0 < tiles_n; ++n0) {
    ComputeEntry_TilesM:
        for (int m0 = 0; m0 < tiles_m; ++m0) {
        ComputeEntry_K:
            for (int k = 0; k < size_k; ++k) {
            ComputeEntry_N:
                for (int n1 = 0; n1 < ((n0 < tiles_n - 1) ? kTileSizeN : (size_n - n0 * kTileSizeN)); ++n1) {
                ComputeEntry_M:
                    for (int m1 = 0; m1 < kTileSizeM; ++m1) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
                        const PackedFloat a_read = a_in.Pop();
                        const PackedFloat b_read = b_in.Pop();
                        PackedFloat a = (m1 == 0) ? a_read : a_buffer;
                        PackedFloat b = (n1 == 0) ? b_read : b_buffer[m1];
                        a_buffer = a;
                        b_buffer[m1] = b;
                        // Ignore contributions from out-of-bound indices
                        const bool in_bounds = (n0 * kTileSizeN + n1 < size_n) && (m0 * kTileSizeM + m1 < size_m);
                        if (!in_bounds) {
                            a.SetZero();
                            b.SetZero();
                        }
                        // Multiplication prologue
                        ap_uint<8 * sizeof(Exponent)> sign_exponent;
                        reinterpret_cast<PackedSignExponent *>(&sign_exponent)->sign = a.GetSignBit() != b.GetSignBit();
                        reinterpret_cast<PackedSignExponent *>(&sign_exponent)->exponent =
                            a.GetExponent() + b.GetExponent();
                        ab_bypass.Push(sign_exponent);
                        a_out.Push(a.GetMantissa());
                        b_out.Push(b.GetMantissa());
                    }
                }
            }
        }
    }
}

void ComputeExit(hlslib::Stream<ap_uint<kMantissaBits + 1>> &ab_mantissa_in,
                 hlslib::Stream<ap_uint<8 * sizeof(Exponent)>> &ab_bypass, hlslib::Stream<PackedFloat> &c_in,
                 hlslib::Stream<PackedFloat> &c_out, int const size_n, int const size_k, int const size_m) {
    PackedFloat c_buffer[kTileSizeN * kTileSizeM];
    const int tiles_n = hlslib::CeilDivide(size_n, kTileSizeN);
    const int tiles_m = hlslib::CeilDivide(size_m, kTileSizeM);
ComputeExit_TilesN:
    for (int n0 = 0; n0 < tiles_n; ++n0) {
    ComputeExit_TilesM:
        for (int m0 = 0; m0 < tiles_m; ++m0) {
        ComputeExit_K:
            for (int k = 0; k < size_k; ++k) {
            ComputeExit_N:
                for (int n1 = 0; n1 < ((n0 < tiles_n - 1) ? kTileSizeN : (size_n - n0 * kTileSizeN)); ++n1) {
                ComputeExit_M:
                    for (int m1 = 0; m1 < kTileSizeM; ++m1) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
                        const auto _ab_mantissa = ab_mantissa_in.Pop();
                        const auto ab_sign_exponent = ab_bypass.Pop();
                        // Matrix multiplication epilogue
                        PackedFloat ab;
                        ab.SetSignExponent(ab_sign_exponent);
                        const bool should_be_shifted = !IsMostSignificantBitSet(_ab_mantissa);
                        const ap_uint<kMantissaBits + 1> m_mantissa =
                            should_be_shifted ? _ab_mantissa : (_ab_mantissa >> 1);
                        ab.SetMantissa(m_mantissa);
                        // If the most significant bit was 0, subtract 1 due to the shift.
                        ab.SetExponent(ab.GetExponent() - (should_be_shifted ? 1 : 0));
                        // Addition
                        const PackedFloat c_read = c_in.Pop();
                        const PackedFloat c = (k == 0) ? c_read : c_buffer[n1 * kTileSizeM + m1];
                        const PackedFloat res = Add(ab, c);
                        c_out.Push(res);
                        c_buffer[n1 * kTileSizeM + m1] = res;
#pragma HLS DEPENDENCE variable = c_buffer false
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

template <int bits>
void StreamingKaratsubaEntry(hlslib::Stream<ap_uint<bits>> &a_in, hlslib::Stream<ap_uint<bits>> &b_in,
                             hlslib::Stream<ap_uint<bits / 2>> &a0_out, hlslib::Stream<ap_uint<bits / 2>> &b0_out,
                             hlslib::Stream<ap_uint<bits / 2>> &a1_out, hlslib::Stream<ap_uint<bits / 2>> &b1_out,
                             hlslib::Stream<ap_uint<bits / 2>> &a0a1_out, hlslib::Stream<ap_uint<bits / 2>> &b0b1_out,
                             hlslib::Stream<bool> &sign_out) {
#pragma HLS PIPELINE II = 1
    using Half = ap_uint<bits / 2>;

    const auto a = a_in.Pop();
    const auto b = b_in.Pop();

    // Decompose input operands into halves for the recursive step
    Half a0 = a.range(bits / 2 - 1, 0);
    Half a1 = a.range(bits - 1, bits / 2);
    Half b0 = b.range(bits / 2 - 1, 0);
    Half b1 = b.range(bits - 1, bits / 2);

    // Compute |a_0 - a_1| and sign(a_0 - a_1)
    bool a0a1_is_neg = a0 < a1;
    Half a0a1 = PipelinedSub(a0a1_is_neg ? a1 : a0, a0a1_is_neg ? a0 : a1);
    // Compute |b_1 - b_0| and sign(b_1 - b_0)
    bool b0b1_is_neg = b1 < b0;
    Half b0b1 = PipelinedSub(b0b1_is_neg ? b0 : b1, b0b1_is_neg ? b1 : b0);

    // XOR the two signs to get the final sign
    bool a0a1b0b1_is_neg = a0a1_is_neg != b0b1_is_neg;

    a0_out.Push(a0);
    a1_out.Push(a1);
    b0_out.Push(b0);
    b1_out.Push(b1);
    a0a1_out.Push(a0a1);
    b0b1_out.Push(b0b1);
    sign_out.Push(a0a1b0b1_is_neg);
}

template <int bits>
void StreamingKaratsubaExit(hlslib::Stream<ap_uint<bits>> &z0_in, hlslib::Stream<ap_uint<bits>> &z2_in,
                            hlslib::Stream<bool> &sign_in, hlslib::Stream<ap_uint<bits>> &a0a1b0b1_in,
                            hlslib::Stream<ap_uint<2 * bits>> &result_out) {
#pragma HLS PIPELINE II = 1
    using Full = ap_uint<bits>;

    // Get results from recursive modules
    const Full z0 = z0_in.Pop();
    const Full z2 = z2_in.Pop();
    const bool a0a1b0b1_is_neg = sign_in.Pop();
    const Full a0a1b0b1 = a0a1b0b1_in.Pop();

    const ap_int<bits + 2> a0a1b0b1_signed = a0a1b0b1_is_neg ? -ap_int<bits + 1>(a0a1b0b1) : ap_int<bits + 2>(a0a1b0b1);
    const ap_uint<bits + 2> z1 = PipelinedAdd<bits + 2>(ap_uint<bits + 2>(a0a1b0b1_signed), PipelinedAdd<bits>(z0, z2));

    // Align everything and combine
    const ap_uint<(2 * bits)> z0z2 = z0 | (ap_uint<(2 * bits)>(z2) << bits);
    const ap_uint<(bits + 2 + bits / 2)> z1_aligned = ap_uint<(bits + 2 + bits / 2)>(z1) << (bits / 2);
    const ap_uint<(2 * bits) + 1> z = PipelinedAdd<2 * bits>(z1_aligned, z0z2);

    result_out.Push(z);
}

template <int bits>
auto StreamingKaratsuba(hlslib::Stream<ap_uint<bits>> &a_in, hlslib::Stream<ap_uint<bits>> &b_in,
                        hlslib::Stream<ap_uint<2 * bits>> &result_out) ->
    typename std::enable_if<(bits > kStreamingBaseBits), void>::type {
    static_assert(bits % 2 == 0, "Number of bits must be even.");
#pragma HLS INLINE
    hlslib::Stream<ap_uint<bits / 2>> a0;
    hlslib::Stream<ap_uint<bits / 2>> b0;
    hlslib::Stream<ap_uint<bits / 2>> a1;
    hlslib::Stream<ap_uint<bits / 2>> b1;
    hlslib::Stream<ap_uint<bits / 2>> a0a1;
    hlslib::Stream<ap_uint<bits / 2>> b0b1;
    hlslib::Stream<bool, 512> sign;
    hlslib::Stream<ap_uint<bits>> z0;
    hlslib::Stream<ap_uint<bits>> z2;
    hlslib::Stream<ap_uint<bits>> a0a1b0b1;
    StreamingKaratsubaEntry<bits>(a_in, b_in, a0, b0, a1, b1, a0a1, b0b1, sign);
    StreamingKaratsuba<(bits / 2)>(a0, b0, z0);
    StreamingKaratsuba<(bits / 2)>(a1, b1, z2);
    StreamingKaratsuba<(bits / 2)>(a0a1, b0b1, a0a1b0b1);
    StreamingKaratsubaExit<bits>(z0, z2, sign, a0a1b0b1, result_out);
}

template <int bits>
auto StreamingKaratsuba(hlslib::Stream<ap_uint<bits>> &a_in, hlslib::Stream<ap_uint<bits>> &b_in,
                        hlslib::Stream<ap_uint<2 * bits>> &result_out) ->
    typename std::enable_if<(bits <= kStreamingBaseBits), void>::type {
#pragma HLS PIPELINE II = 1
    result_out.Push(Karatsuba<bits>(a_in.Pop(), b_in.Pop()));
}

void Truncate(hlslib::Stream<ap_uint<2 * kMantissaBits>> &ab_in, hlslib::Stream<ap_uint<kMantissaBits + 1>> &ab_out) {
#pragma HLS PIPELINE II = 1
    const ap_uint<kMantissaBits + 1> ab_mantissa = ab_in.Pop() >> (kMantissaBits - 1);
    ab_out.Push(ab_mantissa);
}

void FreeRunningMultiplication(hlslib::Stream<MantissaFlat> &a_to_kernel, hlslib::Stream<MantissaFlat> &b_to_kernel,
                               hlslib::Stream<ap_uint<kMantissaBits + 1>> &ab_from_kernel) {
#pragma HLS INTERFACE axis port = a_to_kernel
#pragma HLS INTERFACE axis port = b_to_kernel
#pragma HLS INTERFACE axis port = ab_from_kernel
#pragma HLS interface ap_ctrl_none port = return
#pragma HLS DATAFLOW
    hlslib::Stream<ap_uint<2 * kMantissaBits>> truncate;
    StreamingKaratsuba<kMantissaBits>(a_to_kernel, b_to_kernel, truncate);
    Truncate(truncate, ab_from_kernel);
}

////////////////////////////////////////////////////////////////////////////////

void MatrixMultiplication(DramLine const *const a, DramLine const *const b, DramLine const *const c_read,
                          DramLine *const c_write, const int size_n, const int size_k, int const size_m,
                          hlslib::Stream<MantissaFlat> &a_to_kernel, hlslib::Stream<MantissaFlat> &b_to_kernel,
                          hlslib::Stream<ap_uint<kMantissaBits + 1>> &ab_from_kernel) {
#pragma HLS INTERFACE m_axi offset = slave port = a bundle = a
#pragma HLS INTERFACE m_axi offset = slave port = b bundle = b
// Even though they actually point to the same memory location, we use two separate interfaces for reading and writing
// C, to make sure that the compiler doesn't try to look for dependencies/conflicts
#pragma HLS INTERFACE m_axi offset = slave port = c_read bundle = c_read
#pragma HLS INTERFACE m_axi offset = slave port = c_write bundle = c_write
#pragma HLS INTERFACE s_axilite port = a
#pragma HLS INTERFACE s_axilite port = b
#pragma HLS INTERFACE s_axilite port = c_read
#pragma HLS INTERFACE s_axilite port = c_write
#pragma HLS INTERFACE s_axilite port = size_n
#pragma HLS INTERFACE s_axilite port = size_k
#pragma HLS INTERFACE s_axilite port = size_m
#pragma HLS INTERFACE axis port = a_to_kernel
#pragma HLS INTERFACE axis port = b_to_kernel
#pragma HLS INTERFACE axis port = ab_from_kernel
#pragma HLS STABLE variable = a
#pragma HLS STABLE variable = b
#pragma HLS STABLE variable = c_read
#pragma HLS STABLE variable = c_write
#pragma HLS STABLE variable = size_n
#pragma HLS STABLE variable = size_k
#pragma HLS STABLE variable = size_m
#pragma HLS DATAFLOW
    hlslib::Stream<PackedFloat, 16> a_to_feeder("a_to_feeder");
    hlslib::Stream<PackedFloat, 16> a_to_entry("a_to_entry");
    hlslib::Stream<PackedFloat, 16> b_to_feeder("b_to_feeder");
    hlslib::Stream<PackedFloat, 16> b_to_entry("b_to_entry");
    hlslib::Stream<ap_uint<8 * sizeof(Exponent)>, 1024> ab_bypass("ab_bypass");
    hlslib::Stream<PackedFloat, 16> c_to_feeder("c_to_feeder");
    hlslib::Stream<PackedFloat, 16> c_to_kernel("c_to_kernel");
    hlslib::Stream<PackedFloat, 16> c_from_kernel("c_from_kernel");
    hlslib::Stream<PackedFloat, 16> c_from_exit("c_from_exit");
    hlslib::Stream<PackedFloat, 16> c_from_drainer("c_from_drainer");
    HLSLIB_DATAFLOW_INIT();
    HLSLIB_DATAFLOW_FUNCTION(ReadA, a, a_to_feeder, size_n, size_k, size_m);
    HLSLIB_DATAFLOW_FUNCTION(FeedA, a_to_feeder, a_to_entry, size_n, size_k, size_m);
    HLSLIB_DATAFLOW_FUNCTION(ReadB, b, b_to_feeder, size_n, size_k, size_m);
    HLSLIB_DATAFLOW_FUNCTION(FeedB, b_to_feeder, b_to_entry, size_n, size_k, size_m);
    HLSLIB_DATAFLOW_FUNCTION(ReadC, c_read, c_to_feeder, size_n, size_m);
    HLSLIB_DATAFLOW_FUNCTION(FeedC, c_to_feeder, c_to_kernel, size_n, size_k, size_m);
    HLSLIB_DATAFLOW_FUNCTION(ComputeEntry, a_to_entry, b_to_entry, a_to_kernel, b_to_kernel, ab_bypass, size_n, size_k,
                             size_m);
    HLSLIB_DATAFLOW_FUNCTION(ComputeExit, ab_from_kernel, ab_bypass, c_to_kernel, c_from_kernel, size_n, size_k,
                             size_m);
    HLSLIB_DATAFLOW_FUNCTION(DrainC, c_from_kernel, c_from_drainer, size_n, size_k, size_m);
    HLSLIB_DATAFLOW_FUNCTION(WriteC, c_from_drainer, c_write, size_n, size_m);
    HLSLIB_DATAFLOW_FINALIZE();
}

#include "MatrixMultiplication.h"

#include <hlslib/xilinx/Simulation.h>
#include <hlslib/xilinx/Stream.h>
#include <hlslib/xilinx/Utility.h>  // hlslib::CeilDivide

#include "ArithmeticOperations.h"

// Annoyingly we have to specialize the innermost loop on whether multiple DRAM flits per number are required or not,
// because HLS otherwise gets confused by pragmas applied to a loop of size 1 in the latter case.
template <int lines_per_number>
void ReadAInner(DramLine const *const mem, hlslib::Stream<PackedFloat> &to_kernel, const int size_k, const int n0,
                const int k) {
#pragma HLS INLINE
    for (int n1 = 0; n1 < kTileSizeN; ++n1) {
        DramLine num[kLinesPerNumber];
        for (int i = 0; i < kLinesPerNumber; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
            num[i] = mem[((n0 * kTileSizeN + n1) * size_k + k) * kLinesPerNumber + i];
            if (i == kLinesPerNumber - 1) {
                to_kernel.Push(*reinterpret_cast<PackedFloat const *>(num));
            }
        }
    }
}

template <>
void ReadAInner<1>(DramLine const *const mem, hlslib::Stream<PackedFloat> &to_kernel, const int size_k, const int n0,
                   const int k) {
#pragma HLS INLINE
    for (int n1 = 0; n1 < kTileSizeN; ++n1) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
        const auto num = mem[((n0 * kTileSizeN + n1) * size_k + k) * kLinesPerNumber];
        to_kernel.Push(*reinterpret_cast<PackedFloat const *>(&num));
    }
}

void ReadA(DramLine const *const mem, hlslib::Stream<PackedFloat> &to_kernel, const int size_n, const int size_k,
           const int size_m) {
    const auto tiles_n = hlslib::CeilDivide(size_n, kTileSizeN);
    const auto tiles_m = hlslib::CeilDivide(size_m, kTileSizeM);
    for (int n0 = 0; n0 < tiles_n; ++n0) {
        for (int m0 = 0; m0 < tiles_m; ++m0) {
            for (int k = 0; k < size_k; ++k) {
                ReadAInner<kLinesPerNumber>(mem, to_kernel, size_k, n0, k);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

template <int lines_per_number>
void ReadBInner(DramLine const *const mem, hlslib::Stream<PackedFloat> &to_kernel, const int size_m, const int m0,
                const int k) {
#pragma HLS INLINE
    for (int m1 = 0; m1 < kTileSizeM; ++m1) {
        DramLine num[kLinesPerNumber];
        for (int i = 0; i < kLinesPerNumber; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
            num[i] = mem[(k * size_m + m0 * kTileSizeM + m1) * kLinesPerNumber + i];
            if (i == kLinesPerNumber - 1) {
                to_kernel.Push(*reinterpret_cast<PackedFloat const *>(num));
            }
        }
    }
}

template <>
void ReadBInner<1>(DramLine const *const mem, hlslib::Stream<PackedFloat> &to_kernel, const int size_m, const int m0,
                   const int k) {
#pragma HLS INLINE
    for (int m1 = 0; m1 < kTileSizeM; ++m1) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
        const auto num = mem[(k * size_m + m0 * kTileSizeM + m1) * kLinesPerNumber];
        to_kernel.Push(*reinterpret_cast<PackedFloat const *>(&num));
    }
}

void ReadB(DramLine const *const mem, hlslib::Stream<PackedFloat> &to_kernel, const int size_n, const int size_k,
           const int size_m) {
    const auto tiles_n = hlslib::CeilDivide(size_n, kTileSizeN);
    const auto tiles_m = hlslib::CeilDivide(size_m, kTileSizeM);
    for (int n0 = 0; n0 < tiles_n; ++n0) {
        for (int m0 = 0; m0 < tiles_m; ++m0) {
            for (int k = 0; k < size_k; ++k) {
                ReadBInner<kLinesPerNumber>(mem, to_kernel, size_m, m0, k);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

template <int lines_per_number>
void ReadCInner(DramLine const *const mem, hlslib::Stream<PackedFloat> &to_kernel, const int size_m, const int n0,
                const int m0, const int n1) {
#pragma HLS INLINE
    for (int m1 = 0; m1 < kTileSizeM; ++m1) {
        DramLine num[kLinesPerNumber];
        for (int i = 0; i < kLinesPerNumber; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
            num[i] = mem[((n0 * kTileSizeN + n1) * size_m + m0 * kTileSizeM + m1) * kLinesPerNumber + i];
            if (i == kLinesPerNumber - 1) {
                to_kernel.Push(*reinterpret_cast<PackedFloat const *>(num));
            }
        }
    }
}

template <>
void ReadCInner<1>(DramLine const *const mem, hlslib::Stream<PackedFloat> &to_kernel, const int size_m, const int n0,
                   const int m0, const int n1) {
#pragma HLS INLINE
    for (int m1 = 0; m1 < kTileSizeM; ++m1) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
        const auto num = mem[((n0 * kTileSizeN + n1) * size_m + m0 * kTileSizeM + m1) * kLinesPerNumber];
        to_kernel.Push(*reinterpret_cast<PackedFloat const *>(&num));
    }
}

void ReadC(DramLine const *const mem, hlslib::Stream<PackedFloat> &to_kernel, const int size_n, const int size_m) {
    const auto tiles_n = hlslib::CeilDivide(size_n, kTileSizeN);
    const auto tiles_m = hlslib::CeilDivide(size_m, kTileSizeM);
    for (int n0 = 0; n0 < tiles_n; ++n0) {
        for (int m0 = 0; m0 < tiles_m; ++m0) {
            for (int n1 = 0; n1 < kTileSizeN; ++n1) {
                ReadCInner<kLinesPerNumber>(mem, to_kernel, size_m, n0, m0, n1);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

template <int lines_per_number>
void WriteCInner(hlslib::Stream<PackedFloat> &from_kernel, DramLine *const mem, const int size_m, const int n0,
                 const int m0, const int n1) {
#pragma HLS INLINE
    for (int m1 = 0; m1 < kTileSizeM; ++m1) {
        DramLine num[kLinesPerNumber];
#pragma HLS ARRAY_PARTITION variable = num complete
        for (int i = 0; i < kLinesPerNumber; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
            if (i == 0) {
                *reinterpret_cast<PackedFloat *>(num) = from_kernel.Pop();
            }
            mem[((n0 * kTileSizeN + n1) * size_m + m0 * kTileSizeM + m1) * kLinesPerNumber + i] = num[i];
        }
    }
}

template <>
void WriteCInner<1>(hlslib::Stream<PackedFloat> &from_kernel, DramLine *const mem, const int size_m, const int n0,
                    const int m0, const int n1) {
#pragma HLS INLINE
    for (int m1 = 0; m1 < kTileSizeM; ++m1) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
        const auto num = from_kernel.Pop();
        mem[((n0 * kTileSizeN + n1) * size_m + m0 * kTileSizeM + m1) * kLinesPerNumber] =
            *reinterpret_cast<DramLine const *>(&num);
    }
}

void WriteC(hlslib::Stream<PackedFloat> &from_kernel, DramLine *const mem, const int size_n, int const size_m) {
    for (int n0 = 0; n0 < hlslib::CeilDivide(size_n, kTileSizeN); ++n0) {
        for (int m0 = 0; m0 < hlslib::CeilDivide(size_m, kTileSizeM); ++m0) {
            for (int n1 = 0; n1 < kTileSizeN; ++n1) {
                WriteCInner<kLinesPerNumber>(from_kernel, mem, size_m, n0, m0, n1);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void Compute(hlslib::Stream<PackedFloat> &a_in, hlslib::Stream<PackedFloat> &b_in, hlslib::Stream<PackedFloat> &c_in,
             hlslib::Stream<PackedFloat> &c_out, int const size_n, int const size_k, int const size_m) {
    PackedFloat a_buffer;  // Just to make A symmetric to B and C
    PackedFloat b_buffer[kTileSizeM];
    PackedFloat c_buffer[kTileSizeN * kTileSizeM];
    for (int n0 = 0; n0 < hlslib::CeilDivide(size_n, kTileSizeN); ++n0) {
        for (int m0 = 0; m0 < hlslib::CeilDivide(size_m, kTileSizeM); ++m0) {
            for (int k = 0; k < size_k; ++k) {
                for (int n1 = 0; n1 < kTileSizeN; ++n1) {
                    for (int m1 = 0; m1 < kTileSizeM; ++m1) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
                        PackedFloat a, b, c;
                        if (m1 == 0) {
                            a = a_in.Pop();
                            a_buffer = a;
                        } else {
                            a = a_buffer;
                        }
                        if (n1 == 0) {
                            b = b_in.Pop();
                            b_buffer[m1] = b;
                        } else {
                            b = b_buffer[m1];
                        }
                        if (k == 0) {
                            c = c_in.Pop();
                        } else {
                            c = c_buffer[n1 * kTileSizeM + m1];
                        }
                        // Ignore contributions from out-of-bound indices
                        const bool in_bounds = (n0 * kTileSizeN + n1 < size_n) && (m0 * kTileSizeM + m1 < size_m);
                        // Meat of the computation
                        const auto res = in_bounds ? MultiplyAccumulate(a, b, c) : c;
                        // Write back to buffer
                        c_buffer[n1 * kTileSizeM + m1] = res;
                        // Write out on last slice
                        if (k == size_k - 1) {
                            c_out.Push(res);
                        }
                    }
                }
            }
        }
    }
}

void MatrixMultiplication(DramLine const *const a, DramLine const *const b, DramLine const *const c_read,
                          DramLine *const c_write, const int size_n, const int size_k, int const size_m) {
#pragma HLS INTERFACE m_axi offset = slave port = a bundle = a
#pragma HLS INTERFACE m_axi offset = slave port = b bundle = b
// Even though they actually point to the same memory location, we use two separate interfaces for reading and writing
// C, to make sure that the compiler doesn't try to look for dependencies/conflicts
#pragma HLS INTERFACE m_axi offset = slave port = c_read bundle = c_read
#pragma HLS INTERFACE m_axi offset = slave port = c_write bundle = c_write
#pragma HLS DATAFLOW
    hlslib::Stream<PackedFloat> a_to_kernel("a_to_kernel");
    hlslib::Stream<PackedFloat> b_to_kernel("b_to_kernel");
    hlslib::Stream<PackedFloat> c_to_kernel("c_to_kernel");
    hlslib::Stream<PackedFloat> kernel_to_c("kernel_to_c");
    HLSLIB_DATAFLOW_INIT();
    HLSLIB_DATAFLOW_FUNCTION(ReadA, a, a_to_kernel, size_n, size_k, size_m);
    HLSLIB_DATAFLOW_FUNCTION(ReadB, b, b_to_kernel, size_n, size_k, size_m);
    HLSLIB_DATAFLOW_FUNCTION(ReadC, c_read, c_to_kernel, size_n, size_m);
    HLSLIB_DATAFLOW_FUNCTION(Compute, a_to_kernel, b_to_kernel, c_to_kernel, kernel_to_c, size_n, size_k, size_m);
    HLSLIB_DATAFLOW_FUNCTION(WriteC, kernel_to_c, c_write, size_n, size_m);
    HLSLIB_DATAFLOW_FINALIZE();
}

#include "MatrixMultiplication.h"

#include <hlslib/xilinx/Simulation.h>
#include <hlslib/xilinx/Stream.h>

void ReadA(DramLine const *mem, hlslib::Stream<PackedFloat> &to_kernel, int size_n, int size_k) {
    for (int n = 0; n < size_n; ++n) {
        for (int k = 0; k < size_k; ++k) {
            DramLine num[kLinesPerNumber];
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

void ReadB(DramLine const *mem, hlslib::Stream<PackedFloat> &to_kernel, int size_k, int size_m) {
    for (int k = 0; k < size_k; ++k) {
        for (int m = 0; m < size_m; ++m) {
            DramLine num[kLinesPerNumber];
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

void ReadC(DramLine const *mem, hlslib::Stream<PackedFloat> &to_kernel, int size_n, int size_m) {
    for (int n = 0; n < size_n; ++n) {
        for (int m = 0; m < size_m; ++m) {
            DramLine num[kLinesPerNumber];
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

void WriteC(hlslib::Stream<PackedFloat> &from_kernel, DramLine *mem, int size_n, int size_m) {
    for (int n = 0; n < size_n; ++n) {
        for (int m = 0; m < size_m; ++m) {
            DramLine num[kLinesPerNumber];
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

void Compute(hlslib::Stream<PackedFloat> &a_in, hlslib::Stream<PackedFloat> &b_in, hlslib::Stream<PackedFloat> &c_in,
             hlslib::Stream<PackedFloat> &c_out, int size_n, int size_k, int size_m) {
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
                // Some dummy computation
                c.exponent += a.exponent + b.exponent;
                if (k == size_k - 1) {
                    c_out.Push(c);
                }
            }
        }
    }
}

void MatrixMultiplication(DramLine const *a, DramLine const *b, DramLine const *c_read, DramLine *c_write, int size_n,
                          int size_m, int size_k) {
#pragma HLS INTERFACE m_axi offset = slave port = a bundle = a
#pragma HLS INTERFACE m_axi offset = slave port = b bundle = b
// Even though they actually point to the same memory location, we use two separate interfaces for reading and writing
// C, to make sure that the compiler doesn't try to look for dependencies/conflicts
#pragma HLS INTERFACE m_axi offset = slave port = c_read bundle = c_read
#pragma HLS INTERFACE m_axi offset = slave port = c_write bundle = c_write
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

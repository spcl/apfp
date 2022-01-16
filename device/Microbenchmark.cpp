#include "Microbenchmark.h"

#include <hlslib/xilinx/Simulation.h>
#include <hlslib/xilinx/Stream.h>

#include "ArithmeticOperations.h"

template <int lines_per_number>
void Read(DramLine const *const mem, hlslib::Stream<PackedFloat> &to_kernel, const int size) {
    DramLine num[kLinesPerNumber];
#pragma HLS ARRAY_PARTITION variable = num complete
Read:
    for (int i = 0; i < size; ++i) {
    Read_Flits:
        for (int j = 0; j < kLinesPerNumber; ++j) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
            const auto read = mem[i * kLinesPerNumber + j];
            if (j == kLinesPerNumber - 1) {
                to_kernel.Push(PackedFloat(num));
            }
        }
    }
}

template <>
void Read<1>(DramLine const *const mem, hlslib::Stream<PackedFloat> &to_kernel, const int size) {
Read:
    for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II = 1
        const auto read = mem[i];
        to_kernel.Push(*reinterpret_cast<PackedFloat const *>(&read));
    }
}

template <int lines_per_number>
void Write(hlslib::Stream<PackedFloat> &from_kernel, DramLine *const mem, const int size) {
    DramLine num[kLinesPerNumber];
#pragma HLS ARRAY_PARTITION variable = num complete
Write:
    for (int i = 0; i < size; ++i) {
    Write_Flits:
        for (int j = 0; j < kLinesPerNumber; ++j) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
            if (j == 0) {
                from_kernel.Pop().UnpackFlits(num);
            }
            mem[i * kLinesPerNumber + j] = num[j];
        }
    }
}

template <>
void Write<1>(hlslib::Stream<PackedFloat> &from_kernel, DramLine *const mem, const int size) {
Write:
    for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II = 1
        DramLine num[1];
        from_kernel.Pop().UnpackFlits(num);
        mem[i] = num[0];
    }
}

void Compute(hlslib::Stream<PackedFloat> &a_in, hlslib::Stream<PackedFloat> &b_in, hlslib::Stream<PackedFloat> &c_out,
             const int size) {
Compute:
    for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II = 1
        c_out.Push(Multiply(a_in.Pop(), b_in.Pop()));
    }
}

void Microbenchmark(DramLine const *const a, DramLine const *const b, DramLine *const c, const int size) {
#pragma HLS INTERFACE m_axi offset = slave port = a bundle = a
#pragma HLS INTERFACE m_axi offset = slave port = b bundle = b
#pragma HLS INTERFACE m_axi offset = slave port = c bundle = c
#pragma HLS INTERFACE s_axilite port = a
#pragma HLS INTERFACE s_axilite port = b
#pragma HLS INTERFACE s_axilite port = c
#pragma HLS INTERFACE s_axilite port = size
#pragma HLS STABLE variable = a
#pragma HLS STABLE variable = b
#pragma HLS STABLE variable = c
#pragma HLS STABLE variable = size
#pragma HLS DATAFLOW
    hlslib::Stream<PackedFloat, 16> a_to_kernel("a_to_kernel");
    hlslib::Stream<PackedFloat, 16> b_to_kernel("b_to_kernel");
    hlslib::Stream<PackedFloat, 16> c_from_kernel("c_from_kernel");
    HLSLIB_DATAFLOW_INIT();
    HLSLIB_DATAFLOW_FUNCTION(Read<kLinesPerNumber>, a, a_to_kernel, size);
    HLSLIB_DATAFLOW_FUNCTION(Read<kLinesPerNumber>, b, b_to_kernel, size);
    HLSLIB_DATAFLOW_FUNCTION(Compute, a_to_kernel, b_to_kernel, c_from_kernel, size);
    HLSLIB_DATAFLOW_FUNCTION(Write<kLinesPerNumber>, c_from_kernel, c, size);
    HLSLIB_DATAFLOW_FINALIZE();
}

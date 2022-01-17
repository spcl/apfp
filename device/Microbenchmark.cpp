#include "Microbenchmark.h"

#include <hlslib/xilinx/Simulation.h>
#include <hlslib/xilinx/Stream.h>

#include "ArithmeticOperations.h"

#ifndef APFP_FAKE_MEMORY

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

#else

template <int lines_per_number>
void Read(DramLine const *const mem, hlslib::Stream<PackedFloat> &to_kernel, const int size) {
#pragma HLS INLINE
    DramLine flits[kLinesPerNumber];
#pragma HLS ARRAY_PARTITION variable = flits complete
ReadFlits:
    for (int j = 0; j < kLinesPerNumber; ++j) {
#pragma HLS PIPELINE II = 1
        flits[j] = mem[j];
    }
    const PackedFloat num(flits);
ReadFake:
    for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II = 1
        to_kernel.Push(num);
    }
}

template <int lines_per_number>
void Write(hlslib::Stream<PackedFloat> &from_kernel, DramLine *const mem, const int size) {
    DramLine flits[kLinesPerNumber];
WriteFake:
    for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II = 1
        from_kernel.Pop().UnpackFlits(flits);
    }
WriteFlits:
    for (int j = 0; j < kLinesPerNumber; ++j) {
#pragma HLS PIPELINE II = 1
        mem[j] = flits[j];
    }
}

#endif

void ReadA(DramLine const *const mem, hlslib::Stream<PackedFloat> &to_kernel, const int size) {
    Read<kLinesPerNumber>(mem, to_kernel, size);
}

void ReadB(DramLine const *const mem, hlslib::Stream<PackedFloat> &to_kernel, const int size) {
    Read<kLinesPerNumber>(mem, to_kernel, size);
}

void ReadC(DramLine const *const mem, hlslib::Stream<PackedFloat> &to_kernel, const int size) {
    Read<kLinesPerNumber>(mem, to_kernel, size);
}

void Compute(hlslib::Stream<PackedFloat> &a_in, hlslib::Stream<PackedFloat> &b_in, hlslib::Stream<PackedFloat> &c_in,
             hlslib::Stream<PackedFloat> &res_out, const int size) {
Compute:
    for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II = 1
        const auto a = a_in.Pop();
        const auto b = b_in.Pop();
        const auto c = c_in.Pop();
        const auto res = MultiplyAccumulate(a, b, c);
        res_out.Push(res);
    }
}

void Microbenchmark(DramLine const *const a, DramLine const *const b, DramLine const *const c, DramLine *const res,
                    const int size) {
#pragma HLS INTERFACE m_axi offset = slave port = a bundle = a
#pragma HLS INTERFACE m_axi offset = slave port = b bundle = b
#pragma HLS INTERFACE m_axi offset = slave port = c bundle = c
#pragma HLS INTERFACE m_axi offset = slave port = res bundle = res
#pragma HLS INTERFACE s_axilite port = a
#pragma HLS INTERFACE s_axilite port = b
#pragma HLS INTERFACE s_axilite port = c
#pragma HLS INTERFACE s_axilite port = res
#pragma HLS INTERFACE s_axilite port = size
#pragma HLS STABLE variable = a
#pragma HLS STABLE variable = b
#pragma HLS STABLE variable = c
#pragma HLS STABLE variable = res
#pragma HLS STABLE variable = size
#pragma HLS DATAFLOW
    hlslib::Stream<PackedFloat, 16> a_to_kernel("a_to_kernel");
    hlslib::Stream<PackedFloat, 16> b_to_kernel("b_to_kernel");
    hlslib::Stream<PackedFloat, 16> c_to_kernel("b_to_kernel");
    hlslib::Stream<PackedFloat, 16> res_from_kernel("res_from_kernel");
    HLSLIB_DATAFLOW_INIT();
    HLSLIB_DATAFLOW_FUNCTION(ReadA, a, a_to_kernel, size);
    HLSLIB_DATAFLOW_FUNCTION(ReadB, b, b_to_kernel, size);
    HLSLIB_DATAFLOW_FUNCTION(ReadC, c, c_to_kernel, size);
    HLSLIB_DATAFLOW_FUNCTION(Compute, a_to_kernel, b_to_kernel, c_to_kernel, res_from_kernel, size);
    HLSLIB_DATAFLOW_FUNCTION(Write<kLinesPerNumber>, res_from_kernel, res, size);
    HLSLIB_DATAFLOW_FINALIZE();
}

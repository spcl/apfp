#pragma once

#include <hlslib/xilinx/Stream.h>

#include "Config.h"
#include "DeviceTypes.h"
#include "PackedFloat.h"

extern "C" void MatrixMultiplication(DramLine const *const a, DramLine const *const b, DramLine const *const c_read,
                                     DramLine *const c_write, const int size_n, const int size_k, int const size_m,
                                     hlslib::Stream<MantissaFlat> &a_to_kernel,
                                     hlslib::Stream<MantissaFlat> &b_to_kernel,
                                     hlslib::Stream<ap_uint<kMantissaBits + 1>> &ab_from_kernel);

extern "C" void FreeRunningMultiplication(hlslib::Stream<MantissaFlat> &a_mantissa_in,
                                          hlslib::Stream<MantissaFlat> &b_mantissa_in,
                                          hlslib::Stream<ap_uint<kMantissaBits + 1>> &ab_mantissa_out);


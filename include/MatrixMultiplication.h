#pragma once

#include <hlslib/xilinx/Stream.h>

#include "Config.h"
#include "DeviceTypes.h"
#include "PackedFloat.h"

extern "C" void MatrixMultiplication(DramLine const *a, DramLine const *b, DramLine const *c_read, DramLine *c_write,
                                     int n, int m, int k, hlslib::Stream<PackedFloat> &a_to_kernel,
                                     hlslib::Stream<PackedFloat> &b_to_kernel,
                                     hlslib::Stream<PackedFloat> &ab_from_kernel);

extern "C" void FreeRunningMultiplication(hlslib::Stream<PackedFloat> &a_to_kernel,
                                          hlslib::Stream<PackedFloat> &b_to_kernel,
                                          hlslib::Stream<PackedFloat> &ab_from_kernel);

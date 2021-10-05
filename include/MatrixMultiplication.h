#pragma once

#include <ap_int.h>

#include "Config.h"
#include "PackedFloat.h"

using DramLine = ap_uint<512>;
static_assert(sizeof(DramLine) == 64, "DRAM lines must be tightly packed.");

constexpr int kLinesPerNumber = sizeof(PackedFloat) / sizeof(DramLine);
static_assert(sizeof(PackedFloat) % sizeof(DramLine) == 0, "Numbers must be a multiple of DRAM lines.");

extern "C" void MatrixMultiplication(DramLine const *a, DramLine const *b, DramLine const *c_read, DramLine *c_write,
                                     int n, int m, int k);

#pragma once

#include "Config.h"
#include "DeviceTypes.h"

extern "C" void MatrixMultiplication(DramLine const *a, DramLine const *b, DramLine const *c_read, DramLine *c_write,
                                     int n, int m, int k);

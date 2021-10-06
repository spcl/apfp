#pragma once

#include <ap_int.h>

#include "Config.h"

ap_uint<2 * kBits> Karatsuba(ap_uint<kBits> const &a, ap_uint<kBits> const &b);

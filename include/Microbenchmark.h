#pragma once

#include "Config.h"
#include "DeviceTypes.h"

extern "C" void Microbenchmark(DramLine const *const a, DramLine const *const b, DramLine *const c, const int size);


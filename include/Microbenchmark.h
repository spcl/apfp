#pragma once

#include "Config.h"
#include "DeviceTypes.h"

extern "C" void Microbenchmark(DramLine const *const a, DramLine const *const b, DramLine const *const c,
                               DramLine *const res, const int size);


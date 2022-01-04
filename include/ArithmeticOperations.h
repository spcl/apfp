#pragma once

#include "DeviceFloat.h"
#include "PackedFloat.h"

DeviceFloat MultiplyAccumulate(DeviceFloat const &a, DeviceFloat const &b, DeviceFloat const &c);
DeviceFloat Multiply(DeviceFloat const &a, DeviceFloat const &b);
DeviceFloat Add(DeviceFloat const &a, DeviceFloat const &b);

PackedFloat MultiplyAccumulate(PackedFloat const &a, PackedFloat const &b, PackedFloat const &c);
PackedFloat Multiply(PackedFloat const &a, PackedFloat const &b);
PackedFloat Add(PackedFloat const &a, PackedFloat const &b);

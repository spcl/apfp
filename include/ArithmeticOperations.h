#pragma once

#include "PackedFloat.h"

PackedFloat MultiplyAccumulate(PackedFloat const &a, PackedFloat const &b, PackedFloat const &c);
PackedFloat Multiply(PackedFloat const &a, PackedFloat const &b);
PackedFloat Add(PackedFloat const &a, PackedFloat const &b);

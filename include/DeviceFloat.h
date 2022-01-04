#pragma once

#include <ap_int.h>

#include "Config.h"
#include "Types.h"

using MantissaFlat = ap_uint<kMantissaBits>;

/// This data type is used in the FPGA code to pass around numbers in a packed format, but using a wide integer as the
/// internal format, to ensure that it is generated as a single data bus.
class DeviceFloat {
   public:
    DeviceFloat() = default;
    DeviceFloat(DeviceFloat const &) = default;
    DeviceFloat(DeviceFloat &&) = default;

    DeviceFloat(const DramLine flits[kLinesPerNumber]) {
#pragma HLS INLINE
        for (int i = 0; i < kLinesPerNumber; ++i) {
#pragma HLS UNROLL
            SetFlit(i, flits[i]);
        }
    }

    DeviceFloat &operator=(DeviceFloat const &) = default;
    DeviceFloat &operator=(DeviceFloat &&) = default;

    MantissaFlat GetMantissa() const {
#pragma HLS INLINE
        return data_.range(kMantissaBits - 1, 0);
    }

    void SetMantissa(MantissaFlat const &mantissa) {
#pragma HLS INLINE
        data_.range(kMantissaBits - 1, 0) = mantissa;
    }

    Exponent GetExponent() const {
#pragma HLS INLINE
        return data_.range(kBits - 8 * sizeof(Sign) - 1, kBits - 8 * sizeof(Sign) - 8 * sizeof(Exponent));
    }

    void SetExponent(Exponent const &exponent) {
#pragma HLS INLINE
        data_.range(kBits - 8 * sizeof(Sign) - 1, kBits - 8 * sizeof(Sign) - 8 * sizeof(Exponent)) = exponent;
    }

    Sign GetSign() const {
#pragma HLS INLINE
        return data_.range(kBits - 1, kBits - 8 * sizeof(Sign));
    }

    void SetSign(Sign const &sign) {
#pragma HLS INLINE
        data_.range(kBits - 1, kBits - 8 * sizeof(Sign)) = sign;
    }

    DramLine GetFlit(const size_t i) const {
#pragma HLS INLINE
        return data_.range((i + 1) * 512 - 1, i * 512);
    }

    void SetFlit(const size_t i, DramLine const &flit) {
#pragma HLS INLINE
        data_.range((i + 1) * 512 - 1, i * 512) = flit;
    }

    void operator>>(DramLine flits[kLinesPerNumber]) const {
#pragma HLS INLINE
        for (int i = 0; i < kLinesPerNumber; ++i) {
#pragma HLS UNROLL
            flits[i] = GetFlit(i);
        }
    }

    static DeviceFloat Zero() {
#pragma HLS INLINE
        DeviceFloat x;
        x.data_ = 0;
        return x;
    }

   private:
    ap_uint<kBits> data_;
};
static_assert(sizeof(DeviceFloat) == kBytes, "DeviceFloat must be tightly packed.");

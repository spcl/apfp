#pragma once
#include <gmp.h>

using ApfpInterfaceType = mpf_t;

void InitApfpInterfaceType(ApfpInterfaceType value);

void Init2ApfpInterfaceType(ApfpInterfaceType value, unsigned long precision);

void ClearApfpInterfaceType(ApfpInterfaceType value);

void SwapApfpInterfaceType(ApfpInterfaceType a, ApfpInterfaceType b);

void SetApfpInterfaceType(ApfpInterfaceType dest, const ApfpInterfaceType source);

/// Smart pointer-like wrapper class for GMP/MPFR types
class ApfpInterfaceWrapper {
    ApfpInterfaceType data_;

public:
    ~ApfpInterfaceWrapper();

    ApfpInterfaceWrapper();

    ApfpInterfaceWrapper(ApfpInterfaceWrapper&) = delete;

    ApfpInterfaceWrapper& operator=(const ApfpInterfaceWrapper&) = delete;
    
    ApfpInterfaceType* get() { return &data_; }

    const ApfpInterfaceType* get() const { return &data_; }
};
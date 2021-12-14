#include "ApfpInterfaceType.h"

void InitApfpInterfaceType(ApfpInterfaceTypePtr value) {
    mpf_init(value);
}

void Init2ApfpInterfaceType(ApfpInterfaceTypePtr value, unsigned long precision) {
    mpf_init2(value, precision);
}

void ClearApfpInterfaceType(ApfpInterfaceTypePtr value) {
    mpf_clear(value);
}

void SwapApfpInterfaceType(ApfpInterfaceTypePtr a, ApfpInterfaceTypePtr b) {
    mpf_swap(a, b);
}

void SetApfpInterfaceType(ApfpInterfaceTypePtr dest, ApfpInterfaceTypeConstPtr source) {
    mpf_set(dest, source);
}

ApfpInterfaceWrapper::~ApfpInterfaceWrapper() {
    ClearApfpInterfaceType(data_);
}

ApfpInterfaceWrapper::ApfpInterfaceWrapper() {
    InitApfpInterfaceType(data_);
}

ApfpInterfaceWrapper::ApfpInterfaceWrapper(unsigned long precision) {
    Init2ApfpInterfaceType(data_, precision);
}


ApfpInterfaceWrapper::ApfpInterfaceWrapper(ApfpInterfaceWrapper&& other) {
    SwapApfpInterfaceType(data_, other.data_);
    ClearApfpInterfaceType(other.data_);
}

ApfpInterfaceWrapper& ApfpInterfaceWrapper::operator=(ApfpInterfaceWrapper&& other) {
    SwapApfpInterfaceType(data_, other.data_);
    ClearApfpInterfaceType(other.data_);
    return *this;
}

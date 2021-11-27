#include "ApfpInterfaceType.h"

void InitApfpInterfaceType(ApfpInterfaceType value) {
    mpf_init(value);
}

void Init2ApfpInterfaceType(ApfpInterfaceType value, unsigned long precision) {
    mpf_init2(value, precision);
}

void ClearApfpInterfaceType(ApfpInterfaceType value) {
    mpf_clear(value);
}

void SwapApfpInterfaceType(ApfpInterfaceType a, ApfpInterfaceType b) {
    mpf_swap(a, b);
}

void SetApfpInterfaceType(ApfpInterfaceType dest, const ApfpInterfaceType source) {
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

#include "ApfpInterfaceType.h"

void InitApfpInterfaceType(ApfpInterfaceTypePtr value) {
#ifdef APFP_GMP_INTERFACE_TYPE
    mpf_init(value);
#else
    mpfr_init(value);
#endif
}

void Init2ApfpInterfaceType(ApfpInterfaceTypePtr value, unsigned long precision) {
#ifdef APFP_GMP_INTERFACE_TYPE
    mpf_init2(value, precision);
#else
    mpfr_init(value);
    mpfr_set_prec(value, precision);
#endif
}

void ClearApfpInterfaceType(ApfpInterfaceTypePtr value) {
#ifdef APFP_GMP_INTERFACE_TYPE
    mpf_clear(value);
#else
    mpfr_clear(value);
#endif
}

void SwapApfpInterfaceType(ApfpInterfaceTypePtr a, ApfpInterfaceTypePtr b) {
#ifdef APFP_GMP_INTERFACE_TYPE
    mpf_swap(a, b);
#else
    mpfr_swap(a, b);
#endif
}

void SetApfpInterfaceType(ApfpInterfaceTypePtr dest, ApfpInterfaceTypeConstPtr source) {
#ifdef APFP_GMP_INTERFACE_TYPE
    mpf_set(dest, source);
#else
    mpfr_set(dest, source, mpfr_get_default_rounding_mode());
#endif
}

void SetApfpInterfaceType(ApfpInterfaceTypePtr dest, long int source) {
#ifdef APFP_GMP_INTERFACE_TYPE
    mpf_set_ui(dest, source);
#else
    mpfr_set_si(dest, source, mpfr_get_default_rounding_mode());
#endif
}

void AddApfpInterfaceType(ApfpInterfaceTypePtr dest, ApfpInterfaceTypeConstPtr a, ApfpInterfaceTypeConstPtr b) {
#ifdef APFP_GMP_INTERFACE_TYPE
    mpf_add(dest, a, b);
#else
    mpfr_add(dest, a, b, mpfr_get_default_rounding_mode());
#endif
}

void MulApfpInterfaceType(ApfpInterfaceTypePtr dest, ApfpInterfaceTypeConstPtr a, ApfpInterfaceTypeConstPtr b) {
#ifdef APFP_GMP_INTERFACE_TYPE
    mpf_mul(dest, a, b);
#else
    mpfr_mul(dest, a, b, mpfr_get_default_rounding_mode());
#endif
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

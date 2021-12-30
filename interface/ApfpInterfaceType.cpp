#include "ApfpInterfaceType.h"

namespace apfp::interface {

void Init(Ptr value) {
#ifdef APFP_GMP_INTERFACE_TYPE
    mpf_init(value);
#else
    mpfr_init(value);
#endif
}

void Init2(Ptr value, unsigned long precision) {
#ifdef APFP_GMP_INTERFACE_TYPE
    mpf_init2(value, precision);
#else
    mpfr_init(value);
    mpfr_set_prec(value, precision);
#endif
}

void Clear(Ptr value) {
#ifdef APFP_GMP_INTERFACE_TYPE
    mpf_clear(value);
#else
    mpfr_clear(value);
#endif
}

void Swap(Ptr a, Ptr b) {
#ifdef APFP_GMP_INTERFACE_TYPE
    mpf_swap(a, b);
#else
    mpfr_swap(a, b);
#endif
}

void Set(Ptr dest, ConstPtr source) {
#ifdef APFP_GMP_INTERFACE_TYPE
    mpf_set(dest, source);
#else
    mpfr_set(dest, source, mpfr_get_default_rounding_mode());
#endif
}

void Set(Ptr dest, long int source) {
#ifdef APFP_GMP_INTERFACE_TYPE
    mpf_set_ui(dest, source);
#else
    mpfr_set_si(dest, source, mpfr_get_default_rounding_mode());
#endif
}

void Add(Ptr dest, ConstPtr a, ConstPtr b) {
#ifdef APFP_GMP_INTERFACE_TYPE
    mpf_add(dest, a, b);
#else
    mpfr_add(dest, a, b, mpfr_get_default_rounding_mode());
#endif
}

void Mul(Ptr dest, ConstPtr a, ConstPtr b) {
#ifdef APFP_GMP_INTERFACE_TYPE
    mpf_mul(dest, a, b);
#else
    mpfr_mul(dest, a, b, mpfr_get_default_rounding_mode());
#endif
}

Wrapper::~Wrapper() {
    Clear(data_);
}

Wrapper::Wrapper() {
    Init(data_);
}

Wrapper::Wrapper(unsigned long precision) {
    Init2(data_, precision);
}

Wrapper::Wrapper(Wrapper&& other) {
    Swap(data_, other.data_);
    Clear(other.data_);
}

Wrapper& Wrapper::operator=(Wrapper&& other) {
    Swap(data_, other.data_);
    Clear(other.data_);
    return *this;
}

}  // namespace apfp::interface
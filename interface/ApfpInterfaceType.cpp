#include "ApfpInterfaceType.h"

void InitApfpInterfaceType(ApfpInterfaceType value, unsigned int precision) {
    mpf_init2(value, precision);
}

void ClearApfpInterfaceType(ApfpInterfaceType value) {
    mpf_clear(value);
}

void SwapApfpInterfaceType(ApfpInterfaceType a, ApfpInterfaceType b) {
    mpf_swap(a, b);
}

void SetApfpInterfaceType(ApfpInterfaceType dest, ApfpInterfaceType source) {
    mpf_set(dest, source);
}
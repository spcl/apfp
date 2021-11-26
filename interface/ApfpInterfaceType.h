#pragma once
#include <gmp.h>

using ApfpInterfaceType = mpf_t;

void InitApfpInterfaceType(ApfpInterfaceType value, unsigned int precision);

void ClearApfpInterfaceType(ApfpInterfaceType value);

void SwapApfpInterfaceType(ApfpInterfaceType a, ApfpInterfaceType b);

void SetApfpInterfaceType(ApfpInterfaceType dest, ApfpInterfaceType source);
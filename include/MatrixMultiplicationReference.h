#pragma once

#include "PackedFloat.h"

/// Naive reference implementation of matrix multiplication implemented directly on GMP numbers, used for verification.
void MatrixMultiplicationReference(mpfr_t const *a, mpfr_t const *b, mpfr_t *c, int size_n, int size_k, int size_m);

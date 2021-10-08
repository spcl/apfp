#pragma once

#include "PackedFloat.h"

/// Naive reference implementation of matrix multiplication implemented directly on GMP numbers, used for verification.
void MatrixMultiplicationReference(mpf_t const *a, mpf_t const *b, mpf_t *c, int size_n, int size_k, int size_m);

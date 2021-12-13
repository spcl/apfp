#include "MatrixMultiplicationReference.h"

#include <gmp.h>

void MatrixMultiplicationReference(mpfr_t const *a, mpfr_t const *b, mpfr_t *c, int size_n, int size_k, int size_m) {
    mpfr_t tmp;
    mpfr_init2(tmp, kMantissaBits);
    for (int n = 0; n < size_n; ++n) {
        for (int k = 0; k < size_k; ++k) {
            for (int m = 0; m < size_m; ++m) {
                mpfr_mul(tmp, a[n * size_k + k], b[k * size_m + m], kRoundingMode);
                mpfr_t &_c = c[n * size_m + m];
                mpfr_add(_c, _c, tmp, kRoundingMode);
            }
        }
    }
    mpfr_clear(tmp);
}

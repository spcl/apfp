#include "MatrixMultiplicationReference.h"

#include <gmp.h>

void MatrixMultiplicationReference(mpf_t const *a, mpf_t const *b, mpf_t *c, int size_n, int size_k, int size_m) {
    mpf_t tmp;
    mpf_init2(tmp, kMantissaBits);
    for (int n = 0; n < size_n; ++n) {
        for (int k = 0; k < size_k; ++k) {
            for (int m = 0; m < size_m; ++m) {
                mpf_mul(tmp, a[n * size_k + k], b[k * size_m + m]);
                mpf_t &_c = c[n * size_m + m];
                mpf_add(_c, _c, tmp);
            }
        }
    }
    mpf_clear(tmp);
}

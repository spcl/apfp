#include "MicrobenchmarkReference.h"

void MicrobenchmarkReference(mpfr_t const *a, mpfr_t const *b, mpfr_t *c, int size) {
    mpfr_t tmp;
    mpfr_init2(tmp, kMantissaBits);
    for (int i = 0; i < size; ++i) {
        mpfr_mul(c[i], a[i], b[i], kRoundingMode);
    }
    mpfr_clear(tmp);
}

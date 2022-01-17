#include "MicrobenchmarkReference.h"

void MicrobenchmarkReference(mpfr_t const *a, mpfr_t const *b, mpfr_t *c, int size) {
    mpfr_t tmp;
    mpfr_init2(tmp, kMantissaBits);
#ifndef APFP_FAKE_MEMORY
    for (int i = 0; i < size; ++i) {
        mpfr_mul(c[i], a[i], b[i], kRoundingMode);
    }
#else
    mpfr_mul(c[0], a[0], b[0], kRoundingMode);
#endif
    mpfr_clear(tmp);
}

#include "MicrobenchmarkReference.h"

void MicrobenchmarkReference(mpfr_t const *a, mpfr_t const *b, mpfr_t const *c, mpfr_t *res, int size) {
    mpfr_t tmp;
    mpfr_init2(tmp, kMantissaBits);
#ifndef APFP_FAKE_MEMORY
    for (int i = 0; i < size; ++i) {
        mpfr_mul(tmp, a[i], b[i], kRoundingMode);
        mpfr_add(res[i], c[i], tmp, kRoundingMode);
    }
#else
    mpfr_mul(tmp, a[0], b[0], kRoundingMode);
    mpfr_add(res[0], c[0], tmp, kRoundingMode);
#endif
    mpfr_clear(tmp);
}

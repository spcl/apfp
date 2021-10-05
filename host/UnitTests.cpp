#include <catch.hpp>
#include <iostream>

#include "PackedFloat.h"

TEST_CASE("PackedFloat constructor") {
    mpf_t gmp_num;
    mpf_init2(gmp_num, 8 * sizeof(Mantissa));
    REQUIRE(mpf_get_prec(gmp_num) >= 8 * sizeof(Mantissa));
    mpf_set_si(gmp_num, -42);
    REQUIRE(gmp_num->_mp_exp == 1);
    REQUIRE(gmp_num->_mp_size == -1);
    REQUIRE(gmp_num->_mp_d[0] == 42);
    PackedFloat a(gmp_num);
    mpf_clear(gmp_num);
    REQUIRE(a.sign == 1);
    REQUIRE(a.exponent == 1);
    REQUIRE(a.mantissa[0] == 42);
    for (int i = 1; i < kMantissaBytes; ++i) {
        REQUIRE(a.mantissa[i] == 0);
    }
    *gmp_num = a.ToGmp();
    REQUIRE(gmp_num->_mp_exp == 1);
    REQUIRE(gmp_num->_mp_size == -int(kBytes / sizeof(Limb)));
    REQUIRE(gmp_num->_mp_d[0] == 42);
    for (int i = 1; i < gmp_num->_mp_size; ++i) {
        REQUIRE(gmp_num->_mp_d[i] == 0);
    }
}

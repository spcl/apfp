#include <ap_int.h>

#include <catch.hpp>
#include <iostream>
#include <limits>

#include "Karatsuba.h"
#include "PackedFloat.h"

TEST_CASE("PackedFloat to/from GMP Conversion") {
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

template <int bits>
ap_uint<2 * bits> MultOverflow(ap_uint<bits> const &a, ap_uint<bits> const &b) {
    return ap_uint<2 * bits>(a) * ap_uint<2 * bits>(b);
}

TEST_CASE("Karatsuba") {
    ap_uint<kBits> a, b;
    a = 1;
    b = 1;
    REQUIRE(MultOverflow(a, b) == Karatsuba(a, b));
    a = 0;
    b = 1;
    REQUIRE(MultOverflow(a, b) == Karatsuba(a, b));
    a = 0;
    b = 0;
    REQUIRE(MultOverflow(a, b) == Karatsuba(a, b));
    a = -1;
    b = 1;
    REQUIRE(MultOverflow(a, b) == Karatsuba(a, b));
    a = 12345;
    b = 67890;
    REQUIRE(MultOverflow(a, b) == Karatsuba(a, b));
    a = 1234567890;
    b = 6789012345;
    REQUIRE(MultOverflow(a, b) == Karatsuba(a, b));
    a = std::numeric_limits<uint64_t>::max();
    b = std::numeric_limits<uint64_t>::max();
    REQUIRE(MultOverflow(a, b) == Karatsuba(a, b));
}

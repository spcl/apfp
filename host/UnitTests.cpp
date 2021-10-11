#include <ap_int.h>

#include <catch.hpp>
#include <iostream>
#include <limits>

#include "ArithmeticOperations.h"
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
    PackedFloat num(gmp_num);
    REQUIRE(num.sign == 1);
    REQUIRE(num.exponent == 1);
    REQUIRE(num.mantissa[0] == 42);
    for (int i = 1; i < kMantissaBytes; ++i) {
        REQUIRE(num.mantissa[i] == 0);
    }
    num.ToGmp(gmp_num);
    REQUIRE(gmp_num->_mp_exp == 1);
    REQUIRE(gmp_num->_mp_size == -1);
    REQUIRE(gmp_num->_mp_d[0] == 42);
    for (int i = 1; i < gmp_num->_mp_size; ++i) {
        REQUIRE(gmp_num->_mp_d[i] == 0);
    };
    mpf_clear(gmp_num);
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

TEST_CASE("Multiply") {
    mpf_t gmp_a, gmp_b, gmp_c;
    mpf_init2(gmp_a, kMantissaBits);
    mpf_init2(gmp_b, kMantissaBits);
    mpf_init2(gmp_c, kMantissaBits);

    auto multiply = [&gmp_a, &gmp_b, &gmp_c](int64_t a, int64_t b) {
        mpf_set_si(gmp_a, a);
        mpf_set_si(gmp_b, b);
        mpf_mul(gmp_c, gmp_a, gmp_b);
    };

    multiply(1, 0);
    REQUIRE(Multiply(PackedFloat(gmp_a), PackedFloat(gmp_b)) == PackedFloat(gmp_c));
    multiply(1, 1);
    REQUIRE(Multiply(PackedFloat(gmp_a), PackedFloat(gmp_b)) == PackedFloat(gmp_c));
    multiply(std::numeric_limits<int64_t>::max(), 1);
    REQUIRE(Multiply(PackedFloat(gmp_a), PackedFloat(gmp_b)) == PackedFloat(gmp_c));
    multiply(std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max());
    REQUIRE(Multiply(PackedFloat(gmp_a), PackedFloat(gmp_b)) == PackedFloat(gmp_c));

    mpf_clear(gmp_a);
    mpf_clear(gmp_b);
    mpf_clear(gmp_c);
}

TEST_CASE("Add") {
    mpf_t gmp_a, gmp_b, gmp_c;
    mpf_init2(gmp_a, kMantissaBits);
    mpf_init2(gmp_b, kMantissaBits);
    mpf_init2(gmp_c, kMantissaBits);

    auto add = [&gmp_a, &gmp_b, &gmp_c](int64_t a, int64_t b) {
        mpf_set_si(gmp_a, a);
        mpf_set_si(gmp_b, b);
        mpf_add(gmp_c, gmp_a, gmp_b);
    };

    add(1, 0);
    REQUIRE(Add(PackedFloat(gmp_a), PackedFloat(gmp_b)) == PackedFloat(gmp_c));
    add(1, 1);
    REQUIRE(Add(PackedFloat(gmp_a), PackedFloat(gmp_b)) == PackedFloat(gmp_c));
    add(std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max());
    REQUIRE(Add(PackedFloat(gmp_a), PackedFloat(gmp_b)) == PackedFloat(gmp_c));
    add(std::numeric_limits<int64_t>::max(), 1);
    REQUIRE(Add(PackedFloat(gmp_a), PackedFloat(gmp_b)) == PackedFloat(gmp_c));

    mpf_clear(gmp_a);
    mpf_clear(gmp_b);
    mpf_clear(gmp_c);
}

TEST_CASE("MultiplyAccumulate") {
    mpf_t gmp_a, gmp_b, gmp_c, gmp_tmp;
    mpf_init2(gmp_a, kMantissaBits);
    mpf_init2(gmp_b, kMantissaBits);
    mpf_init2(gmp_c, kMantissaBits);
    mpf_init2(gmp_tmp, kMantissaBits);

    auto multiply_accumulate = [&gmp_a, &gmp_b, &gmp_c, &gmp_tmp](int64_t a, int64_t b, int64_t c) {
        mpf_set_si(gmp_a, a);
        mpf_set_si(gmp_b, b);
        mpf_set_si(gmp_c, c);
        mpf_mul(gmp_tmp, gmp_a, gmp_b);
        mpf_add(gmp_tmp, gmp_tmp, gmp_c);
    };

    multiply_accumulate(1, 1, 0);
    REQUIRE(MultiplyAccumulate(PackedFloat(gmp_a), PackedFloat(gmp_b), PackedFloat(gmp_c)) == PackedFloat(gmp_tmp));
    multiply_accumulate(1, 2, 0);
    REQUIRE(MultiplyAccumulate(PackedFloat(gmp_a), PackedFloat(gmp_b), PackedFloat(gmp_c)) == PackedFloat(gmp_tmp));
    multiply_accumulate(1, 1, 1);
    REQUIRE(MultiplyAccumulate(PackedFloat(gmp_a), PackedFloat(gmp_b), PackedFloat(gmp_c)) == PackedFloat(gmp_tmp));
    multiply_accumulate(1, 2, 1);
    REQUIRE(MultiplyAccumulate(PackedFloat(gmp_a), PackedFloat(gmp_b), PackedFloat(gmp_c)) == PackedFloat(gmp_tmp));
    multiply_accumulate(std::numeric_limits<int64_t>::max(), 1, 0);
    REQUIRE(MultiplyAccumulate(PackedFloat(gmp_a), PackedFloat(gmp_b), PackedFloat(gmp_c)) == PackedFloat(gmp_tmp));
    multiply_accumulate(std::numeric_limits<int64_t>::max(), 1, std::numeric_limits<int64_t>::max());
    REQUIRE(MultiplyAccumulate(PackedFloat(gmp_a), PackedFloat(gmp_b), PackedFloat(gmp_c)) == PackedFloat(gmp_tmp));

    multiply_accumulate(std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(), 0);
    REQUIRE(MultiplyAccumulate(PackedFloat(gmp_a), PackedFloat(gmp_b), PackedFloat(gmp_c)) == PackedFloat(gmp_tmp));
    multiply_accumulate(std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(),
                        std::numeric_limits<int64_t>::max());
    REQUIRE(MultiplyAccumulate(PackedFloat(gmp_a), PackedFloat(gmp_b), PackedFloat(gmp_c)) == PackedFloat(gmp_tmp));

    mpf_clear(gmp_a);
    mpf_clear(gmp_b);
    mpf_clear(gmp_c);
    mpf_clear(gmp_tmp);
}

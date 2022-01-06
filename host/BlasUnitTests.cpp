#include <catch.hpp>
#include <iostream>
#include <limits>

#include "Config.h"

// #include "ArithmeticOperations.h"
// #include "Karatsuba.h"
// #include "PackedFloat.h"
#include "ApfpBlas.h"
#include "Random.h"

void ApfpSetup() {
#ifdef APFP_GMP_INTERFACE_TYPE
    mpf_set_default_prec(kMantissaBits);
#else
    mpfr_set_default_prec(kMantissaBits);
#endif
    auto apfp_error_code = apfp::Init(kMantissaBits);
    REQUIRE(apfp_error_code);
}

void ApfpTeardown() {
    apfp::Finalize();
}

bool IsZero(apfp::interface::ConstPtr a) {
#ifdef APFP_GMP_INTERFACE_TYPE
    return mpf_sgn(a) == 0;
#else
    return mpfr_sgn(a) == 0;
#endif
}

bool IsClose(apfp::interface::ConstPtr a, apfp::interface::ConstPtr b) {
    // Avoids divide by zero if a = b = 0
    if (IsZero(a) && IsZero(b)) {
        return true;
    }

    apfp::interface::Wrapper diff, sum, ratio;
#ifdef APFP_GMP_INTERFACE_TYPE
    mpf_sub(diff.get(), a, b);
    mpf_add(sum.get(), a, b);
    mpf_div(ratio.get(), diff.get(), sum.get());
    long exp;
    mpf_get_d_2exp(&exp, ratio.get());
#else
    mpfr_sub(diff.get(), a, b, kRoundingMode);
    mpfr_add(sum.get(), a, b, kRoundingMode);
    mpfr_div(ratio.get(), diff.get(), sum.get(), kRoundingMode);
    auto exp = mpfr_get_exp(ratio.get());
#endif
    // Require the numbers to match to the first 90% decimal places
    return exp < -((kMantissaBits * 3 * 9) / 10);
}

TEST_CASE("Init_Teardown") {
    ApfpSetup();
    ApfpTeardown();
}

TEST_CASE("SYRK") {
    ApfpSetup();

    auto rng = RandomNumberGenerator();

    unsigned long N = GENERATE(0, 1, 2, 8, 15, 16, 31, 32, 33);
    unsigned long K = GENERATE(0, 1, 2, 8, 15, 16, 31, 32, 33);
    auto mode = GENERATE(apfp::BlasTrans::normal, apfp::BlasTrans::transpose);
    auto uplo_mode = GENERATE(apfp::BlasUplo::upper, apfp::BlasUplo::lower);
    // Test SYRK
    // In 'N' mode, we perform AA^T + C
    // A is NxK (A : R^K -> R^N)
    // C is NxN
    // Matrices are stored column major because BLAS
    {
        std::vector<apfp::interface::Wrapper> a_matrix;
        a_matrix.resize(N * K);
        for (auto& v : a_matrix) {
            rng.Generate(v.get());
        }

        std::vector<apfp::interface::Wrapper> c_matrix;
        c_matrix.resize(N * N);
        for (auto& v : c_matrix) {
            rng.Generate(v.get());
        }

        std::vector<apfp::interface::Wrapper> ref_result;
        ref_result.resize(N * N);

        // Compute reference result
        apfp::interface::Wrapper prod_temp;
        for (unsigned long j = 0; j < N; ++j) {
            // lower half
            for (unsigned long i = 0; i < N; ++i) {
                auto r_idx = i + j * N;
                apfp::interface::Set(ref_result.at(r_idx).get(), c_matrix.at(r_idx).get());

                for (unsigned long k = 0; k < K; ++k) {
                    // A is NxK if N, KxN if T
                    if (mode == apfp::BlasTrans::normal) {
                        // (AB)_ij = sum_k A(i,k)B(k,j)
                        apfp::interface::Mul(prod_temp.get(), a_matrix.at(i + k * N).get(),
                                             a_matrix.at(j + k * N).get());
                    } else {
                        // (AB)_ij = sum_k A(i,k) B(k,j)
                        apfp::interface::Mul(prod_temp.get(), a_matrix.at(k + i * K).get(),
                                             a_matrix.at(k + j * K).get());
                    }
                    apfp::interface::Add(ref_result.at(r_idx).get(), prod_temp.get(), ref_result.at(r_idx).get());
                }
            }
        }

        // Use APFP BLAS library
        auto error_code = apfp::Syrk(
            uplo_mode, mode, N, K, [&](unsigned long i) { return a_matrix.at(i).get(); },
            mode == apfp::BlasTrans::normal ? N : K, [&](unsigned long i) { return c_matrix.at(i).get(); }, N);
        REQUIRE(error_code);

        // Check all entries are sufficiently close
        apfp::interface::Wrapper diff;
        for (unsigned long j = 0; j < N; ++j) {
            // lower half
            for (unsigned long i = 0; i < j; ++i) {
                auto ref_value = uplo_mode == apfp::BlasUplo::lower ? ref_result.at(i + j * N).get()
                                                                    : ref_result.at(j + i * N).get();
                auto test_value =
                    uplo_mode == apfp::BlasUplo::lower ? c_matrix.at(i + j * N).get() : c_matrix.at(j + i * N).get();
                REQUIRE(IsClose(ref_value, test_value));
            }
        }
    }

    ApfpTeardown();
}
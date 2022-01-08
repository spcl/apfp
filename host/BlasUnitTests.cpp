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
    REQUIRE(!apfp_error_code);
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
    // Also demand double precision downcast exactly equal
    return (exp < -((kMantissaBits * 3 * 9) / 10)) && (mpfr_get_d(a, kRoundingMode) == mpfr_get_d(b, kRoundingMode));
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

    auto mode = GENERATE(apfp::BlasTrans::transpose, apfp::BlasTrans::normal);
    auto uplo_mode = GENERATE(apfp::BlasUplo::upper, apfp::BlasUplo::lower);
    // Test SYRK
    // In 'N' mode, we perform AA^T + C
    // A is NxK (A : R^K -> R^N)
    // C is NxN
    // Matrices are stored column major because BLAS
    {
        CAPTURE(N, K, mode, uplo_mode);
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

        // Capture inputs for when we explode
        std::vector<double> a_matrix_d, c_matrix_d;
        a_matrix_d.resize(a_matrix.size());
        std::transform(a_matrix.begin(), a_matrix.end(), a_matrix_d.begin(),
                       [](const auto& v) { return mpfr_get_d(v.get(), kRoundingMode); });
        c_matrix_d.resize(c_matrix.size());
        std::transform(c_matrix.begin(), c_matrix.end(), c_matrix_d.begin(),
                       [](const auto& v) { return mpfr_get_d(v.get(), kRoundingMode); });
        CAPTURE(a_matrix_d, c_matrix_d);

        // Compute reference result
        apfp::interface::Wrapper prod_temp;
        for (unsigned long j = 0; j < N; ++j) {
            // lower half
            for (unsigned long i = 0; i < N; ++i) {
                auto r_idx = mode == apfp::BlasTrans::normal ? i + j * N : j + i * N;
                apfp::interface::Set(ref_result.at(r_idx).get(), c_matrix.at(r_idx).get());

                for (unsigned long k = 0; k < K; ++k) {
                    // (AB)_ij = sum_k A(i,k)B(k,j)
                    apfp::interface::Mul(prod_temp.get(), a_matrix.at(i + k * N).get(), a_matrix.at(j + k * N).get());
                    apfp::interface::Add(ref_result.at(r_idx).get(), prod_temp.get(), ref_result.at(r_idx).get());
                }
            }
        }

        // Use APFP BLAS library
        auto error_code = apfp::Syrk(
            uplo_mode, mode, N, K, [&](unsigned long i) { return a_matrix.at(i).get(); },
            mode == apfp::BlasTrans::normal ? N : K, [&](unsigned long i) { return c_matrix.at(i).get(); }, N);
        REQUIRE(!error_code);

        std::vector<double> c_matrix_result_d, c_matrix_ref_result_d;
        c_matrix_result_d.resize(c_matrix.size());
        c_matrix_ref_result_d.resize(c_matrix.size());
        std::transform(c_matrix.begin(), c_matrix.end(), c_matrix_result_d.begin(),
                       [](const auto& v) { return mpfr_get_d(v.get(), kRoundingMode); });
        std::transform(ref_result.begin(), ref_result.end(), c_matrix_ref_result_d.begin(),
                       [](const auto& v) { return mpfr_get_d(v.get(), kRoundingMode); });

        CAPTURE(c_matrix_result_d, c_matrix_ref_result_d);

        // Check all entries are sufficiently close
        apfp::interface::Wrapper diff;
        for (unsigned long j = 0; j < N; ++j) {
            // upper half
            for (unsigned long i = 0; i < j; ++i) {
                auto ref_value = uplo_mode == apfp::BlasUplo::upper ? ref_result.at(i + j * N).get()
                                                                    : ref_result.at(j + i * N).get();
                auto test_value =
                    uplo_mode == apfp::BlasUplo::upper ? c_matrix.at(i + j * N).get() : c_matrix.at(j + i * N).get();
                CAPTURE(i, j);
                CAPTURE(PackedFloat(ref_value), PackedFloat(test_value));
                CAPTURE(mpfr_get_d(ref_value, kRoundingMode), mpfr_get_d(test_value, kRoundingMode));
                REQUIRE(IsClose(ref_value, test_value));
            }
        }
    }

    ApfpTeardown();
}

TEST_CASE("GEMM") {
    ApfpSetup();

    auto rng = RandomNumberGenerator();

    unsigned long M = GENERATE(0, 1, 2, 8, 15, 16, 31, 32, 33);
    unsigned long N = GENERATE(0, 1, 2, 8, 15, 16, 31, 32, 33);
    unsigned long K = GENERATE(0, 1, 2, 8, 15, 16, 31, 32, 33);

    {
        CAPTURE(M, N, K);
        std::vector<apfp::interface::Wrapper> a_matrix;
        a_matrix.resize(M * K);
        for (auto& v : a_matrix) {
            rng.Generate(v.get());
        }

        std::vector<apfp::interface::Wrapper> b_matrix;
        b_matrix.resize(K * N);
        for (auto& v : b_matrix) {
            rng.Generate(v.get());
        }

        std::vector<apfp::interface::Wrapper> c_matrix;
        c_matrix.resize(M * N);
        for (auto& v : a_matrix) {
            rng.Generate(v.get());
        }

        std::vector<apfp::interface::Wrapper> ref_result;
        ref_result.resize(M * N);

        // Capture inputs for when we explode
        std::vector<double> a_matrix_d, c_matrix_d;
        a_matrix_d.resize(a_matrix.size());
        std::transform(a_matrix.begin(), a_matrix.end(), a_matrix_d.begin(),
                       [](const auto& v) { return mpfr_get_d(v.get(), kRoundingMode); });
        c_matrix_d.resize(c_matrix.size());
        std::transform(c_matrix.begin(), c_matrix.end(), c_matrix_d.begin(),
                       [](const auto& v) { return mpfr_get_d(v.get(), kRoundingMode); });
        CAPTURE(a_matrix_d, c_matrix_d);

        // Compute reference result
        apfp::interface::Wrapper prod_temp;
        for (unsigned long j = 0; j < N; ++j) {
            // lower half
            for (unsigned long i = 0; i < M; ++i) {
                auto r_idx = i + j * M;
                apfp::interface::Set(ref_result.at(r_idx).get(), c_matrix.at(r_idx).get());

                for (unsigned long k = 0; k < K; ++k) {
                    // (AB)_ij = sum_k A(i,k)B(k,j)
                    apfp::interface::Mul(prod_temp.get(), a_matrix.at(i + k * M).get(), b_matrix.at(k + j * K).get());
                    apfp::interface::Add(ref_result.at(r_idx).get(), prod_temp.get(), ref_result.at(r_idx).get());
                }
            }
        }

        // Use APFP BLAS library
        auto error_code = apfp::Gemm(
            apfp::BlasTrans::normal, apfp::BlasTrans::normal, M, N, K,
            [&](unsigned long i) { return a_matrix.at(i).get(); }, M,
            [&](unsigned long i) { return b_matrix.at(i).get(); }, K,
            [&](unsigned long i) { return c_matrix.at(i).get(); }, M);
        REQUIRE(!error_code);

        std::vector<double> c_matrix_result_d, c_matrix_ref_result_d;
        c_matrix_result_d.resize(c_matrix.size());
        c_matrix_ref_result_d.resize(c_matrix.size());
        std::transform(c_matrix.begin(), c_matrix.end(), c_matrix_result_d.begin(),
                       [](const auto& v) { return mpfr_get_d(v.get(), kRoundingMode); });
        std::transform(ref_result.begin(), ref_result.end(), c_matrix_ref_result_d.begin(),
                       [](const auto& v) { return mpfr_get_d(v.get(), kRoundingMode); });

        CAPTURE(c_matrix_result_d, c_matrix_ref_result_d);

        // Check all entries are sufficiently close
        apfp::interface::Wrapper diff;
        for (unsigned long j = 0; j < N; ++j) {
            // upper half
            for (unsigned long i = 0; i < M; ++i) {
                auto ref_value = ref_result.at(i + j * M).get();
                auto test_value = c_matrix.at(i + j * M).get();
                CAPTURE(i, j);
                CAPTURE(PackedFloat(ref_value), PackedFloat(test_value));
                CAPTURE(mpfr_get_d(ref_value, kRoundingMode), mpfr_get_d(test_value, kRoundingMode));
                REQUIRE(IsClose(ref_value, test_value));
            }
        }
    }

    ApfpTeardown();
}
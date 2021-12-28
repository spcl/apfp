#include "Config.h"
#include <catch.hpp>
#include <iostream>
#include <limits>

// #include "ArithmeticOperations.h"
// #include "Karatsuba.h"
// #include "PackedFloat.h"
#include "Random.h"

#include "ApfpBlas.h"

void ApfpSetup() {
#ifdef APFP_GMP_INTERFACE_TYPE
    mpf_set_default_prec(kMantissaBits);
#else
    mpfr_set_default_prec(kMantissaBits);
#endif
    auto apfp_error_code = ApfpInit(kMantissaBits);
    REQUIRE(apfp_error_code == ApfpBlasError::success);
}

void ApfpTeardown() {
    ApfpFinalize();
}

bool IsZero(ApfpInterfaceTypeConstPtr a) {
#ifdef APFP_GMP_INTERFACE_TYPE 
    return mpf_sgn(a) == 0;
#else
    return mpfr_sgn(a) == 0;
#endif
}

bool IsClose(ApfpInterfaceTypeConstPtr a, ApfpInterfaceTypeConstPtr b) {
    // Avoids divide by zero if a = b = 0
    if(IsZero(a) && IsZero(b)) {
        return true;
    }

    ApfpInterfaceWrapper diff, sum, ratio;
#ifdef APFP_GMP_INTERFACE_TYPE
    mpf_sub(diff.get(), a, b);
    mpf_add(sum.get(), a, b);
    mpf_div(ratio.get(), diff.get(), sum.get());
    long exp;
    mpf_get_d_2exp(&exp, ratio.get());
#else
    auto rounding_mode = mpfr_get_default_rounding_mode();
    mpfr_sub(diff.get(), a, b, rounding_mode);
    mpfr_add(sum.get(), a, b, rounding_mode);
    mpfr_div(ratio.get(), diff.get(), sum.get(), rounding_mode);
    auto exp = mpfr_get_exp(ratio.get());
#endif
    // Require the numbers to match to the first 90% decimal places
    return exp < -((kMantissaBits*3 * 9)/10);
}

TEST_CASE("Init_Teardown") {
    ApfpSetup();
    ApfpTeardown();
}

TEST_CASE("SYRK") {
    ApfpSetup();

    auto rng = RandomNumberGenerator();

    unsigned long N = GENERATE(0, 1, 8, 15, 16, 31, 32, 33);
    unsigned long K = GENERATE(0, 1, 8, 15, 16, 31, 32, 33);
    char mode = GENERATE('N', 'T');
    // Test SYRK
    // In 'N' mode, we perform AA^T + C
    // A is NxK (A : R^K -> R^N)
    // C is NxN
    // Matrices are stored column major because BLAS
    {
        std::vector<ApfpInterfaceWrapper> a_matrix;
        a_matrix.resize(N*K);
        for(auto& v : a_matrix) {
            rng.Generate(v.get());
        }

        std::vector<ApfpInterfaceWrapper> c_matrix;
        c_matrix.resize(N*N);
        for(auto& v : c_matrix) {
            rng.Generate(v.get());
        }

        std::vector<ApfpInterfaceWrapper> ref_result;
        ref_result.resize(N*N);

        // Compute reference result
        ApfpInterfaceWrapper prod_temp, sum_temp;
        for(unsigned long j = 0; j < N; ++j) {
            // lower half
            for(unsigned long i = 0; i < j; ++i) {
                auto r_idx = i + j*N;
                SetApfpInterfaceType(ref_result.at(r_idx).get(), c_matrix.at(r_idx).get());
                
                for(unsigned long k = 0; k < K; ++k) {
                    // A is NxK if N, KxN if T
                    if (mode == 'N') {
                        // (AB)_ij = sum_k A(i,k)B(k,j)
                        MulApfpInterfaceType(prod_temp.get(), a_matrix.at(i + k*N).get(), a_matrix.at(j + k*N).get());
                    } else {
                        // (AB)_ij = sum_k A(i,k) B(k,j)
                        MulApfpInterfaceType(prod_temp.get(), a_matrix.at(k + i*K).get(), a_matrix.at(k + j*K).get());
                    }
                    AddApfpInterfaceType(sum_temp.get(), prod_temp.get(), ref_result.at(r_idx).get());
                    SetApfpInterfaceType(ref_result.at(r_idx).get(), sum_temp.get());
                }
            }
        }

        // Use APFP BLAS library
        auto error_code = ApfpSyrk('L', mode, N, K, 
            [&](unsigned long i) { return a_matrix.at(i).get(); }, mode == 'N' ? N : K,  
            [&](unsigned long i) { return c_matrix.at(i).get(); }, N);
        REQUIRE(error_code == ApfpBlasError::success);

        // Check all entries are sufficiently close
        ApfpInterfaceWrapper diff;
        for(unsigned long j = 0; j < N; ++j) {
            // lower half
            for(unsigned long i = 0; i < j; ++i) {
                REQUIRE(IsClose(ref_result.at(i + j*N).get(), c_matrix.at(i + j*N).get()));
            }
        }
    }

    ApfpTeardown();
}
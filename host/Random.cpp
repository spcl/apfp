#include "Random.h"

RandomNumberGenerator::RandomNumberGenerator() {
    small_rng_.seed(std::random_device()());
    gmp_randinit_default(state_);
}

RandomNumberGenerator::~RandomNumberGenerator() {
    gmp_randclear(state_);
}

PackedFloat RandomNumberGenerator::Generate() {
    mpfr_t mpfr_num;
    GenerateMpfr(mpfr_num);
    PackedFloat num(mpfr_num);
    mpfr_clear(mpfr_num);
    return num;
}

void RandomNumberGenerator::GenerateGmp(mpf_ptr num) {
    mpf_init2(num, kMantissaBits);
    Generate(num);
}

void RandomNumberGenerator::GenerateMpfr(mpfr_ptr num) {
    mpfr_init2(num, kMantissaBits);
    Generate(num);
}

void RandomNumberGenerator::Generate(mpfr_ptr num) {
    std::unique_lock<std::mutex> lock(mutex_);
    mpfr_urandom(num, state_, kRoundingMode);
    
    // Set exponent
    int exp = exp_distr_(small_rng_);
    if (u01_distr_(small_rng_) < 0.5) {
        exp = -exp;
    }
    mpfr_set_exp(num, exp);

    if(u01_distr_(small_rng_) < zero_frac_) {
        mpfr_set_ui(num, 0, kRoundingMode);
    }

    if(u01_distr_(small_rng_) < one_frac_) {
        mpfr_set_ui(num, 1, kRoundingMode);
    }

    // randomly flip sign bit
    mpfr_setsign(num, num, (u01_distr_(small_rng_) < neg_frac_ ? 1 : 0), kRoundingMode);
}

void RandomNumberGenerator::Generate(mpf_ptr num) {
    std::unique_lock<std::mutex> lock(mutex_);
    mpf_urandomb(num, state_, kMantissaBits);
}

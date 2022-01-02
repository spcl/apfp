#include "Random.h"

RandomNumberGenerator::RandomNumberGenerator() {
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
}

void RandomNumberGenerator::Generate(mpf_ptr num) {
    std::unique_lock<std::mutex> lock(mutex_);
    mpf_urandomb(num, state_, kMantissaBits);
}

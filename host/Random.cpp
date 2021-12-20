#include "Random.h"

RandomNumberGenerator::RandomNumberGenerator() {
    gmp_randinit_default(state_);
}

RandomNumberGenerator::~RandomNumberGenerator() {
    gmp_randclear(state_);
}

PackedFloat RandomNumberGenerator::Generate() {
    mpfr_t mpfr_num = {GenerateMpfr()};
    PackedFloat num(mpfr_num);
    mpfr_clear(mpfr_num);
    return num;
}

__mpf_struct RandomNumberGenerator::GenerateGmp() {
    mpf_t num;
    mpf_init2(num, kMantissaBits);
    Generate(num);
    return num[0];
}

__mpfr_struct RandomNumberGenerator::GenerateMpfr() {
    mpfr_t num;
    mpfr_init2(num, kBits);
    Generate(num);
    return num[0];
}

void RandomNumberGenerator::Generate(mpfr_ptr num) {
    std::unique_lock<std::mutex> lock(mutex_);
    mpfr_urandom(num, state_, kRoundingMode);
}

void RandomNumberGenerator::Generate(mpf_ptr num) {
    std::unique_lock<std::mutex> lock(mutex_);
    mpf_urandomb(num, state_, kMantissaBits);
}

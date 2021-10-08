#include "Random.h"

RandomNumberGenerator::RandomNumberGenerator() {
    gmp_randinit_default(state_);
}

RandomNumberGenerator::~RandomNumberGenerator() {
    gmp_randclear(state_);
}

PackedFloat RandomNumberGenerator::Generate() {
    mpf_t gmp_num = {GenerateGmp()};
    PackedFloat num(gmp_num);
    mpf_clear(gmp_num);
    return num;
}

__mpf_struct RandomNumberGenerator::GenerateGmp() {
    std::unique_lock<std::mutex> lock(mutex_);
    mpf_t num;
    mpf_init2(num, kMantissaBits);
    mpf_urandomb(num, state_, kMantissaBits);
    return num[0];
}

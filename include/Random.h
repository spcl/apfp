#include <gmp.h>

#include <mutex>
#include <random>

#include "Config.h"
#include "PackedFloat.h"

class RandomNumberGenerator {
   public:
    RandomNumberGenerator();

    RandomNumberGenerator(RandomNumberGenerator const &) = delete;
    RandomNumberGenerator(RandomNumberGenerator &&) = delete;
    RandomNumberGenerator &operator=(RandomNumberGenerator const &) = delete;
    RandomNumberGenerator &operator=(RandomNumberGenerator &&) = delete;

    ~RandomNumberGenerator();

    /// Generate a random number using GMP, then package it as a PackedFloat before returning it.
    PackedFloat Generate();

    /// Generate a random GMP number.
    void GenerateGmp(mpf_ptr);

    /// Generate a random GMP number into the specified output variable.
    void Generate(mpf_ptr);

    /// Generate a random MPFR number.
    void GenerateMpfr(mpfr_ptr);

    /// Generate a random MPFR into the specified output variable.
    void Generate(mpfr_ptr);

   private:
    std::mt19937_64 small_rng_;
    static constexpr double neg_frac_ = 1.0/3.0;
    std::poisson_distribution<> exp_distr_;

    gmp_randstate_t state_;
    std::mutex mutex_;
};

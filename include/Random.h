#include <gmp.h>

#include <mutex>

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
    __mpf_struct GenerateGmp();

    /// Generate a random GMP number into the specified output variable.
    void Generate(mpf_t &);

    /// Generate a random MPFR number.
    __mpfr_struct GenerateMpfr();

    /// Generate a random MPFR into the specified output variable.
    void Generate(mpfr_t &);

   private:
    gmp_randstate_t state_;
    std::mutex mutex_;
};

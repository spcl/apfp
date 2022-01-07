#pragma once
#include <gmp.h>
#include <mpfr.h>

#include "Config.h"

/* This header abstracts away the choice of MPFR or GMP in the interface
 * It defines four types: Value, Ptr, ConstPtr, Wrapper
 * The first three directly correspond to MPFR/GMP types
 * The last one is a wrapper that manages the memory footprint with RAII
 */
namespace apfp::interface {

#ifdef APFP_GMP_INTERFACE_TYPE  // Interface with GMP types
using Value = mpf_t;
using Ptr = mpf_ptr;
using ConstPtr = mpf_srcptr;
#else
#include <mpfr.h>
using Value = mpfr_t;
using Ptr = mpfr_ptr;
using ConstPtr = mpfr_srcptr;
#endif

void Init(Ptr value);

void Init2(Ptr value, unsigned long precision);

void Clear(Ptr value);

void Swap(Ptr a, Ptr b);

void Set(Ptr dest, ConstPtr source);

void Set(Ptr dest, long int source);

void Add(Ptr dest, ConstPtr a, ConstPtr b);

void Mul(Ptr dest, ConstPtr a, ConstPtr b);

/// Smart pointer-like wrapper class for GMP/MPFR types
class Wrapper {
    Value data_;

   public:
    ~Wrapper();

    Wrapper();

    Wrapper(unsigned long precision);

    Wrapper(Wrapper&&);

    Wrapper(Wrapper&) = delete;

    Wrapper& operator=(const Wrapper&) = delete;

    Wrapper& operator=(Wrapper&&);

    // This decays to the pointer type
    Ptr get() {
        return data_;
    }

    ConstPtr get() const {
        return data_;
    }
};

}  // namespace apfp::interface
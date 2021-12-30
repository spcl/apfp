#pragma once
#include <gmp.h> 
#include <functional>
#include "ApfpInterfaceType.h"

namespace apfp {

using IndexFunction = std::function<interface::Ptr(unsigned long)>;
using ConstIndexFunction = std::function<interface::ConstPtr(unsigned long)>;


/// Null terminated string describing the most recent library error if available
/// Pointer is only guaranteed to live until the next library call
const char* ApfpErrorDescription();

int Init(unsigned long precision);

int Finalize();

/// See netlib's documentation on Syrk for usage. Alpha and beta unsupported
int Syrk(char uplo, char trans, unsigned long N, unsigned long K, interface::ConstPtr A, unsigned long LDA, interface::Ptr C, unsigned long LDC);
int Syrk(char uplo, char trans, unsigned long N, unsigned long K, ConstIndexFunction A, unsigned long LDA, IndexFunction C, unsigned long LDC);

enum BlasError : int {
    success             = 0,
    unknown             = 1,
    unimplemented       = 2,
    bitwidth            = 3,
    uninitialized       = 4,
    kernel_not_found    = 5,
};

}
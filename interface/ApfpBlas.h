#pragma once
#include <gmp.h> 
#include <functional>
#include "ApfpInterfaceType.h"

namespace apfp {

using IndexFunction = std::function<ApfpInterfaceTypePtr(unsigned long)>;
using ConstIndexFunction = std::function<ApfpInterfaceTypeConstPtr(unsigned long)>;


/// Null terminated string describing the most recent library error if available
/// Pointer is only guaranteed to live until the next library call
const char* ApfpErrorDescription();

int ApfpInit(unsigned long precision);

int ApfpFinalize();

/// See netlib's documentation on Syrk for usage. Alpha and beta unsupported
int ApfpSyrk(char uplo, char trans, unsigned long N, unsigned long K, ApfpInterfaceTypeConstPtr A, unsigned long LDA, ApfpInterfaceTypePtr C, unsigned long LDC);
int ApfpSyrk(char uplo, char trans, unsigned long N, unsigned long K, ConstIndexFunction A, unsigned long LDA, IndexFunction C, unsigned long LDC);

enum ApfpBlasError : int {
    success             = 0,
    unknown             = 1,
    unimplemented       = 2,
    bitwidth            = 3,
    uninitialized       = 4,
    kernel_not_found    = 5,
};

}
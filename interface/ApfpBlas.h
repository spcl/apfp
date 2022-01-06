#pragma once
#include <gmp.h>

#include <functional>

#include "ApfpInterfaceType.h"

namespace apfp {

enum class BlasError : int {
    success = 0,
    unknown = 1,
    unimplemented = 2,
    bitwidth = 3,
    uninitialized = 4,
    kernel_not_found = 5,
    argument_error = 6,
};

enum class BlasUplo : char { upper = 'U', lower = 'L' };

enum class BlasTrans : char {
    normal = 'N',
    transpose = 'T',
};


using IndexFunction = std::function<interface::Ptr(unsigned long)>;
using ConstIndexFunction = std::function<interface::ConstPtr(unsigned long)>;
/// Null terminated string describing the most recent library error if available
/// Pointer is only guaranteed to live until the next library call
const char* ErrorDescription();

/// Convert a return code to a BlasError type
/// Negative return codes are converted to BlasError::argument_error
BlasError InterpretError(int a);

int Init(unsigned long precision);

int Finalize();

/// See netlib's documentation on Syrk for usage. Alpha and beta unsupported
int Syrk(BlasUplo uplo, BlasTrans trans, unsigned long N, unsigned long K, interface::ConstPtr A, unsigned long LDA,
         interface::Ptr C, unsigned long LDC);
int Syrk(BlasUplo uplo, BlasTrans trans, unsigned long N, unsigned long K, ConstIndexFunction A, unsigned long LDA,
         IndexFunction C, unsigned long LDC);

}  // namespace apfp
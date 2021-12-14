#pragma once
#include <gmp.h> 
#include <functional>
#include "ApfpInterfaceType.h"

// 
using IndexFunction = std::function<ApfpInterfaceTypePtr(unsigned long)>;
using ConstIndexFunction = std::function<ApfpInterfaceTypeConstPtr(unsigned long)>;

int ApfpInit(unsigned long precision);

int ApfpFinalize();

/// See netlib's documentation on Syrk for usage. Alpha and beta unsupported
int ApfpSyrk(char uplo, char trans, unsigned long N, unsigned long K, ApfpInterfaceTypeConstPtr A, unsigned long LDA, ApfpInterfaceTypePtr C, unsigned long LDC);
int ApfpSyrk(char uplo, char trans, unsigned long N, unsigned long K, ConstIndexFunction A, unsigned long LDA, IndexFunction C, unsigned long LDC);

enum ApfpBlasError : int {
    success         = 0,
    unknown         = 1,
    unimplemented   = 2,
    bitwidth        = 3
};



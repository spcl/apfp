#pragma once
#include <gmp.h> 
#include <functional>
#include "ApfpInterfaceType.h"

// 
using IndexFunction = std::function<ApfpInterfaceType*(unsigned long)>;
using ConstIndexFunction = std::function<const ApfpInterfaceType*(unsigned long)>;

int ApfpInit(unsigned int precision);

int ApfpFinalize();

/// See netlib's documentation on Syrk for usage. Alpha and beta unsupported
int ApfpSyrk(char uplo, char trans, unsigned long N, unsigned long K, const ApfpInterfaceType* A, unsigned long LDA, ApfpInterfaceType* C, unsigned long LDC);
int ApfpSyrk(char uplo, char trans, unsigned long N, unsigned long K, ConstIndexFunction A, unsigned long LDA, IndexFunction C, unsigned long LDC);

enum ApfpBlasError : int {
    success         = 0,
    unknown         = 1,
    unimplemented   = 2,
    bitwidth        = 3
};



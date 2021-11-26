#include "ApfpBlas.h"
#include "Apfp.h"
#include "Config.h"
#include <optional>
#include <stdexcept>

static std::optional<Apfp> apfp;

enum ApfpBlasUplo : char {
    upper = 'U',
    lower = 'L'
};

enum ApfpBlasTrans : char {
    normal = 'N',
    transpose = 'T',
};

int ApfpInit(unsigned int precision) {
    try {
        if (precision > kBits) {
            // Requested bit width too large
            return ApfpBlasError::bitwidth;
        }
        apfp.emplace();
        return ApfpBlasError::success;
    }catch(...) {
        // Unknown exception
        return ApfpBlasError::unknown;
    }
}

int ApfpFinalize() {
    apfp.reset();
    return ApfpBlasError::success;
}

/// Copy the upper or lower triangle from an NxN matrix A to a full size buffer
template<typename ptr_function_type>
void CopyFromMatrixUplo(ApfpBlasUplo uplo, unsigned long N, ptr_function_type A, unsigned long LDA, ApfpInterfaceType* buffer) {
    auto dest_lda = N;
    // Col major layout
    for (unsigned long j = 0; j < N; ++j) {
        for (unsigned long i = 0; i <= j; ++i) {
            auto source = uplo == ApfpBlasUplo::lower ? A(i + j * LDA) : A(j + i * LDA);
            buffer[i + j * dest_lda] = *source;
            buffer[j + i * dest_lda] = *source;
        }
    }
}

/// Copy from a full size buffer to the upper or lower triangle of an NxN matrix A
template<typename ptr_function_type>
void CopyToMatrixUplo(ApfpBlasUplo uplo, unsigned long N, ptr_function_type A, unsigned long LDA, ApfpInterfaceType* buffer) {
    auto source_lda = N;
    // Col major layout
    for (unsigned long j = 0; j < N; ++j) {
        for (unsigned long i = 0; i <= j; ++i) {
            auto dest = uplo == ApfpBlasUplo::lower ? A(i + j * LDA) : A(j + i * LDA);
            *dest = buffer[i + j * source_lda];
        }
    }
}

/// Copy from an NxK matrix A to a full size buffer
template<typename ptr_function_type>
void CopyFromMatrix(unsigned long N, unsigned long K, ptr_function_type A, unsigned long LDA, ApfpInterfaceType* buffer) {
    auto dest_lda = N;
    // Col major layout
    for (unsigned long j = 0; j < N; ++j) {
        for (unsigned long i = 0; i < K; ++i) {
            buffer[i + j * dest_lda] = *A(i + j * LDA);
        }
    }
}

/// Copy to an NxK matrix A from a full size buffer
template<typename ptr_function_type>
void CopyToMatrix(unsigned long N, unsigned long K, ptr_function_type A, unsigned long LDA, ApfpInterfaceType* buffer) {
    auto source_lda = N;
    // Col major layout
    for (unsigned long j = 0; j < N; ++j) {
        for (unsigned long i = 0; i < K; ++i) {
            *A(i + j * LDA) = buffer[i + j * source_lda];
        }
    }
}

template<typename ptr_function_type_a, typename ptr_function_type_c>
int ApfpSyrkTemplate(char uplo, char trans, unsigned long N, unsigned long K, ptr_function_type_a A, unsigned long LDA, ptr_function_type_c C, unsigned long LDC) {
    try {
        // ==== library input validation stuff ====f
        if (std::toupper(uplo) != 'U' && std::toupper(uplo) != 'L') { return -1; }
        auto uplo_validated = static_cast<ApfpBlasUplo>(uplo);
        
        if (std::toupper(trans) != 'N' && std::toupper(trans) != 'T') { return -2; }
        // Let's not worry about this mode with N and K being different meanings for now
        if (trans == ApfpBlasTrans::transpose) {
            return ApfpBlasError::unimplemented;
        }

        if (LDA < N) { return -6; }
        if (LDC < N) { return -8; }

        // Empty matrix no-op
        if (N == 0) { return ApfpBlasError::success; }
        if (K == 0) { return ApfpBlasError::success; }
        
        // ==== setup ====

        std::vector<ApfpInterfaceType> host_a, host_c;
        host_a.resize(N*K);
        CopyFromMatrix(N, K, A, LDA, host_a.data());
        auto device_a = apfp->AllocateDeviceMatrix(N, K);
        device_a.TransferToDevice(host_a.data(), host_a.size());

        host_c.resize(N*N);
        CopyFromMatrixUplo(uplo_validated, N, C, LDC, host_c.data());
        auto device_c = apfp->AllocateDeviceMatrix(N, N);
        device_c.TransferToDevice(host_c.data(), host_c.size());

        // ==== compute and teardown ====
        // apfp.MatrixMultiply()

        device_c.TransferToHost(host_c.data(), host_c.size());
        CopyToMatrixUplo(uplo_validated, N, C, LDC, host_c.data());
    } catch (...) {
        return ApfpBlasError::unknown;
    }

    return ApfpBlasError::success;
}

/// See netlib's documentation on Syrk for usage. Alpha and beta unsupported
int ApfpSyrk(char uplo, char trans, unsigned long N, unsigned long K, const ApfpInterfaceType* A, unsigned long LDA, ApfpInterfaceType* C, unsigned long LDC) {
    auto a_ptr_function = [&](unsigned long i) -> const ApfpInterfaceType* { return &(A[i]); };
    auto c_ptr_function = [&](unsigned long i) -> ApfpInterfaceType* { return &(C[i]); };
    return ApfpSyrkTemplate(uplo, trans, N, K, a_ptr_function, LDA, c_ptr_function, LDC);
}

int ApfpSyrk(char uplo, char trans, unsigned long N, unsigned long K, ConstIndexFunction A, unsigned long LDA, IndexFunction C, unsigned long LDC) {
    return ApfpSyrkTemplate(uplo, trans, N, K, A, LDA, C, LDC);
}


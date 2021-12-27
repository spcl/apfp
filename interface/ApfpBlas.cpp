#include "ApfpBlas.h"
#include "Apfp.h"
#include "Config.h"
#include <optional>
#include <stdexcept>

static std::optional<Apfp> apfp;
static std::string last_error_message;

enum ApfpBlasUplo : char {
    upper = 'U',
    lower = 'L'
};

enum ApfpBlasTrans : char {
    normal = 'N',
    transpose = 'T',
};

int ApfpInit(unsigned long precision) {
    try {
        if (precision > kBits) {
            // Requested bit width too large
            last_error_message = "Requested bitwidth too large";
            return ApfpBlasError::bitwidth;
        }
        apfp.emplace();
        return ApfpBlasError::success;
    }catch(const std::exception& e) {
        // Unknown exception
        last_error_message = e.what();
        return ApfpBlasError::unknown;
    }
}

int ApfpFinalize() {
    apfp.reset();
    return ApfpBlasError::success;
}

bool ApfpIsInitialized() {
    return apfp.has_value();
}

const char* ApfpErrorDescription() {
    return last_error_message.c_str();
}

/// Copy the upper or lower triangle from an NxN matrix A to a full size buffer
template<typename ptr_function_type>
void CopyFromMatrixUplo(ApfpBlasUplo uplo, unsigned long N, ptr_function_type A, unsigned long LDA, ApfpInterfaceWrapper* buffer) {
    auto dest_lda = N;
    // Col major layout
    for (unsigned long j = 0; j < N; ++j) {
        for (unsigned long i = 0; i <= j; ++i) {
            auto source = uplo == ApfpBlasUplo::lower ? A(i + j * LDA) : A(j + i * LDA);
            SetApfpInterfaceType(buffer[i + j * dest_lda].get(), source);
            SetApfpInterfaceType(buffer[j + i * dest_lda].get(), source);
        }
    }
}

/// Copy from a full size buffer to the upper or lower triangle of an NxN matrix A
template<typename ptr_function_type>
void CopyToMatrixUplo(ApfpBlasUplo uplo, unsigned long N, ptr_function_type A, unsigned long LDA, ApfpInterfaceWrapper* buffer) {
    auto source_lda = N;
    // Col major layout
    for (unsigned long j = 0; j < N; ++j) {
        for (unsigned long i = 0; i <= j; ++i) {
            auto dest = uplo == ApfpBlasUplo::lower ? A(i + j * LDA) : A(j + i * LDA);
            SetApfpInterfaceType(dest, buffer[i + j * source_lda].get());
        }
    }
}

/// Copy from an NxK matrix A to a full size buffer
template<typename ptr_function_type>
void CopyFromMatrix(unsigned long N, unsigned long K, ptr_function_type A, unsigned long LDA, ApfpInterfaceWrapper* buffer) {
    auto dest_lda = N;
    // Col major layout
    for (unsigned long j = 0; j < K; ++j) {
        for (unsigned long i = 0; i < N; ++i) {
            SetApfpInterfaceType(buffer[i + j * dest_lda].get(), A(i + j * LDA));
        }
    }
}

/// Copy the transpose of a NxK matrix A to a full size buffer
template<typename ptr_function_type>
void CopyTransposeFromMatrix(unsigned long N, unsigned long K, ptr_function_type A, unsigned long LDA, ApfpInterfaceWrapper* buffer) {
    auto dest_lda = K;
    // Col major layout
    for (unsigned long j = 0; j < K; ++j) {
        for (unsigned long i = 0; i < N; ++i) {
            SetApfpInterfaceType(buffer[i * dest_lda + j].get(), A(i + j * LDA));
        }
    }
}

/// Copy to an NxK matrix A from a full size buffer
template<typename ptr_function_type>
void CopyToMatrix(unsigned long N, unsigned long K, ptr_function_type A, unsigned long LDA, ApfpInterfaceWrapper* buffer) {
    auto source_lda = N;
    // Col major layout
    for (unsigned long j = 0; j < K; ++j) {
        for (unsigned long i = 0; i < N; ++i) {
            SetApfpInterfaceType(A(i + j * LDA), buffer[i + j * source_lda].get());
        }
    }
}

template<typename ptr_function_type_a, typename ptr_function_type_c>
int ApfpSyrkImpl(char uplo, char trans, unsigned long N, unsigned long K, ptr_function_type_a A, unsigned long LDA, ptr_function_type_c C, unsigned long LDC) {
    try {
        // ==== library input validation stuff ====
        if(!ApfpIsInitialized()) {
            return ApfpBlasError::uninitialized;
        }

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
        std::vector<ApfpInterfaceWrapper> host_a, host_a_transpose, host_c;
        host_a.resize(N*K);
        CopyFromMatrix(N, K, A, LDA, host_a.data());
        auto device_a = apfp->AllocateDeviceMatrix(N, K);
        device_a.TransferToDevice(host_a.data(), host_a.size());

        host_a_transpose.resize(K*N);
        CopyTransposeFromMatrix(N, K, A, LDA, host_a_transpose.data());
        auto device_a_transpose = apfp->AllocateDeviceMatrix(K, N);
        device_a_transpose.TransferToDevice(host_a_transpose.data(), host_a_transpose.size());

        host_c.resize(N*N);
        CopyFromMatrixUplo(uplo_validated, N, C, LDC, host_c.data());

        // ==== compute and teardown ====
        auto mul_result = apfp->AllocateDeviceMatrix(N, N);
        apfp->MatrixMultiplication(device_a, device_a_transpose, &mul_result);
        std::vector<ApfpInterfaceWrapper> host_result;
        host_result.resize(N*N);

        mul_result.TransferToHost(host_result.data(), host_result.size());

        ApfpInterfaceWrapper add_result;
        for(unsigned long i = 0; i < host_result.size(); ++i) {
            AddApfpInterfaceType(add_result.get(), host_result[i].get(), host_c[i].get());
            SetApfpInterfaceType(host_c[i].get(), add_result.get());
        }

        CopyToMatrixUplo(uplo_validated, N, C, LDC, host_c.data());
    } catch(const std::exception& e) {
        last_error_message = e.what();
        return ApfpBlasError::unknown;
    }

    return ApfpBlasError::success;
}

/// See netlib's documentation on Syrk for usage. Alpha and beta unsupported
int ApfpSyrk(char uplo, char trans, unsigned long N, unsigned long K, ApfpInterfaceTypeConstPtr A, unsigned long LDA, ApfpInterfaceTypePtr C, unsigned long LDC) {
    auto a_ptr_function = [&](unsigned long i) -> ApfpInterfaceTypeConstPtr { return A + i; };
    auto c_ptr_function = [&](unsigned long i) -> ApfpInterfaceTypePtr { return C + i; };
    return ApfpSyrkImpl(uplo, trans, N, K, a_ptr_function, LDA, c_ptr_function, LDC);
}

int ApfpSyrk(char uplo, char trans, unsigned long N, unsigned long K, ConstIndexFunction A, unsigned long LDA, IndexFunction C, unsigned long LDC) {
    return ApfpSyrkImpl(uplo, trans, N, K, A, LDA, C, LDC);
}


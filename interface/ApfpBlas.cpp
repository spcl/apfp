#include "ApfpBlas.h"

#include <optional>
#include <stdexcept>

#include "Apfp.h"
#include "Config.h"

namespace apfp {

static std::optional<Apfp> apfp;
static std::string last_error_message;

int Init(unsigned long precision) {
    try {
        if (precision > kBits) {
            // Requested bit width too large
            last_error_message = "Requested bitwidth too large";
            return static_cast<int>(BlasError::bitwidth);
        }
        apfp.emplace();
        return static_cast<int>(BlasError::success);

    } catch (const KernelNotFoundException& e) {
        last_error_message = e.what();
        return static_cast<int>(BlasError::kernel_not_found);

    } catch (const std::exception& e) {
        // Unknown exception
        last_error_message = e.what();
        return static_cast<int>(BlasError::unknown);
    }
}

int Finalize() {
    apfp.reset();
    return static_cast<int>(BlasError::success);
}

bool IsInitialized() {
    return apfp.has_value();
}

const char* ErrorDescription() {
    return last_error_message.c_str();
}

BlasError InterpretError(int a) {
    return a < 0 ? BlasError::argument_error : static_cast<BlasError>(a);
}

/// Copy the upper or lower triangle from an NxN matrix A to a full size buffer
template <typename ptr_function_type>
void CopyFromMatrixUplo(BlasUplo uplo, unsigned long N, ptr_function_type A, unsigned long LDA,
                        interface::Wrapper* buffer) {
    auto dest_lda = N;
    // Col major layout
    for (unsigned long j = 0; j < N; ++j) {
        for (unsigned long i = 0; i <= j; ++i) {
            auto source = uplo == BlasUplo::lower ? A(i + j * LDA) : A(j + i * LDA);
            interface::Set(buffer[i + j * dest_lda].get(), source);
            interface::Set(buffer[j + i * dest_lda].get(), source);
        }
    }
}

/// Copy from a full size buffer to the upper or lower triangle of an NxN matrix A
template <typename ptr_function_type>
void CopyToMatrixUplo(BlasUplo uplo, unsigned long N, ptr_function_type A, unsigned long LDA,
                      interface::Wrapper* buffer) {
    auto source_lda = N;
    // Col major layout
    for (unsigned long j = 0; j < N; ++j) {
        for (unsigned long i = 0; i <= j; ++i) {
            auto dest = uplo == BlasUplo::lower ? A(i + j * LDA) : A(j + i * LDA);
            interface::Set(dest, buffer[i + j * source_lda].get());
        }
    }
}

/// Copy from an NxK matrix A to a full size buffer
template <typename ptr_function_type>
void CopyFromMatrix(unsigned long N, unsigned long K, ptr_function_type A, unsigned long LDA,
                    interface::Wrapper* buffer) {
    auto dest_lda = N;
    // Col major layout
    for (unsigned long j = 0; j < K; ++j) {
        for (unsigned long i = 0; i < N; ++i) {
            interface::Set(buffer[i + j * dest_lda].get(), A(i + j * LDA));
        }
    }
}

/// Copy the transpose of a NxK matrix A to a full size buffer
template <typename ptr_function_type>
void CopyTransposeFromMatrix(unsigned long N, unsigned long K, ptr_function_type A, unsigned long LDA,
                             interface::Wrapper* buffer) {
    auto dest_lda = K;
    // Col major layout
    for (unsigned long j = 0; j < K; ++j) {
        for (unsigned long i = 0; i < N; ++i) {
            interface::Set(buffer[i * dest_lda + j].get(), A(i + j * LDA));
        }
    }
}

/// Copy to an NxK matrix A from a full size buffer
template <typename ptr_function_type>
void CopyToMatrix(unsigned long N, unsigned long K, ptr_function_type A, unsigned long LDA,
                  interface::Wrapper* buffer) {
    auto source_lda = N;
    // Col major layout
    for (unsigned long j = 0; j < K; ++j) {
        for (unsigned long i = 0; i < N; ++i) {
            interface::Set(A(i + j * LDA), buffer[i + j * source_lda].get());
        }
    }
}

template <typename ptr_function_type_a, typename ptr_function_type_c>
int SyrkImpl(BlasUplo uplo, BlasTrans trans, unsigned long N, unsigned long K, ptr_function_type_a A,
                 unsigned long LDA, ptr_function_type_c C, unsigned long LDC) {
    try {
        // ==== library input validation stuff ====
        if (!IsInitialized()) {
            return static_cast<int>(BlasError::uninitialized);
        }

        // A is NxK if 'N', KxN if 'T'
        // C is always NxN
        // N mode
        // A A^T + C
        // T mode
        // A^T A + C
        bool use_transpose = trans == BlasTrans::transpose;

        unsigned long A_rows = use_transpose ? K : N;
        unsigned long A_cols = use_transpose ? N : K;

        if (LDA < (use_transpose ? K : N)) {
            return -6;
        }
        if (LDC < N) {
            return -8;
        }

        // Empty matrix no-op
        if (N == 0) {
            return static_cast<int>(BlasError::success);
        }
        if (K == 0) {
            return static_cast<int>(BlasError::success);
        }

        // ==== setup ====
        std::vector<interface::Wrapper> host_a, host_a_transpose, host_c;
        host_a.resize(N * K);
        CopyFromMatrix(A_rows, A_cols, A, LDA, host_a.data());
        auto device_a = apfp->AllocateDeviceMatrix(A_rows, A_cols);
        device_a.TransferToDevice(host_a.data(), host_a.size());

        host_a_transpose.resize(K * N);
        CopyTransposeFromMatrix(A_rows, A_cols, A, LDA, host_a_transpose.data());
        auto device_a_transpose = apfp->AllocateDeviceMatrix(A_cols, A_rows);
        device_a_transpose.TransferToDevice(host_a_transpose.data(), host_a_transpose.size());

        host_c.resize(N * N);
        CopyFromMatrixUplo(uplo, N, C, LDC, host_c.data());
        auto device_c = apfp->AllocateDeviceMatrix(N, N);
        device_c.TransferToDevice(host_c.data(), host_c.size());

        // ==== compute and teardown ====
        auto mul_result = apfp->AllocateDeviceMatrix(N, N);
        if (use_transpose) {
            apfp->MatrixMultiplication(device_a_transpose, device_a, &device_c);
        } else {
            apfp->MatrixMultiplication(device_a, device_a_transpose, &device_c);
        }

        device_c.TransferToHost(host_c.data(), host_c.size());
        CopyToMatrixUplo(uplo, N, C, LDC, host_c.data());
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return static_cast<int>(BlasError::unknown);
    }

    return static_cast<int>(BlasError::success);
}

/// See netlib's documentation on Syrk for usage. Alpha and beta unsupported
int Syrk(BlasUplo uplo, BlasTrans trans, unsigned long N, unsigned long K, interface::ConstPtr A, unsigned long LDA,
         interface::Ptr C, unsigned long LDC) {
    auto a_ptr_function = [&](unsigned long i) -> interface::ConstPtr { return A + i; };
    auto c_ptr_function = [&](unsigned long i) -> interface::Ptr { return C + i; };
    return SyrkImpl(uplo, trans, N, K, a_ptr_function, LDA, c_ptr_function, LDC);
}

int Syrk(BlasUplo uplo, BlasTrans trans, unsigned long N, unsigned long K, ConstIndexFunction A, unsigned long LDA,
         IndexFunction C, unsigned long LDC) {
    return SyrkImpl(uplo, trans, N, K, A, LDA, C, LDC);
}

}  // namespace apfp
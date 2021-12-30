#pragma once
#include <gmp.h>
#include <hlslib/xilinx/OpenCL.h>

#include <optional>

#include "MatrixMultiplication.h"
#include "PackedFloat.h"

#include "ApfpInterfaceType.h"

#include <functional>

class DeviceMatrix;

/// Object oriented interface for Apfp
class Apfp {
    hlslib::ocl::Context context_;
    std::optional<hlslib::ocl::Program> program_;

    std::size_t lines_per_number_;
   
    static std::string FindKernel();
   public:
    Apfp();

    /// Allocate a buffer on the device
    DeviceMatrix AllocateDeviceMatrix(std::size_t rows, std::size_t cols);

    /// Two argument matrix multiply allocating the output buffer
    DeviceMatrix MatrixMultiplication(const DeviceMatrix& a, const DeviceMatrix& b);

    /// Three argument matrix multiply with supplied output buffer
    void MatrixMultiplication(const DeviceMatrix& a, const DeviceMatrix& b, DeviceMatrix* result);

    /// Three argument matrix addition with supplied output buffer
    void MatrixAddition(const DeviceMatrix& a, const DeviceMatrix& b, DeviceMatrix* result);

    // Transpose a matrix in place
    void TransposeInPlace(DeviceMatrix* a);

    // Transpose a matrix and allocate a new buffer
    DeviceMatrix Transpose(const DeviceMatrix& a);
};

/// Helper class to track matrices on the device
/// We should probably refactor the interface to Apfp to something more controlled?
class DeviceMatrix {
    std::size_t num_rows_;
    std::size_t num_cols_;
    hlslib::ocl::Buffer<DramLine, hlslib::ocl::Access::readWrite> buffer_;

    friend Apfp;

    DeviceMatrix() = default;

   public:
    std::size_t rows() const {
        return num_rows_;
    }

    std::size_t cols() const {
        return num_cols_;
    }

    /// Transfer from the host to the device
    /// TODO: Make this take input iterators
    void TransferToDevice(ApfpInterfaceTypeConstPtr buffer_ptr, std::size_t buffer_size);
    void TransferToDevice(const ApfpInterfaceWrapper* buffer_ptr, std::size_t buffer_size);


    /// Transfer from the device to the host
    /// TODO: Make this take output iterators
    void TransferToHost(ApfpInterfaceTypePtr buffer_ptr, std::size_t buffer_size);
    void TransferToHost(ApfpInterfaceWrapper* buffer_ptr, std::size_t buffer_size);

   private:
    template<typename ptr_function_type>
    void TransferToDeviceImpl(ptr_function_type buffer_ptr_func, std::size_t buffer_size);

    template<typename ptr_function_type>
    void TransferToHostImpl(ptr_function_type buffer_ptr_func, std::size_t buffer_size);
};

// === Custom exception types ===
struct ApfpException : public std::exception {
    std::string e;
    
    ApfpException() {
        e = "";
    }

    ApfpException(const std::string& what_arg) {
        e = what_arg;
    }

    virtual const char* what() const noexcept {
        return e.c_str();
    }
};

struct KernelNotFoundException : public ApfpException {
    using ApfpException::ApfpException;
};

struct UnimplementedException : public ApfpException {
    using ApfpException::ApfpException;
};

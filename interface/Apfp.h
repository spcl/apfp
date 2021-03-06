#pragma once
#include <gmp.h>
#include <hlslib/xilinx/OpenCL.h>

#include <optional>

#include "MatrixMultiplication.h"
#include "PackedFloat.h"

class DeviceMatrix;

/// Object oriented interface for Apfp
class Apfp {
    hlslib::ocl::Context context_;
    std::optional<hlslib::ocl::Program> program_;

    std::size_t lines_per_number_;
    const std::string kernel_path_ = "";

   public:
    Apfp();

    /// Allocate a buffer on the device
    DeviceMatrix AllocateDeviceMatrix(std::size_t rows, std::size_t cols);

    /// Two argument matrix multiply allocating the output buffer
    DeviceMatrix MatrixMultiplication(const DeviceMatrix& a, const DeviceMatrix& b);

    /// Three argument matrix multiply with supplied output buffer
    void MatrixMultiplication(const DeviceMatrix& a, const DeviceMatrix& b, DeviceMatrix* result);

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
    void TransferToDevice(const mpf_t* buffer_ptr, std::size_t buffer_size);

    /// Transfer from the device to the host
    /// TODO: Make this take output iterators
    void TransferToHost(mpf_t* buffer_ptr, std::size_t buffer_size);
};

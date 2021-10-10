#include "Apfp.hpp"
#include <stdexcept>

#include "Config.h"

Apfp::Apfp() {
    program_.emplace(context_.MakeProgram(kernel_path_));
    lines_per_number_ = kLinesPerNumber; 
}

DeviceMatrix Apfp::AllocateDeviceMatrix(std::size_t rows, std::size_t cols) {
    // This seems like poor encapsulation, is there a better way?

    DeviceMatrix matrix;
    matrix.num_rows_ = rows;
    matrix.num_cols_ = cols;
    matrix.buffer_ = context_.MakeBuffer<DramLine, hlslib::ocl::Access::readWrite>(lines_per_number_ * rows * cols);
    return matrix;
}

DeviceMatrix Apfp::MatrixMultiplication(const DeviceMatrix& a, const DeviceMatrix& b) {
    auto result = AllocateDeviceMatrix(a.rows(), b.cols());
    MatrixMultiplication(a, b, &result);
    return result;
}

void Apfp::MatrixMultiplication(const DeviceMatrix& a, const DeviceMatrix& b, DeviceMatrix* result) {
    if(a.cols() != b.rows() || result->rows() != a.rows() || result->cols() != b.cols()) {
        throw std::logic_error("Matrix dimension mismatch");
    }

    auto kernel = program_->MakeKernel("MatrixMultiplication", 
        a.buffer_, b.buffer_, result->buffer_, result->buffer_,
        a.rows(), b.rows(), result->cols());
    kernel.ExecuteTask();
}

void Apfp::TransposeInPlace(DeviceMatrix*) {
    throw std::exception();
}

DeviceMatrix Apfp::Transpose(const DeviceMatrix& a) {
    throw std::exception();
}

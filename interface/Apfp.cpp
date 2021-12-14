#include "Apfp.h"

#include <MatrixMultiplication.h>

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
    if (a.cols() != b.rows() || result->rows() != a.rows() || result->cols() != b.cols()) {
        throw std::logic_error("Matrix dimension mismatch");
    }
    auto kernel = program_->MakeKernel("MatrixMultiplication", a.buffer_, b.buffer_, result->buffer_, result->buffer_,
                                       static_cast<int>(a.rows()), static_cast<int>(b.rows()), static_cast<int>(result->cols()));
    kernel.ExecuteTask();
}

void Apfp::TransposeInPlace(DeviceMatrix*) {
    throw std::exception();
}

DeviceMatrix Apfp::Transpose(const DeviceMatrix& a) {
    throw std::exception();
}

template<typename ptr_function_type>
void DeviceMatrix::TransferToDeviceImpl(ptr_function_type buffer_ptr_func, std::size_t buffer_size) {
    if (rows() * cols() > buffer_size) {
        throw std::runtime_error("Source host buffer size smaller than destination device matrix size");
    }

    // TODO: This all assumes a bit width and will break once we need different runtime sizes
    std::vector<PackedFloat> host_buffer;
    host_buffer.resize(cols() * rows());

    for(std::size_t i = 0; i < host_buffer.size(); ++i) {
        host_buffer[i] = PackedFloat(*buffer_ptr_func(i));
    }

    buffer_.CopyFromHost(0, host_buffer.size() * kLinesPerNumber,
                         reinterpret_cast<DramLine const*>(host_buffer.data()));
}

void DeviceMatrix::TransferToDevice(const ApfpInterfaceType* buffer_ptr, std::size_t buffer_size) {
    TransferToDeviceImpl([&](std::size_t i) { return &buffer_ptr[i]; }, buffer_size);
}

void DeviceMatrix::TransferToDevice(const ApfpInterfaceWrapper* buffer_ptr, std::size_t buffer_size) {
    TransferToDeviceImpl([&](std::size_t i) { return buffer_ptr[i].get(); }, buffer_size);
}

void PackedFloatToInterfaceType(const PackedFloat& packed, mpfr_t dest) {
    packed.ToMpfr(dest);
}

void PackedFloatToInterfaceType(const PackedFloat& packed, mpf_t dest) {
    packed.ToGmp(dest);
}

template<typename ptr_function_type>
void DeviceMatrix::TransferToHostImpl(ptr_function_type buffer_ptr_func, std::size_t buffer_size) {
        if (rows() * cols() >= buffer_size) {
        throw std::runtime_error("Destination host buffer size smaller than source device matrix size");
    }

    std::vector<PackedFloat> host_buffer;
    host_buffer.resize(cols() * rows());

    buffer_.CopyToHost(0, kLinesPerNumber * rows() * cols(), reinterpret_cast<DramLine*>(host_buffer.data()));

    ApfpInterfaceWrapper scratch;
    for(std::size_t i = 0; i < host_buffer.size(); ++i) {
        PackedFloatToInterfaceType(host_buffer[i], *buffer_ptr_func(i));
    }
}

void DeviceMatrix::TransferToHost(ApfpInterfaceType* buffer_ptr, std::size_t buffer_size) {
    TransferToHostImpl([&](std::size_t i) { return &(buffer_ptr[i]); }, buffer_size);
}

void DeviceMatrix::TransferToHost(ApfpInterfaceWrapper* buffer_ptr, std::size_t buffer_size) {
    TransferToHostImpl([&](std::size_t i) { return buffer_ptr[i].get(); }, buffer_size);
}

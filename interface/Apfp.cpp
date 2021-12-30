#include "Apfp.h"

#include <MatrixMultiplication.h>

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "Config.h"

namespace apfp {

Apfp::Apfp() {
    auto kernel_path = FindKernel();
    program_.emplace(context_.MakeProgram(kernel_path));
    lines_per_number_ = kLinesPerNumber;
}

std::string Apfp::FindKernel() {
    {  // Specify a path to the APFP kernel manually
        char* apfp_kernel_env_var = std::getenv("APFP_KERNEL");
        if (apfp_kernel_env_var != nullptr) {
            auto kernel_override_path = std::filesystem::path(apfp_kernel_env_var);

            if (!std::filesystem::exists(kernel_override_path)) {
                throw std::runtime_error(
                    "APFP kernel path specified with APFP_KERNEL environment variable does not exist");
            }
            return kernel_override_path.string();
        }
    }

    char* apfp_use_simulation_env_var = std::getenv("APFP_USE_SIMULATION");
    auto apfp_use_simulation =
        apfp_use_simulation_env_var != nullptr && !std::string(apfp_use_simulation_env_var).empty();
    auto kernel_name = std::filesystem::path(apfp_use_simulation ? "MatrixMultiplication_hw_emu.xclbin"
                                                                 : "MatrixMultiplication_hw.xclbin");

    {  // Search for the kernel in /lib, /usr/lib, LD_LIBRARY_PATH, current directory
        std::vector<std::filesystem::path> search_paths;
        // System dirs
        search_paths.push_back(std::filesystem::path("/lib"));
        search_paths.push_back(std::filesystem::path("/usr/lib"));

        // LD_LIBRARY_PATH
        char* ld_library_path_env_var = std::getenv("LD_LIBRARY_PATH");
        auto ld_library_path = (ld_library_path_env_var == nullptr) ? "" : std::string(ld_library_path_env_var);

        for (std::string::iterator seg_begin = ld_library_path.begin(), seg_end; seg_begin < ld_library_path.end();
             seg_begin = seg_end + 1) {
            seg_end = std::find(seg_begin, ld_library_path.end(), ':');

            std::string candidate_path(seg_begin, seg_end);
            search_paths.push_back(std::filesystem::path(candidate_path));
        }

        // Current working directory
        search_paths.push_back(std::filesystem::current_path());

        // Search
        for (auto candidate_dir : search_paths) {
            auto candidate_kernel_path = candidate_dir / kernel_name;
            if (std::filesystem::exists(candidate_kernel_path)) {
                return candidate_kernel_path.string();
            }
        }
    }

    throw KernelNotFoundException("Unable to find FPGA kernel");
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
    auto kernel =
        program_->MakeKernel("MatrixMultiplication", a.buffer_, b.buffer_, result->buffer_, result->buffer_,
                             static_cast<int>(a.rows()), static_cast<int>(b.rows()), static_cast<int>(result->cols()));
    kernel.ExecuteTask();
}

void Apfp::MatrixAddition(const DeviceMatrix&, const DeviceMatrix&, DeviceMatrix*) {
    throw UnimplementedException();
}

void Apfp::TransposeInPlace(DeviceMatrix*) {
    throw UnimplementedException();
}

DeviceMatrix Apfp::Transpose(const DeviceMatrix&) {
    throw UnimplementedException();
}

template <typename ptr_function_type>
void DeviceMatrix::TransferToDeviceImpl(ptr_function_type buffer_ptr_func, std::size_t buffer_size) {
    if (rows() * cols() > buffer_size) {
        throw std::runtime_error("Source host buffer size smaller than destination device matrix size");
    }

    // TODO: This all assumes a bit width and will break once we need different runtime sizes
    std::vector<PackedFloat> host_buffer;
    host_buffer.resize(cols() * rows());

    for (std::size_t i = 0; i < host_buffer.size(); ++i) {
        host_buffer[i] = PackedFloat(buffer_ptr_func(i));
    }

    buffer_.CopyFromHost(0, host_buffer.size() * kLinesPerNumber,
                         reinterpret_cast<DramLine const*>(host_buffer.data()));
}

void DeviceMatrix::TransferToDevice(interface::ConstPtr buffer_ptr, std::size_t buffer_size) {
    TransferToDeviceImpl([&](std::size_t i) { return buffer_ptr + i; }, buffer_size);
}

void DeviceMatrix::TransferToDevice(const interface::Wrapper* buffer_ptr, std::size_t buffer_size) {
    TransferToDeviceImpl([&](std::size_t i) { return buffer_ptr[i].get(); }, buffer_size);
}

void PackedFloatToInterfaceType(const PackedFloat& packed, mpfr_ptr dest) {
    packed.ToMpfr(dest);
}

void PackedFloatToInterfaceType(const PackedFloat& packed, mpf_ptr dest) {
    packed.ToGmp(dest);
}

template <typename ptr_function_type>
void DeviceMatrix::TransferToHostImpl(ptr_function_type buffer_ptr_func, std::size_t buffer_size) {
    if (rows() * cols() > buffer_size) {
        throw std::runtime_error("Destination host buffer size smaller than source device matrix size");
    }

    std::vector<PackedFloat> host_buffer;
    host_buffer.resize(cols() * rows());

    buffer_.CopyToHost(0, kLinesPerNumber * host_buffer.size(), reinterpret_cast<DramLine*>(host_buffer.data()));

    interface::Wrapper scratch;
    for (std::size_t i = 0; i < host_buffer.size(); ++i) {
        PackedFloatToInterfaceType(host_buffer[i], buffer_ptr_func(i));
    }
}

void DeviceMatrix::TransferToHost(interface::Ptr buffer_ptr, std::size_t buffer_size) {
    TransferToHostImpl([&](std::size_t i) -> interface::Ptr { return buffer_ptr + i; }, buffer_size);
}

void DeviceMatrix::TransferToHost(interface::Wrapper* buffer_ptr, std::size_t buffer_size) {
    TransferToHostImpl([&](std::size_t i) -> interface::Ptr { return buffer_ptr[i].get(); }, buffer_size);
}

}  // namespace apfp
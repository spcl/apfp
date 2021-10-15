#include <hlslib/xilinx/OpenCL.h>
#include <hlslib/xilinx/Utility.h>

#include <cstdlib>  // putenv
#include <iostream>
#include <string>

#include "Config.h"
#include "MatrixMultiplication.h"
#include "MatrixMultiplicationReference.h"
#include "Random.h"

#ifdef HLSLIB_SIMULATE_OPENCL
bool RunTestSimulation(int size_n, int size_k, int size_m) {
    const std::string kernel_path("");
#else
bool RunTest(std::string const &kernel_path, int size_n, int size_k, int size_m) {
#endif
    hlslib::ocl::Context context;
    auto program = context.MakeProgram(kernel_path);
    // Initialize some random data
    std::vector<__mpf_struct> a_gmp, b_gmp, c_gmp;
    RandomNumberGenerator rng;
    for (int n = 0; n < size_n; ++n) {
        for (int k = 0; k < size_k; ++k) {
            a_gmp.emplace_back(rng.GenerateGmp());
            mpf_set_si(&a_gmp[n * size_k + k], 1);
        }
    }
    for (int k = 0; k < size_k; ++k) {
        for (int m = 0; m < size_m; ++m) {
            b_gmp.emplace_back(rng.GenerateGmp());
            mpf_set_si(&b_gmp[k * size_m + m], 1);
        }
    }
    for (int n = 0; n < size_n; ++n) {
        for (int m = 0; m < size_m; ++m) {
            c_gmp.emplace_back(rng.GenerateGmp());
            mpf_set_si(&c_gmp[n * size_m + m], 1);
        }
    }
    // Convert to PackedFloat format
    std::vector<PackedFloat> a_host, b_host, c_host;
    for (auto &x : a_gmp) {
        a_host.emplace_back(&x);
    }
    for (auto &x : b_gmp) {
        b_host.emplace_back(&x);
    }
    for (auto &x : c_gmp) {
        c_host.emplace_back(&x);
    }
    // Allocate device memory, padding each buffer to the tile size
    auto a_device = context.MakeBuffer<DramLine, hlslib::ocl::Access::read>(
        kLinesPerNumber * (hlslib::CeilDivide(size_n, kTileSizeN) * kTileSizeN) * size_k);
    auto b_device = context.MakeBuffer<DramLine, hlslib::ocl::Access::read>(
        kLinesPerNumber * size_k * (hlslib::CeilDivide(size_m, kTileSizeM) * kTileSizeM));
    auto c_device = context.MakeBuffer<DramLine, hlslib::ocl::Access::readWrite>(
        kLinesPerNumber * (hlslib::CeilDivide(size_n, kTileSizeN) * kTileSizeN) *
        (hlslib::CeilDivide(size_m, kTileSizeM) * kTileSizeM));
    // Copy data to the accelerator cast to 512-bit DRAM lines
    a_device.CopyFromHost(0, kLinesPerNumber * size_n * size_k, reinterpret_cast<DramLine const *>(&a_host[0]));
    b_device.CopyFromHost(0, kLinesPerNumber * size_k * size_m, reinterpret_cast<DramLine const *>(&b_host[0]));
    c_device.CopyFromHost(0, kLinesPerNumber * size_n * size_m, reinterpret_cast<DramLine const *>(&c_host[0]));
    // In simulation mode, this will call the function "MatrixMultiplication" and run it in software.
    // Otherwise, the provided path to a kernel binary will be loaded and executed.
    auto kernel = program.MakeKernel(MatrixMultiplication, "MatrixMultiplication", a_device, b_device, c_device,
                                     c_device, size_n, size_k, size_m);
    kernel.ExecuteTask();
    // Copy back result
    c_device.CopyToHost(0, kLinesPerNumber * size_n * size_m, reinterpret_cast<DramLine *>(&c_host[0]));
    // Run reference implementation. Because of GMP's "clever" way of wrapping their struct in an array of size 1,
    // allocating and passing arrays of GMP numbers is a mess
    MatrixMultiplicationReference(reinterpret_cast<mpf_t const *>(&a_gmp[0]),
                                  reinterpret_cast<mpf_t const *>(&b_gmp[0]), reinterpret_cast<mpf_t *>(&c_gmp[0]),
                                  size_n, size_k, size_m);
    // Verify results
    for (int n = 0; n < size_n; ++n) {
        for (int m = 0; m < size_m; ++m) {
            const PackedFloat res = c_host[n * size_m + m];
            const PackedFloat ref(&c_gmp[n * size_m + m]);
            if (ref != res) {
                std::cerr << "Verification failed at (" << n << ", " << m << "):\n\t" << res << "\n\t" << ref << "\n";
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char **argv) {
#ifndef HLSLIB_SIMULATE_OPENCL
    // Parse input
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " [hw_emu/hw] n k m\n";
        return 1;
    }
    const std::string mode_str(argv[1]);
    const int size_n = std::stoi(argv[2]);
    const int size_k = std::stoi(argv[3]);
    const int size_m = std::stoi(argv[4]);
    if (mode_str == "hw_emu") {
        const auto emu_str = "XCL_EMULATION_MODE=hw_emu";
        putenv(const_cast<char *>(emu_str));
        const auto conf_str = std::string("EMCONFIG_PATH=") + kBuildDir;
        putenv(const_cast<char *>(conf_str.c_str()));
        return !RunTest(kBuildDir + std::string("/MatrixMultiplication_hw_emu.xclbin"), size_n, size_k, size_m);
    } else if (mode_str == "hw") {
        return !RunTest(kBuildDir + std::string("/MatrixMultiplication_hw.xclbin"), size_n, size_k, size_m);
    } else {
        throw std::invalid_argument("Invalid mode specified.");
    }
#else
    // Parse input
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " n k m\n";
        return 1;
    }
    const int size_n = std::stoi(argv[1]);
    const int size_k = std::stoi(argv[2]);
    const int size_m = std::stoi(argv[3]);
    return !RunTestSimulation(size_n, size_k, size_m);
#endif
    return 0;
}

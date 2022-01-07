#include <hlslib/xilinx/OpenCL.h>
#include <hlslib/xilinx/Utility.h>

#include <cstdlib>  // putenv
#include <iostream>
#include <string>

#include "Config.h"
#include "MatrixMultiplication.h"
#include "MatrixMultiplicationReference.h"
#include "Random.h"

struct MpfrWrapper {
    mpfr_t x;

    operator mpfr_ptr() { return x; }
    operator mpfr_srcptr() const { return x; }
};

#ifdef HLSLIB_SIMULATE_OPENCL
bool RunTestSimulation(int size_n, int size_k, int size_m, bool verify) {
    const std::string kernel_path("");
#else
bool RunTest(std::string const &kernel_path, int size_n, int size_k, int size_m, bool verify) {
#endif
    hlslib::ocl::Context context;
    std::cout << "Configuring the device..." << std::flush;
    auto program = context.MakeProgram(kernel_path);
    std::cout << " Done.\n";
    // Initialize some random data
    std::cout << "Initializing input data..." << std::flush;
    std::vector<MpfrWrapper> a_mpfr, b_mpfr, c_mpfr;
    RandomNumberGenerator rng;
    for (int n = 0; n < size_n; ++n) {
        for (int k = 0; k < size_k; ++k) {
            a_mpfr.emplace_back();
            rng.GenerateMpfr(a_mpfr.back());
        }
    }
    for (int k = 0; k < size_k; ++k) {
        for (int m = 0; m < size_m; ++m) {
            b_mpfr.emplace_back();
            rng.GenerateMpfr(b_mpfr.back());
        }
    }
    for (int n = 0; n < size_n; ++n) {
        for (int m = 0; m < size_m; ++m) {
            c_mpfr.emplace_back();
            rng.GenerateMpfr(c_mpfr.back());
        }
    }
    // Convert to PackedFloat format
    std::vector<PackedFloat> a_host, b_host, c_host;
    for (auto &x : a_mpfr) {
        a_host.emplace_back(x);
    }
    for (auto &x : b_mpfr) {
        b_host.emplace_back(x);
    }
    for (auto &x : c_mpfr) {
        c_host.emplace_back(x);
    }
    std::cout << " Done.\n";
    // Allocate device memory, padding each buffer to the tile size
    std::cout << "Copying data to the device..." << std::flush;
    auto a_device = context.MakeBuffer<DramLine, hlslib::ocl::Access::read>(
        hlslib::ocl::StorageType::DDR, 1,
        kLinesPerNumber * (hlslib::CeilDivide(size_n, kTileSizeN) * kTileSizeN) * size_k);
    auto b_device = context.MakeBuffer<DramLine, hlslib::ocl::Access::read>(
        hlslib::ocl::StorageType::DDR, 1,
        kLinesPerNumber * size_k * (hlslib::CeilDivide(size_m, kTileSizeM) * kTileSizeM));
    auto c_device = context.MakeBuffer<DramLine, hlslib::ocl::Access::readWrite>(
        hlslib::ocl::StorageType::DDR, 1,
        kLinesPerNumber * (hlslib::CeilDivide(size_n, kTileSizeN) * kTileSizeN) *
            (hlslib::CeilDivide(size_m, kTileSizeM) * kTileSizeM));
    // Copy data to the accelerator cast to 512-bit DRAM lines
    a_device.CopyFromHost(0, kLinesPerNumber * size_n * size_k, reinterpret_cast<DramLine const *>(&a_host[0]));
    b_device.CopyFromHost(0, kLinesPerNumber * size_k * size_m, reinterpret_cast<DramLine const *>(&b_host[0]));
    c_device.CopyFromHost(0, kLinesPerNumber * size_n * size_m, reinterpret_cast<DramLine const *>(&c_host[0]));
    std::cout << " Done.\n";
    // In simulation mode, this will call the function "MatrixMultiplication" and run it in software.
    // Otherwise, the provided path to a kernel binary will be loaded and executed.
    auto kernel = program.MakeKernel(MatrixMultiplication, "MatrixMultiplication", a_device, b_device, c_device,
                                     c_device, size_n, size_k, size_m);
    const unsigned long expected_cycles = hlslib::CeilDivide(size_n, kTileSizeN) *
                                          hlslib::CeilDivide(size_m, kTileSizeM) * kTileSizeN * kTileSizeM * size_k;
    const float expected_runtime = expected_cycles / 0.3e9;
    std::cout << "The expected number of cycles to completion is " << expected_cycles << ", which is "
              << expected_runtime << " seconds at 300 MHz.\n";
    const auto communication_volume = hlslib::CeilDivide(size_n, kTileSizeN) * hlslib::CeilDivide(size_m, kTileSizeM) *
                                      ((kTileSizeN + kTileSizeM) * size_k + 2 * kTileSizeN * kTileSizeM);
    std::cout << "This communicates " << 1e-6 * kBytes * communication_volume << " MB, requiring a bandwidth of "
              << 1e-9 * kBytes * communication_volume / expected_runtime << " GB/s.\n";
    std::cout << "Executing kernel...\n";
    const auto elapsed = kernel.ExecuteTask();
    std::cout << "Ran in " << elapsed.first << " seconds.\n";

    if (!verify) {
        return true;
    }

    // Copy back result
    c_device.CopyToHost(0, kLinesPerNumber * size_n * size_m, reinterpret_cast<DramLine *>(&c_host[0]));
    // Run reference implementation. Because of GMP's "clever" way of wrapping their struct in an array of size 1,
    // allocating and passing arrays of GMP numbers is a mess
    std::cout << "Running reference implementation...\n";
    const auto start = std::chrono::high_resolution_clock::now();
    MatrixMultiplicationReference(reinterpret_cast<mpfr_t const *>(&a_mpfr[0]),
                                  reinterpret_cast<mpfr_t const *>(&b_mpfr[0]), reinterpret_cast<mpfr_t *>(&c_mpfr[0]),
                                  size_n, size_k, size_m);
    const auto end = std::chrono::high_resolution_clock::now();
    const double elapsed_reference = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Ran in " << elapsed_reference << " seconds.\n";
    // Verify results
    for (int n = 0; n < size_n; ++n) {
        for (int m = 0; m < size_m; ++m) {
            const PackedFloat res = c_host[n + m * size_n];
            const PackedFloat ref(c_mpfr[n + m * size_n]);
            if (ref != res) {
                std::cerr << "Verification failed at (" << n << ", " << m << "):\n\t" << res << "\n\t" << ref << "\n";
                return false;
            }
        }
    }
    std::cout << "Results successfully verified against MPFR.\n";

    // Clean up
    for (int n = 0; n < size_n; ++n) {
        for (int k = 0; k < size_k; ++k) {
            mpfr_clear(a_mpfr[n + k * size_n]);
        }
    }
    for (int k = 0; k < size_k; ++k) {
        for (int m = 0; m < size_m; ++m) {
            mpfr_clear(b_mpfr[k + m * size_k]);
        }
    }
    for (int n = 0; n < size_n; ++n) {
        for (int m = 0; m < size_m; ++m) {
            mpfr_clear(c_mpfr[n + m * size_n]);
        }
    }

    return true;
}

int main(int argc, char **argv) {
#ifndef HLSLIB_SIMULATE_OPENCL
    // Parse input
    if (argc < 5 || argc > 6) {
        std::cerr << "Usage: " << argv[0] << " [hw_emu/hw] n k m <verify [on/off]>\n";
        return 1;
    }
    const std::string mode_str(argv[1]);
    const int size_n = std::stoi(argv[2]);
    const int size_k = std::stoi(argv[3]);
    const int size_m = std::stoi(argv[4]);
    bool verify = true;
    if (argc == 6) {
        const std::string verify_str(argv[5]);
        if (verify_str == "on") {
            verify = true;
        } else if (verify_str == "off") {
            verify = false;
        } else {
            std::cerr << "Expected on/off.\n";
            return 1;
        }
    }
    if (mode_str == "hw_emu") {
        const auto emu_str = "XCL_EMULATION_MODE=hw_emu";
        putenv(const_cast<char *>(emu_str));
        const auto conf_str = std::string("EMCONFIG_PATH=") + kBuildDir;
        putenv(const_cast<char *>(conf_str.c_str()));
        return !RunTest(kBuildDir + std::string("/MatrixMultiplication_hw_emu.xclbin"), size_n, size_k, size_m, verify);
    } else if (mode_str == "hw") {
        return !RunTest(kBuildDir + std::string("/MatrixMultiplication_hw.xclbin"), size_n, size_k, size_m, verify);
    } else {
        throw std::invalid_argument("Invalid mode specified.");
    }
#else
    // Parse input
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: " << argv[0] << " n k m <verify [on/off]>\n";
        return 1;
    }
    const int size_n = std::stoi(argv[1]);
    const int size_k = std::stoi(argv[2]);
    const int size_m = std::stoi(argv[3]);
    bool verify = true;
    if (argc == 5) {
        const std::string verify_str(argv[4]);
        if (verify_str == "on") {
            verify = true;
        } else if (verify_str == "off") {
            verify = false;
        } else {
            std::cerr << "Expected on/off.\n";
            return 1;
        }
    }
    return !RunTestSimulation(size_n, size_k, size_m, verify);
#endif
}

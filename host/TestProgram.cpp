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

    operator mpfr_ptr() {
        return x;
    }
    operator mpfr_srcptr() const {
        return x;
    }
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

    // Compute partitions
    int n_begin[kComputeUnits];
    int n_end[kComputeUnits];
    int n_partition_size[kComputeUnits];
    unsigned long expected_cycles = 0;
    for (int i = 0; i < kComputeUnits; ++i) {
        n_begin[i] = (i * size_n) / kComputeUnits;
        n_end[i] = ((i + 1) * size_n) / kComputeUnits;
        n_partition_size[i] = n_end[i] - n_begin[i];
        expected_cycles =
            std::max(expected_cycles,
                     (unsigned long)(hlslib::CeilDivide(n_partition_size[i], kTileSizeN) *
                                     hlslib::CeilDivide(size_m, kTileSizeM) * kTileSizeN * kTileSizeM * size_k));
    }

    // Allocate device memory, padding each buffer to the tile size
    std::cout << "Copying data to the device..." << std::flush;
    constexpr int kDramMapping[] = {1, 0, 2, 3};
    std::vector<hlslib::ocl::Buffer<DramLine, hlslib::ocl::Access::read>> a_device;
    std::vector<hlslib::ocl::Buffer<DramLine, hlslib::ocl::Access::read>> b_device;
    std::vector<hlslib::ocl::Buffer<DramLine, hlslib::ocl::Access::readWrite>> c_device;
    for (int i = 0; i < kComputeUnits; ++i) {
        const auto bank = i % 4;
        a_device.emplace_back(
            context, hlslib::ocl::StorageType::DDR, kDramMapping[bank],
            kLinesPerNumber * (hlslib::CeilDivide(n_partition_size[i], kTileSizeN) * kTileSizeN) * size_k);
        b_device.emplace_back(context, hlslib::ocl::StorageType::DDR, kDramMapping[bank],
                              kLinesPerNumber * size_k * (hlslib::CeilDivide(size_m, kTileSizeM) * kTileSizeM));
        c_device.emplace_back(context, hlslib::ocl::StorageType::DDR, kDramMapping[bank],
                              kLinesPerNumber * (hlslib::CeilDivide(n_partition_size[i], kTileSizeN) * kTileSizeN) *
                                  (hlslib::CeilDivide(size_m, kTileSizeM) * kTileSizeM));
        // Copy data to the accelerator cast to 512-bit DRAM lines
        a_device[i].CopyFromHost(0, kLinesPerNumber * n_partition_size[i] * size_k,
                                 reinterpret_cast<DramLine const *>(&a_host[n_begin[i] * size_k]));
        b_device[i].CopyFromHost(0, kLinesPerNumber * size_k * size_m, reinterpret_cast<DramLine const *>(&b_host[0]));
        c_device[i].CopyFromHost(0, kLinesPerNumber * n_partition_size[i] * size_m,
                                 reinterpret_cast<DramLine const *>(&c_host[n_begin[i] * size_m]));
    }
    std::cout << " Done.\n";

    // In simulation mode, this will call the function "MatrixMultiplication" and run it in software.
    // Otherwise, the provided path to a kernel binary will be loaded and executed.
    std::vector<hlslib::ocl::Kernel> kernels;
    for (int i = 0; i < kComputeUnits; ++i) {
        kernels.emplace_back(program.MakeKernel(MatrixMultiplication, "MatrixMultiplication", a_device[i], b_device[i],
                                                c_device[i], c_device[i], n_partition_size[i], size_k, size_m));
    }

    const float expected_runtime = expected_cycles / 0.3e9;
    std::cout << "The expected number of cycles to completion is " << expected_cycles << ", which is "
              << expected_runtime << " seconds at 300 MHz.\n";
    const auto communication_volume = hlslib::CeilDivide(size_n, kTileSizeN) * hlslib::CeilDivide(size_m, kTileSizeM) *
                                      ((kTileSizeN + kTileSizeM) * size_k + 2 * kTileSizeN * kTileSizeM);
    const auto bandwidth = 1e-9 * kBytes * communication_volume / expected_runtime;
    std::cout << "This communicates " << 1e-6 * kBytes * communication_volume << " MB, requiring a bandwidth of "
              << bandwidth << " GB/s.\n";

    std::cout << "Executing kernel...\n";
    std::vector<hlslib::ocl::Event> events;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kComputeUnits; ++i) {
        events.emplace_back(kernels[i].ExecuteTaskAsync());
    }
    hlslib::ocl::WaitForEvents(events);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Ran in " << elapsed << " seconds.\n";

    if (!verify) {
        return true;
    }

    // Copy back result
    std::cout << "Copying back result..." << std::flush;
    std::vector<PackedFloat> result(size_n * size_m);
    for (int i = 0; i < kComputeUnits; ++i) {
        c_device[i].CopyToHost(0, kLinesPerNumber * n_partition_size[i] * size_m,
                               reinterpret_cast<DramLine *>(&result[n_begin[i] * size_m]));
    }
    std::cout << "Done.\n";

    // Run reference implementation. Because of GMP's "clever" way of wrapping their struct in an array of size 1,
    // allocating and passing arrays of GMP numbers is a mess
    std::cout << "Running reference implementation...\n";
    start = std::chrono::high_resolution_clock::now();
    MatrixMultiplicationReference(reinterpret_cast<mpfr_t const *>(&a_mpfr[0]),
                                  reinterpret_cast<mpfr_t const *>(&b_mpfr[0]), reinterpret_cast<mpfr_t *>(&c_mpfr[0]),
                                  size_n, size_k, size_m);
    end = std::chrono::high_resolution_clock::now();
    const double elapsed_reference = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Ran in " << elapsed_reference << " seconds.\n";

    // Verify results
    for (int n = 0; n < size_n; ++n) {
        for (int m = 0; m < size_m; ++m) {
            const PackedFloat res = result[n * size_m + m];
            const PackedFloat ref(c_mpfr[n * size_m + m]);
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
            mpfr_clear(a_mpfr[n * size_k + k]);
        }
    }
    for (int k = 0; k < size_k; ++k) {
        for (int m = 0; m < size_m; ++m) {
            mpfr_clear(b_mpfr[k * size_m + m]);
        }
    }
    for (int n = 0; n < size_n; ++n) {
        for (int m = 0; m < size_m; ++m) {
            mpfr_clear(c_mpfr[n * size_m + m]);
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

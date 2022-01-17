#include <hlslib/xilinx/OpenCL.h>
#include <hlslib/xilinx/Utility.h>

#include <cstdlib>  // putenv
#include <iostream>
#include <string>

#include "Config.h"
#include "Microbenchmark.h"
#include "MicrobenchmarkReference.h"
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
bool RunTestSimulation(int size, bool verify) {
    const std::string kernel_path("");
#else
bool RunTest(std::string const &kernel_path, int size, bool verify) {
#endif

    hlslib::ocl::Context context;
    std::cout << "Configuring the device..." << std::flush;
    auto program = context.MakeProgram(kernel_path);
    std::cout << " Done.\n";

    // Compute partitions
    int i_begin[kComputeUnits];
    int i_end[kComputeUnits];
    int partition_size[kComputeUnits];
    unsigned long expected_cycles = 0;
    for (int i = 0; i < kComputeUnits; ++i) {
        i_begin[i] = (i * size) / kComputeUnits;
        i_end[i] = ((i + 1) * size) / kComputeUnits;
        partition_size[i] = i_end[i] - i_begin[i];
        expected_cycles = std::max(expected_cycles, (unsigned long)(partition_size[i]));
    }

    // Initialize some random data
    std::cout << "Initializing input data..." << std::flush;
    std::vector<MpfrWrapper> a_mpfr, b_mpfr, c_mpfr;
    RandomNumberGenerator rng;
#ifndef APFP_FAKE_MEMORY
    for (int i = 0; i < size; ++i) {
        a_mpfr.emplace_back();
        rng.GenerateMpfr(a_mpfr.back());
    }
    for (int i = 0; i < size; ++i) {
        b_mpfr.emplace_back();
        rng.GenerateMpfr(b_mpfr.back());
    }
    for (int i = 0; i < size; ++i) {
        c_mpfr.emplace_back();
        rng.GenerateMpfr(c_mpfr.back());
    }
#else
    a_mpfr.emplace_back();
    b_mpfr.emplace_back();
    c_mpfr.emplace_back();
    rng.GenerateMpfr(a_mpfr[0]);
    rng.GenerateMpfr(b_mpfr[0]);
    rng.GenerateMpfr(c_mpfr[0]);
#endif

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
    constexpr int kDramMapping[] = {1, 0, 2, 3};
    std::vector<hlslib::ocl::Buffer<DramLine, hlslib::ocl::Access::read>> a_device;
    std::vector<hlslib::ocl::Buffer<DramLine, hlslib::ocl::Access::read>> b_device;
    std::vector<hlslib::ocl::Buffer<DramLine, hlslib::ocl::Access::readWrite>> c_device;
    for (int i = 0; i < kComputeUnits; ++i) {
        const auto bank = i % 4;
        a_device.emplace_back(context, hlslib::ocl::StorageType::DDR, kDramMapping[bank],
                              kLinesPerNumber * partition_size[i]);
        b_device.emplace_back(context, hlslib::ocl::StorageType::DDR, kDramMapping[bank],
                              kLinesPerNumber * partition_size[i]);
        c_device.emplace_back(context, hlslib::ocl::StorageType::DDR, kDramMapping[bank],
                              kLinesPerNumber * partition_size[i]);
        // Copy data to the accelerator cast to 512-bit DRAM lines
#ifndef APFP_FAKE_MEMORY
        a_device[i].CopyFromHost(0, kLinesPerNumber * partition_size[i],
                                 reinterpret_cast<DramLine const *>(&a_host[i_begin[i]]));
        b_device[i].CopyFromHost(0, kLinesPerNumber * partition_size[i],
                                 reinterpret_cast<DramLine const *>(&b_host[i_begin[i]]));
        c_device[i].CopyFromHost(0, kLinesPerNumber * partition_size[i],
                                 reinterpret_cast<DramLine const *>(&c_host[i_begin[i]]));
#else
        a_device[i].CopyFromHost(0, kLinesPerNumber, reinterpret_cast<DramLine const *>(&a_host[0]));
        b_device[i].CopyFromHost(0, kLinesPerNumber, reinterpret_cast<DramLine const *>(&b_host[0]));
        c_device[i].CopyFromHost(0, kLinesPerNumber, reinterpret_cast<DramLine const *>(&c_host[0]));
#endif
    }
    std::cout << " Done.\n";

    // In simulation mode, this will call the function "Microbenchmark" and run it in software.
    // Otherwise, the provided path to a kernel binary will be loaded and executed.
    std::vector<hlslib::ocl::Kernel> kernels;
    for (int i = 0; i < kComputeUnits; ++i) {
        kernels.emplace_back(program.MakeKernel(Microbenchmark, "Microbenchmark", a_device[i], b_device[i], c_device[i],
                                                partition_size[i]));
    }

    const float expected_runtime = expected_cycles / 0.3e9;
    std::cout << "The expected number of cycles to completion is " << expected_cycles << ", which is "
              << expected_runtime << " seconds at 300 MHz.\n";
    const auto communication_volume = 3 * size;
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
#ifndef APFP_FAKE_MEMORYH
    std::vector<PackedFloat> result(size);
    for (int i = 0; i < kComputeUnits; ++i) {
        c_device[i].CopyToHost(0, kLinesPerNumber * partition_size[i],
                               reinterpret_cast<DramLine *>(&result[i_begin[i]]));
    }
#else
    std::vector<PackedFloat> result(kComputeUnits);
    for (int i = 0; i < kComputeUnits; ++i) {
        c_device[i].CopyToHost(0, kLinesPerNumber, reinterpret_cast<DramLine *>(&result[i]));
    }
#endif
    std::cout << "Done.\n";

    // Run reference implementation. Because of GMP's "clever" way of wrapping their struct in an array of size 1,
    // allocating and passing arrays of GMP numbers is a mess
    std::cout << "Running reference implementation..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    MicrobenchmarkReference(reinterpret_cast<mpfr_t const *>(&a_mpfr[0]), reinterpret_cast<mpfr_t const *>(&b_mpfr[0]),
                            reinterpret_cast<mpfr_t *>(&c_mpfr[0]), size);
    end = std::chrono::high_resolution_clock::now();
    const double elapsed_reference = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Ran in " << elapsed_reference << " seconds.\n";

    // Verify results
#ifndef APFP_FAKE_MEMORY
    for (int i = 0; i < size; ++i) {
        const PackedFloat res = result[i];
        const PackedFloat ref(c_mpfr[i]);
        if (ref != res) {
            std::cerr << "Verification failed at " << i << ":\n\t" << res << "\n\t" << ref << "\n";
            return false;
        }
    }
#else
    for (int i = 0; i < kComputeUnits; ++i) {
        const PackedFloat res = result[i];
        const PackedFloat ref(c_mpfr[0]);
        if (res != ref) {
            std::cerr << "Verification failed for compute unit " << i << ":\n\t" << res << "\n\t" << ref << "\n";
            return false;
        }
    }
#endif
    std::cout << "Results successfully verified against MPFR.\n";

    // Clean up
#ifndef APFP_FAKE_MEMORY
    for (int i = 0; i < size; ++i) {
        mpfr_clear(a_mpfr[i]);
    }
    for (int i = 0; i < size; ++i) {
        mpfr_clear(b_mpfr[i]);
    }
    for (int i = 0; i < size; ++i) {
        mpfr_clear(c_mpfr[i]);
    }
#else
    mpfr_clear(a_mpfr[0]);
    mpfr_clear(b_mpfr[0]);
    mpfr_clear(c_mpfr[0]);
#endif

    return true;
}

int main(int argc, char **argv) {
#ifndef HLSLIB_SIMULATE_OPENCL
    // Parse input
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " [hw_emu/hw] n <verify [on/off]>\n";
        return 1;
    }
    const std::string mode_str(argv[1]);
    const int size = std::stoi(argv[2]);
    bool verify = true;
    if (argc == 4) {
        const std::string verify_str(argv[3]);
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
        return !RunTest(kBuildDir + std::string("/Microbenchmark_hw_emu.xclbin"), size, verify);
    } else if (mode_str == "hw") {
        return !RunTest(kBuildDir + std::string("/Microbenchmark_hw.xclbin"), size, verify);
    } else {
        throw std::invalid_argument("Invalid mode specified.");
    }
#else
    // Parse input
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " n k m <verify [on/off]>\n";
        return 1;
    }
    const int size = std::stoi(argv[1]);
    bool verify = true;
    if (argc == 3) {
        const std::string verify_str(argv[2]);
        if (verify_str == "on") {
            verify = true;
        } else if (verify_str == "off") {
            verify = false;
        } else {
            std::cerr << "Expected on/off.\n";
            return 1;
        }
    }
    return !RunTestSimulation(size, verify);
#endif
}

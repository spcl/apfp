#include <hlslib/xilinx/OpenCL.h>

#include <cstdlib>  // putenv
#include <iostream>
#include <string>

#include "Config.h"
#include "MatrixMultiplication.h"

#ifdef HLSLIB_SIMULATE_OPENCL
void RunTestSimulation(int size_n, int size_k, int size_m) {
    const std::string kernel_path("");
#else
void RunTest(std::string const &kernel_path, int size_n, int size_k, int size_m) {
#endif
    hlslib::ocl::Context context;
    auto program = context.MakeProgram(kernel_path);
    auto a_device = context.MakeBuffer<DramLine, hlslib::ocl::Access::read>(kLinesPerNumber * size_n * size_k);
    auto b_device = context.MakeBuffer<DramLine, hlslib::ocl::Access::read>(kLinesPerNumber * size_k * size_m);
    auto c_device = context.MakeBuffer<DramLine, hlslib::ocl::Access::read>(kLinesPerNumber * size_n * size_m);
    // In simulation mode, this will call the function "MatrixMultiplication" and run it in software.
    // Otherwise, the provided path to a kernel binary will be loaded and executed.
    auto kernel = program.MakeKernel(MatrixMultiplication, "MatrixMultiplication", a_device, b_device, c_device,
                                     c_device, size_n, size_k, size_m);
    kernel.ExecuteTask();
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
        RunTest(kBuildDir + std::string("/MatrixMultiplication_hw_emu.xclbin"), size_n, size_k, size_m);
    } else if (mode_str == "hw") {
        RunTest(kBuildDir + std::string("/MatrixMultiplication_hw.xclbin"), size_n, size_k, size_m);
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
    RunTestSimulation(size_n, size_k, size_m);
#endif
    return 0;
}

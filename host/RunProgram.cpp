#include <hlslib/xilinx/OpenCL.h>

#include <cstdlib>  // putenv
#include <iostream>
#include <string>

#include "Config.h"
#include "MatrixMultiplication.h"

#ifdef HLSLIB_SIMULATE_OPENCL
void RunSimulation() {
    const std::string kernel_path("");
#else
void Run(std::string const &kernel_path) {
#endif
    hlslib::ocl::Context context;
    auto program = context.MakeProgram(kernel_path);
    auto mantissa_device = context.MakeBuffer<Mantissa, hlslib::ocl::Access::read>(1);
    auto exponent_device = context.MakeBuffer<Exponent, hlslib::ocl::Access::read>(1);
    auto sign_device = context.MakeBuffer<Sign, hlslib::ocl::Access::read>(1);
    // In simulation mode, this will call the function "MatrixMultiplication" and run it in software.
    // Otherwise, the provided path to a kernel binary will be loaded and executed.
    auto kernel = program.MakeKernel(MatrixMultiplication, "MatrixMultiplication", mantissa_device, exponent_device,
                                     sign_device, mantissa_device, exponent_device, sign_device, mantissa_device,
                                     exponent_device, sign_device, 0, 0, 0);
}

#ifndef HLSLIB_SIMULATE_OPENCL

void RunHardwareEmulation() {
    const auto env_str = "XCL_EMULATION_MODE=hw_emu";
    putenv(const_cast<char *>(env_str));
    Run(kBuildDir + std::string("MatrixMultiplication_hw_emu.xclbin"));
}

void RunHardware() {
    Run(kBuildDir + std::string("MatrixMultiplication_hw.xclbin"));
}

#endif

int main(int argc, char **argv) {
#ifndef HLSLIB_SIMULATE_OPENCL
    // Parse input
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [sw/hw_emu/hw]";
        return 1;
    }
    const std::string mode_str(argv[1]);
    if (mode_str == "hw_emu") {
        RunHardwareEmulation();
    } else if (mode_str == "hw") {
        RunHardware();
    } else {
        throw std::invalid_argument("Invalid mode specified.");
    }
#else
    RunSimulation();
#endif

    return 0;
}

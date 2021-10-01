cmake_minimum_required(VERSION 3.0)
project(apfp)

set(CMAKE_CXX_STANDARD 17)

# Target options 
set(APFP_PLATFORM "xilinx_u280_xdma_201920_3" CACHE STRING "Platform string for Vitis.")
set(APFP_MANTISSA_BITS 1024 CACHE STRING "Number of bits to use for the mantissa.")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/hlslib/cmake ${CMAKE_SOURCE_DIR}/cmake)

find_package(Vitis REQUIRED)
find_package(GMP REQUIRED)

configure_file(include/Config.h.in Config.h)

set(APFP_KERNEL_FILES device/MatrixMultiplication.cpp)

# Setup FPGA kernel targets
add_vitis_kernel(MatrixMultiplication FILES ${APFP_KERNEL_FILES}
                 INCLUDE_DIRS include ${CMAKE_BINARY_DIR}
                 DEPENDS ${CMAKE_BINARY_DIR}/Config.h
                         include/MatrixMultiplication.h)
add_vitis_program(MatrixMultiplication ${APFP_PLATFORM})

include_directories(${Vitis_INCLUDE_DIRS} ${CMAKE_BINARY_DIR} include hlslib/include)

add_library(simulation ${APFP_KERNEL_FILES})

add_executable(RunSimulation host/RunProgram.cpp)
target_link_libraries(RunSimulation simulation ${Vitis_LIBRARIES} ${GMP_LIBRARIES}) 
target_compile_definitions(RunSimulation PRIVATE HLSLIB_SIMULATE_OPENCL)

add_executable(RunHardware host/RunProgram.cpp)
target_link_libraries(RunHardware simulation ${Vitis_LIBRARIES} ${GMP_LIBRARIES}) 

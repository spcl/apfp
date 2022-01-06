cmake_minimum_required(VERSION 3.0)
project(apfp)

set(CMAKE_CXX_STANDARD 17)

# Target options 
set(APFP_PLATFORM "xilinx_u250_gen3x16_xdma_3_1_202020_1" CACHE STRING "Platform string for Vitis.")
set(APFP_BITS 1024 CACHE STRING "Number of bits to use for a floating point number, including mantissa, exponent, and sign.")
set(APFP_MULT_BASE_BITS 18 CACHE STRING "Number of bits to bottom out the multiplication at and use native multiplication.")
set(APFP_TILE_SIZE_N 32 CACHE STRING "Tile size in the N-dimension when running matrix-matrix multiplication.")
set(APFP_TILE_SIZE_M 32 CACHE STRING "Tile size in the M-dimension when running matrix-matrix multiplication.")
set(APFP_SEMANTICS "MPFR" CACHE STRING "Which semantics to use for floating point operations [GMP/MPFR].")
set(APFP_DEBUGGING OFF CACHE BOOL "Enable debugging in generated kernels.")
set(APFP_PROFILING OFF CACHE BOOL "Enable profiling in generated kernels.")
set(APFP_SAVE_TEMPS OFF CACHE BOOL "Save temporary files from kernel builds.")
set_property(CACHE APFP_SEMANTICS PROPERTY STRINGS GMP MPFR)

# One day we might accept both
set(APFP_INTERFACE_TYPE ${APFP_SEMANTICS})
# but not today

# Validation and derived numbers
math(EXPR APFP_ALIGNED "${APFP_BITS} % 512")
if(NOT APFP_ALIGNED EQUAL 0)
    message(FATAL_ERROR "Number of bits ${APFP_BITS} must be aligned to the DRAM line size of 512 bits.")
endif()
math(EXPR APFP_MAX_BITS "${APFP_BITS} * 2")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/hlslib/cmake ${CMAKE_SOURCE_DIR}/cmake)

find_package(Vitis REQUIRED)
find_package(MPFR REQUIRED)
find_package(GMP REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-unused-label -Wno-unknown-pragmas -Wno-class-memaccess -DAPFP_${APFP_SEMANTICS}_SEMANTICS")
include_directories(${CMAKE_BINARY_DIR} include SYSTEM hlslib/include ${Vitis_INCLUDE_DIRS} interface)

configure_file(include/Config.h.in Config.h)

set(APFP_KERNEL_FILES device/MatrixMultiplication.cpp
                      device/ArithmeticOperations.cpp
                      device/Karatsuba.cpp)

# Setup FPGA kernel targets
add_vitis_kernel(MatrixMultiplication FILES ${APFP_KERNEL_FILES}
                 INCLUDE_DIRS include hlslib/include ${CMAKE_BINARY_DIR} ${GMP_INCLUDES}
                 HLS_FLAGS "-DAP_INT_MAX_W=${APFP_MAX_BITS} -DAPFP_${APFP_SEMANTICS}_SEMANTICS"
                 HLS_CONFIG "config_compile -pipeline_style frp\nconfig_dataflow -fifo_depth 16"
                 DEPENDS ${CMAKE_BINARY_DIR}/Config.h
                         include/ArithmeticOperations.h
                         include/Karatsuba.h
                         include/MatrixMultiplication.h
                         include/PackedFloat.h
                PORT_MAPPING m_axi_a:DDR[1]
                             m_axi_b:DDR[1]
                             m_axi_c_read:DDR[1]
                             m_axi_c_write:DDR[1])
add_vitis_program(MatrixMultiplication ${APFP_PLATFORM}
                  PROFILING ${APFP_PROFILING}
                  DEBUGGING ${APFP_DEBUGGING}
                  SAVE_TEMPS ${APFP_SAVE_TEMPS})

# Internal library 
add_library(apfp host/Random.cpp host/MatrixMultiplicationReference.cpp)
target_link_libraries(apfp ${GMP_LIBRARIES} ${MPFR_LIBRARIES})

# Library necessary to run in simulation mode
add_library(simulation ${APFP_KERNEL_FILES})
target_compile_options(simulation PRIVATE -Wno-unknown-pragmas -DAP_INT_MAX_W=${APFP_MAX_BITS})
target_link_libraries(simulation ${CMAKE_THREAD_LIBS_INIT})

add_library(apfpHostlib SHARED interface/Apfp.cpp interface/ApfpBlas.cpp interface/ApfpInterfaceType.cpp)
target_link_libraries(apfpHostlib ${Vitis_LIBRARIES} ${GMP_LIBRARIES}) 
target_compile_definitions(apfpHostlib PRIVATE HLSLIB_SIMULATE_OPENCL)

# Executable used to run in simulation mode, calling the kernel as a C++ function directly
add_executable(TestSimulation host/TestProgram.cpp)
target_link_libraries(TestSimulation apfp simulation ${Vitis_LIBRARIES} ${GMP_LIBRARIES} ${MPFR_LIBRARIES}) 
target_compile_definitions(TestSimulation PRIVATE HLSLIB_SIMULATE_OPENCL)

# Executable used to run from an xclbin binary
add_executable(TestHardware host/TestProgram.cpp)
target_link_libraries(TestHardware apfp simulation ${Vitis_LIBRARIES} ${GMP_LIBRARIES} ${MPFR_LIBRARIES}) 

# Testing
enable_testing()
add_test(TestSimulation TestSimulation 4 4 4)
add_library(Catch host/Catch.cpp)
add_executable(UnitTests host/UnitTests.cpp)
target_link_libraries(UnitTests Catch ${GMP_LIBRARIES} ${MPFR_LIBRARIES} apfp apfpHostlib simulation)
add_test(UnitTests UnitTests)

add_executable(BlasUnitTests host/BlasUnitTests.cpp)
target_link_libraries(BlasUnitTests Catch ${GMP_LIBRARIES} ${MPFR_LIBRARIES} apfp apfpHostlib simulation)

install(TARGETS apfpHostlib)
install(FILES
    interface/Apfp.h
    interface/ApfpBlas.h
    interface/ApfpInterfaceType.h
    ${CMAKE_BINARY_DIR}/Config.h
    DESTINATION include/apfp)
install(FILES
    ${CMAKE_BINARY_DIR}/MatrixMultiplication_hw.xclbin
    ${CMAKE_BINARY_DIR}/MatrixMultiplication_hw_emu.xclbin
    DESTINATION lib)

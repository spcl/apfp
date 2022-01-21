cmake_minimum_required(VERSION 3.0)
project(apfp)

# Target options 
set(APFP_PLATFORM "xilinx_u250_gen3x16_xdma_3_1_202020_1" CACHE STRING "Platform string for Vitis.")
set(APFP_FREQUENCY "" CACHE STRING "Target frequency for design (if left empty, the shell's default will be used).")
set(APFP_BITS 1024 CACHE STRING "Number of bits to use for a floating point number, including mantissa, exponent, and sign.")
set(APFP_MULT_BASE_BITS 36 CACHE STRING "Number of bits to bottom out the multiplication at and use native multiplication.")
set(APFP_ADD_BASE_BITS 256 CACHE STRING "Number of bits to bottom out and use the built-in adder.")
set(APFP_USE_PIPELINED_ADD ON CACHE BOOL "Use custom pipelined adder to insert more pipeline stages.")
set(APFP_TILE_SIZE_N 32 CACHE STRING "Tile size in the N-dimension when running matrix-matrix multiplication.")
set(APFP_TILE_SIZE_M 32 CACHE STRING "Tile size in the M-dimension when running matrix-matrix multiplication.")
set(APFP_COMPUTE_UNITS 1 CACHE STRING "Number of replications of the kernel to instantiate.")
set(APFP_FIX_SLRS OFF CACHE STRING "Fix compute units to SLRs. Will not work for larger kernels that spill across SLRs.")
set(APFP_SEMANTICS "MPFR" CACHE STRING "Which semantics to use for floating point operations [GMP/MPFR].")
set(APFP_DEBUGGING OFF CACHE BOOL "Enable debugging in generated kernels.")
set(APFP_PROFILING OFF CACHE BOOL "Enable profiling in generated kernels.")
set(APFP_SAVE_TEMPS ON CACHE BOOL "Save temporary files from kernel builds.")
set_property(CACHE APFP_SEMANTICS PROPERTY STRINGS GMP MPFR)

# Validation and derived numbers
math(EXPR APFP_ALIGNED "${APFP_BITS} % 512")
if(NOT APFP_ALIGNED EQUAL 0)
    message(FATAL_ERROR "Number of bits ${APFP_BITS} must be aligned to the DRAM line size of 512 bits.")
endif()
math(EXPR APFP_MAX_BITS "${APFP_BITS} * 2 + 1")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/hlslib/cmake ${CMAKE_SOURCE_DIR}/cmake)

find_package(Vitis REQUIRED)
find_package(MPFR REQUIRED)
find_package(GMP REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -Wpedantic -Wno-unused-label -Wno-class-memaccess -Wno-unknown-pragmas -DAPFP_${APFP_SEMANTICS}_SEMANTICS -DAP_INT_MAX_W=${APFP_MAX_BITS}")
if(APFP_USE_PIPELINED_ADD)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAPFP_USE_PIPELINED_ADD")
endif()

include_directories(${CMAKE_BINARY_DIR} include SYSTEM hlslib/include ${Vitis_INCLUDE_DIRS} )

configure_file(include/Config.h.in Config.h)

# Mapping to DDR ports (currently hardcoded to U250, should be configurable)
set(APFP_BANK_ROTATION 1 0 2 3)
foreach(APFP_CU RANGE 1 ${APFP_COMPUTE_UNITS})
    math(EXPR APFP_CU_INDEX "(${APFP_CU} - 1) % 4")
    list(GET APFP_BANK_ROTATION ${APFP_CU_INDEX} APFP_BANK_INDEX)
    set(APFP_MMM_PORT_MAPPING ${APFP_MMM_PORT_MAPPING}
                              MatrixMultiplication_${APFP_CU}.m_axi_a:DDR[${APFP_BANK_INDEX}]
                              MatrixMultiplication_${APFP_CU}.m_axi_b:DDR[${APFP_BANK_INDEX}]
                              MatrixMultiplication_${APFP_CU}.m_axi_c_read:DDR[${APFP_BANK_INDEX}]
                              MatrixMultiplication_${APFP_CU}.m_axi_c_write:DDR[${APFP_BANK_INDEX}])
    set(APFP_MICROBENCHMARK_PORT_MAPPING ${APFP_MICROBENCHMARK_PORT_MAPPING}
                                         Microbenchmark_${APFP_CU}.m_axi_a:DDR[${APFP_BANK_INDEX}]
                                         Microbenchmark_${APFP_CU}.m_axi_b:DDR[${APFP_BANK_INDEX}]
                                         Microbenchmark_${APFP_CU}.m_axi_c:DDR[${APFP_BANK_INDEX}])
    if(APFP_FIX_SLRS)
        set(APFP_SLR_MAPPING ${APFP_SLR_MAPPING} MatrixMultiplication_${APFP_CU}:SLR${APFP_BANK_INDEX})
        set(APFP_SLR_MAPPING ${APFP_SLR_MAPPING} Microbenchmark_${APFP_CU}:SLR${APFP_BANK_INDEX})
    endif()
endforeach()

# Setup FPGA kernel target for the matrix multiplication accelerator
set(APFP_INCLUDES ${CMAKE_BINARY_DIR}/Config.h
                  include/ArithmeticOperations.h
                  include/DeviceTypes.h
                  include/Karatsuba.h
                  include/PackedFloat.h
                  include/PipelinedAdd.h)
add_vitis_kernel(MatrixMultiplication
                 FILES device/MatrixMultiplication.cpp
                       device/ArithmeticOperations.cpp
                       device/Karatsuba.cpp
                 COMPUTE_UNITS ${APFP_COMPUTE_UNITS}
                 INCLUDE_DIRS include hlslib/include ${CMAKE_BINARY_DIR}
                 HLS_FLAGS ${CMAKE_CXX_FLAGS}
                 HLS_CONFIG "config_compile -pipeline_style frp\nconfig_dataflow -fifo_depth 16"
                 DEPENDS ${APFP_INCLUDES} include/MatrixMultiplication.h
                 PORT_MAPPING ${APFP_MMM_PORT_MAPPING}
                 SLR_MAPPING ${APFP_MMM_SLR_MAPPING})
add_vitis_program(MatrixMultiplication ${APFP_PLATFORM}
                  CLOCK ${APFP_FREQUENCY}
                  PROFILING ${APFP_PROFILING}
                  DEBUGGING ${APFP_DEBUGGING}
                  SAVE_TEMPS ${APFP_SAVE_TEMPS})

# Setup FPGA kernel targets for APFP floating point multiplication microbenchmark
add_vitis_kernel(Microbenchmark
                 FILES device/Microbenchmark.cpp
                       device/ArithmeticOperations.cpp
                       device/Karatsuba.cpp
                 COMPUTE_UNITS ${APFP_COMPUTE_UNITS}
                 INCLUDE_DIRS include hlslib/include ${CMAKE_BINARY_DIR}
                 HLS_FLAGS ${CMAKE_CXX_FLAGS}
                 HLS_CONFIG "config_compile -pipeline_style frp\nconfig_dataflow -fifo_depth 16"
                 DEPENDS ${APFP_INCLUDES} include/Microbenchmark.h
                 PORT_MAPPING ${APFP_MICROBENCHMARK_PORT_MAPPING}
                 SLR_MAPPING ${APFP_MICROBENCHMARK_SLR_MAPPING})
add_vitis_program(Microbenchmark ${APFP_PLATFORM}
                  CLOCK ${APFP_FREQUENCY}
                  PROFILING ${APFP_PROFILING}
                  DEBUGGING ${APFP_DEBUGGING}
                  SAVE_TEMPS ${APFP_SAVE_TEMPS})

# Internal library 
add_library(apfp host/Random.cpp host/MatrixMultiplicationReference.cpp host/MicrobenchmarkReference.cpp)
target_link_libraries(apfp ${GMP_LIBRARIES} ${MPFR_LIBRARIES})

# Library necessary to run in simulation mode
add_library(simulation
            device/Karatsuba.cpp
            device/ArithmeticOperations.cpp
            device/MatrixMultiplication.cpp 
            device/Microbenchmark.cpp)
target_compile_options(simulation PRIVATE -DAP_INT_MAX_W=${APFP_MAX_BITS})
target_link_libraries(simulation ${CMAKE_THREAD_LIBS_INIT})

add_library(ApfpHostlib SHARED interface/Apfp.cpp)
target_link_libraries(ApfpHostlib ${Vitis_LIBRARIES} ${GMP_LIBRARIES}) 
target_compile_definitions(ApfpHostlib PRIVATE HLSLIB_SIMULATE_OPENCL)

# Executables used to run in simulation mode, calling kernels as a C++ function directly
add_executable(TestMatrixMultiplicationSimulation host/TestMatrixMultiplication.cpp)
target_link_libraries(TestMatrixMultiplicationSimulation apfp simulation ${Vitis_LIBRARIES} ${GMP_LIBRARIES} ${MPFR_LIBRARIES}) 
target_compile_definitions(TestMatrixMultiplicationSimulation PRIVATE HLSLIB_SIMULATE_OPENCL)
add_executable(MicrobenchmarkSimulation host/Microbenchmark.cpp)
target_link_libraries(MicrobenchmarkSimulation apfp simulation ${Vitis_LIBRARIES} ${GMP_LIBRARIES} ${MPFR_LIBRARIES}) 
target_compile_definitions(MicrobenchmarkSimulation PRIVATE HLSLIB_SIMULATE_OPENCL)

# Executables used to run from an xclbin binary
add_executable(TestMatrixMultiplicationHardware host/TestMatrixMultiplication.cpp)
target_link_libraries(TestMatrixMultiplicationHardware apfp simulation ${Vitis_LIBRARIES} ${GMP_LIBRARIES} ${MPFR_LIBRARIES}) 
add_executable(MicrobenchmarkHardware host/Microbenchmark.cpp)
target_link_libraries(MicrobenchmarkHardware apfp simulation ${Vitis_LIBRARIES} ${GMP_LIBRARIES} ${MPFR_LIBRARIES}) 

# Testing
enable_testing()
add_test(TestMatrixMultiplication_OneTile TestMatrixMultiplicationSimulation ${APFP_TILE_SIZE_N} 2 ${APFP_TILE_SIZE_M})
math(EXPR APFP_TEST_SIZE_N "${APFP_TILE_SIZE_N} / 2") 
math(EXPR APFP_TEST_SIZE_M "${APFP_TILE_SIZE_M} / 2") 
add_test(TestMatrixMultiplication_LessThanOneTile TestMatrixMultiplicationSimulation ${APFP_TEST_SIZE_N} 2 ${APFP_TEST_SIZE_M})
math(EXPR APFP_TEST_SIZE_N "${APFP_TILE_SIZE_N} + 1") 
math(EXPR APFP_TEST_SIZE_M "${APFP_TILE_SIZE_M} + 1") 
add_test(TestMatrixMultiplication_MultipleTiles TestMatrixMultiplicationSimulation ${APFP_TEST_SIZE_N} 2 ${APFP_TEST_SIZE_M})
add_test(MicrobenchmarkSimulation MicrobenchmarkSimulation 129)
add_library(Catch host/Catch.cpp)
add_executable(UnitTests host/UnitTests.cpp)
target_link_libraries(UnitTests Catch ${GMP_LIBRARIES} ${MPFR_LIBRARIES} apfp simulation)
add_test(UnitTests UnitTests)

install(TARGETS ApfpHostlib)

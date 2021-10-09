cmake_minimum_required(VERSION 3.0)
project(apfp)

set(CMAKE_CXX_STANDARD 17)

# Target options 
set(APFP_PLATFORM "xilinx_u280_xdma_201920_3" CACHE STRING "Platform string for Vitis.")
set(APFP_BITS 1024 CACHE STRING "Number of bits to use for a floating point number, including mantissa, exponent, and sign.")
set(APFP_MULT_BASE_BITS 16 CACHE STRING "Number of bits to bottom out the multiplication at and use native multiplication.")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/hlslib/cmake ${CMAKE_SOURCE_DIR}/cmake)

find_package(Vitis REQUIRED)
find_package(GMP REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

configure_file(include/Config.h.in Config.h)

set(APFP_KERNEL_FILES device/MatrixMultiplication.cpp device/Karatsuba.cpp)

# Setup FPGA kernel targets
add_vitis_kernel(MatrixMultiplication FILES ${APFP_KERNEL_FILES}
                 INCLUDE_DIRS include hlslib/include ${CMAKE_BINARY_DIR}
                 HLS_FLAGS "-DAP_INT_MAX_W=${APFP_BITS}"
                 HLS_CONFIG "config_compile -pipeline_style frp\nconfig_dataflow -fifo_depth 16"
                 DEPENDS ${CMAKE_BINARY_DIR}/Config.h
                         include/MatrixMultiplication.h
                         include/Karatsuba.h)
add_vitis_program(MatrixMultiplication ${APFP_PLATFORM})

include_directories(${CMAKE_BINARY_DIR} include SYSTEM hlslib/include ${Vitis_INCLUDE_DIRS} )

add_library(simulation ${APFP_KERNEL_FILES})
target_compile_options(simulation PRIVATE -Wno-unknown-pragmas -DAP_INT_MAX_W=${APFP_BITS})
target_link_libraries(simulation ${CMAKE_THREAD_LIBS_INIT})

add_library(Apfp interface/Apfp.cpp)
target_link_libraries(Apfp ${Vitis_LIBRARIES} ${GMP_LIBRARIES}) 
target_compile_definitions(Apfp PRIVATE HLSLIB_SIMULATE_OPENCL)

add_executable(TestSimulation host/TestProgram.cpp)
target_link_libraries(TestSimulation simulation ${Vitis_LIBRARIES} ${GMP_LIBRARIES}) 
target_compile_definitions(TestSimulation PRIVATE HLSLIB_SIMULATE_OPENCL)

add_executable(TestHardware host/TestProgram.cpp)
target_link_libraries(TestHardware simulation ${Vitis_LIBRARIES} ${GMP_LIBRARIES}) 

enable_testing()
add_test(TestSimulation TestSimulation 4 4 4)
add_library(Catch host/Catch.cpp)
add_executable(UnitTests host/UnitTests.cpp)
target_link_libraries(UnitTests Catch ${GMP_LIBRARIES} simulation)
add_test(UnitTests UnitTests)

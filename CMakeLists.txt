cmake_minimum_required(VERSION 3.0)
project(apfp)

# Target options 
set(APFP_PLATFORM "xilinx_u280_xdma_201920_3" CACHE STRING "Platform string for Vitis.")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/hlslib/cmake)

find_package(Vitis REQUIRED)

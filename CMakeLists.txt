# Setup project

cmake_minimum_required(VERSION 3.23.1)
project(kernel LANGUAGES CXX CUDA)


# Build dependencies

include_directories(${Python3_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
find_package(CUDAToolkit 11.5 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)


# Environment

set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23")


# Target library

python3_add_library(kernel MODULE src/kernel.cu WITH_SOABI)
set_target_properties(kernel PROPERTIES PREFIX "" SUFFIX ".so")
target_link_libraries(kernel PRIVATE pybind11::headers CUDA::cudart_static)
install(TARGETS kernel LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

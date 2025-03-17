cmake_minimum_required(VERSION 3.23.1)
project(stargaze LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit 11.5 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS
	Interpreter
	Development)

set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23")

add_subdirectory(src/pybind11)
include_directories(${Python3_INCLUDE_DIRS})

add_library(stargaze MODULE
	src/stargaze.cu)

target_link_libraries(stargaze PRIVATE
	CUDA::cudart)
set_target_properties(stargaze PROPERTIES
	SUFFIX ".so"
	PREFIX "")
target_compile_options(stargaze PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
	--expt-relaxed-constexpr
	>)

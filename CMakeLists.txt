#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

cmake_minimum_required(VERSION 3.28)

# set the project name
project(Sprux CXX)
set(CMAKE_CXX_STANDARD 17)
set(SPRUX_CXX_FLAGS -Wall -Wextra -pedantic) # -O1 -g -fsanitize=address -static-libsan)

message("* Build type: " ${CMAKE_BUILD_TYPE})
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/BundleStaticLibrary.cmake")

# Add cmake module path for custom find modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# general settings
set(SPRUX_BUILD_TESTS ON CACHE BOOL "If on (default), tests are build")
set(SPRUX_BUILD_EXAMPLES ON CACHE BOOL "If on (default), exampels/benchmarks are build")

# some fancy colors for cmake messages
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(Red "${Esc}[31m")
  set(Green "${Esc}[32m")
  set(Yellow "${Esc}[33m")
  set(Blue "${Esc}[34m")
  set(Magenta "${Esc}[35m")
  set(Cyan "${Esc}[36m")
  set(White "${Esc}[37m")
endif()

# CUDA
set(SPRUX_USE_CUBLAS ON CACHE BOOL "If on, CUDA support is enabled")
set(SPRUX_CUDA_ARCHS "detect" CACHE STRING "List of cuda architecture (eg '60;62;75'), or 'detect', or 'torch'")

if(SPRUX_USE_CUBLAS)
  message("${Cyan}==============================[ CUDA ]==================================${ColourReset}")
  enable_language(CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  if(SPRUX_CUDA_ARCHS STREQUAL "detect")
    message("* Detecting CUDA architecture...")
    set(CUDA_DETECT_BIN "${CMAKE_BINARY_DIR}/detect_cuda")
    set(CUDA_DETECT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/cmake/detect_cuda.cu")

    message("${Yellow}  ${CMAKE_CUDA_COMPILER} ${CUDA_DETECT_SRC} -o ${CUDA_DETECT_BIN}${ColourReset}")
    execute_process(COMMAND "${CMAKE_CUDA_COMPILER}" "${CUDA_DETECT_SRC}" -o "${CUDA_DETECT_BIN}"
      OUTPUT_VARIABLE NVCC_OUT
      ERROR_VARIABLE NVCC_ERR)

    if(NOT NVCC_OUT STREQUAL "")
      message("${NVCC_OUT}")
    endif()

    if(NOT NVCC_ERR STREQUAL "")
      message("${Red}${NVCC_ERR}${ColourReset}")
    endif()

    execute_process(COMMAND "${CUDA_DETECT_BIN}"
      RESULT_VARIABLE CUDA_RETURN_CODE
      OUTPUT_VARIABLE RESULT)

    if(CUDA_RETURN_CODE EQUAL 0)
      message("* CUDA architectures: '${RESULT}' (autodetected installed GPU)")
      set(SPRUX_CUDA_ARCHITECTURES "${RESULT}")
    else()
      message(FATAL_ERROR "CUDA architecture detection failed: ${RESULT}")
    endif()
  elseif(SPRUX_CUDA_ARCHS STREQUAL "torch")
    message("* Getting Torch CUDA architectures...")
    set(CUDA_GET_TORCH_ARCHS_BIN "${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_torch_cuda_archs.py")
    message("${Yellow}  python ${CUDA_GET_TORCH_ARCHS_BIN}${ColourReset}")
    execute_process(COMMAND "python" "${CUDA_GET_TORCH_ARCHS_BIN}"
      RESULT_VARIABLE TORCH_RETURN_CODE
      OUTPUT_VARIABLE RESULT
      ERROR_VARIABLE TORCH_ERROR)

    if(NOT TORCH_ERROR STREQUAL "")
      message("${Red}${TORCH_ERROR}${ColourReset}")
    endif()

    if(TORCH_RETURN_CODE EQUAL 0)
      message("* CUDA architectures: '${RESULT}' (from torch)")

      if(RESULT STREQUAL "")
        message(FATAL_ERROR "Torch architectures returned an empty result.")
      endif()

      set(SPRUX_CUDA_ARCHITECTURES "${RESULT}")
    else()
      message(FATAL_ERROR "Torch CUDA architecture detection failed: ${RESULT}")
    endif()
  else()
    # https://cmake.org/cmake/help/git-stage/prop_tgt/CUDA_ARCHITECTURES.html
    message("* CUDA architectures: '${SPRUX_CUDA_ARCHS}' (manually set)")
    set(SPRUX_CUDA_ARCHITECTURES "${SPRUX_CUDA_ARCHS}")
  endif()

  list(APPEND CUDA_NVCC_FLAGS
    -Xcompiler -Wall,-Wextra,-Werror,-Wno-unknown-pragmas
    --expt-relaxed-constexpr)
  message("* CUDA Includes : " ${CUDAToolkit_INCLUDE_DIRS})
  message("* CUDA Libraries: " ${CUDAToolkit_LIBRARY_DIR})
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
  add_compile_definitions(SPRUX_USE_CUBLAS)
endif()

# METAL (macOS/iOS GPU backend)
set(SPRUX_USE_METAL OFF CACHE BOOL "If on, Metal support is enabled (macOS/iOS only)")

if(SPRUX_USE_METAL)
  message("${Cyan}==============================[ Metal ]==================================${ColourReset}")

  if(NOT APPLE)
    message(FATAL_ERROR "Metal backend is only supported on macOS/iOS")
  endif()

  # Find Metal frameworks
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

  message("* Metal Framework: ${METAL_FRAMEWORK}")
  message("* MPS Framework: ${MPS_FRAMEWORK}")
  message("* Foundation Framework: ${FOUNDATION_FRAMEWORK}")

  # Note: We do NOT call enable_language(OBJCXX) here because it causes
  # cmake to treat ALL .m files as Objective-C++, which breaks projects
  # that embed Sprux and have their own Objective-C code (like IREE).
  # Instead, we explicitly set compile flags for .mm files in the
  # baspacho/baspacho/CMakeLists.txt.

  add_compile_definitions(SPRUX_USE_METAL)
endif()

# OpenCL (with CLBlast for BLAS operations)
set(SPRUX_USE_OPENCL OFF CACHE BOOL "If on, OpenCL support is enabled (with CLBlast)")

if(SPRUX_USE_OPENCL)
  message("${Cyan}==============================[ OpenCL ]=================================${ColourReset}")

  find_package(OpenCL REQUIRED)
  message("* OpenCL Version: ${OpenCL_VERSION_STRING}")
  message("* OpenCL Include: ${OpenCL_INCLUDE_DIRS}")
  message("* OpenCL Libraries: ${OpenCL_LIBRARIES}")

  # CLBlast provides high-performance BLAS for OpenCL
  find_package(CLBlast REQUIRED)
  message("* CLBlast Libraries: ${CLBLAST_LIBRARIES}")

  include_directories(${OpenCL_INCLUDE_DIRS})
  add_compile_definitions(SPRUX_USE_OPENCL)
endif()

# BLAS. a few possibilities are:
# * ATLAS
# * OpenBLAS
# * Intel10_64{i}lp{_seq}
# (if 'i' indices are 64bits, if '_seq' sequential ie single thread)
set(SPRUX_USE_BLAS ON CACHE BOOL "If on, BLAS support is enabled")

if(SPRUX_USE_BLAS)
  message("${Cyan}==============================[ BLAS ]==================================${ColourReset}")
  set(BLA_VENDOR OpenBLAS CACHE STRING
    "BLAS Library to be searched, possibilities are: ATLAS, OpenBLAS, Intel10_64ilp_seq, etc")
  set(BLA_STATIC OFF CACHE STRING
    "Link statically to BLAS Library")
  message("* BLA_VENDOR = " ${BLA_VENDOR})
  message("* BLA_STATIC = " ${BLA_STATIC})
  find_package(BLAS REQUIRED)
  message("* BLAS linkerFlags = " ${BLAS_LINKER_FLAGS})
  message("* BLAS Libraries = " ${BLAS_LIBRARIES})

  if(BLA_VENDOR MATCHES "Intel10_*")
    message("* BLAS appears to be MKL, MKL_ROOT should be set")
    message("* Adding includes: " $ENV{MKLROOT}/include)
    include_directories($ENV{MKLROOT}/include) # if MKL
    add_compile_definitions(SPRUX_USE_MKL)
  endif()

  add_compile_definitions(SPRUX_USE_BLAS)
endif()

# use fetchcontent to download dependencies (needed by SuiteSparse and others below)
include(FetchContent)
cmake_policy(SET CMP0077 NEW)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30")
  cmake_policy(SET CMP0169 OLD) # do not complain about FetchContent_Populate
endif()

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
  cmake_policy(SET CMP0135 NEW)
endif()

# SuiteSparse (vendored via FetchContent)
# Provides: AMD (fill-reducing ordering), CHOLMOD (fast symbolic analysis)
message("${Cyan}===========================[ SuiteSparse ]==============================${ColourReset}")
set(SPRUX_USE_SYSTEM_SUITESPARSE OFF CACHE BOOL
  "If on, use system-installed SuiteSparse instead of vendored")

if(SPRUX_USE_SYSTEM_SUITESPARSE)
  # System SuiteSparse: find AMD, CHOLMOD, BTF via pkg-config or cmake
  find_path(SUITESPARSE_AMD_INCLUDE_DIR NAMES amd.h
    PATHS $ENV{AMDDIR} ${INCLUDE_INSTALL_DIR}
    PATH_SUFFIXES suitesparse ufsparse
    REQUIRED)
  find_library(SUITESPARSE_AMD_LIBRARIES amd
    PATHS $ENV{AMDDIR} ${LIB_INSTALL_DIR}
    REQUIRED)
  find_path(CHOLMOD_INCLUDE_DIR NAMES cholmod.h
    PATHS ${INCLUDE_INSTALL_DIR}
    PATH_SUFFIXES suitesparse ufsparse
    REQUIRED)
  find_library(CHOLMOD_LIBRARIES cholmod
    PATHS ${LIB_INSTALL_DIR}
    REQUIRED)
  include_directories(${SUITESPARSE_AMD_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR})
  message("* Using system SuiteSparse")
  message("* AMD Include = " ${SUITESPARSE_AMD_INCLUDE_DIR})
  message("* CHOLMOD Include = " ${CHOLMOD_INCLUDE_DIR})

  set(SUITESPARSE_AMD_TARGET ${SUITESPARSE_AMD_LIBRARIES})
  set(SUITESPARSE_CHOLMOD_TARGET ${CHOLMOD_LIBRARIES})
else()
  # Vendored SuiteSparse via FetchContent - build only what we need
  if(NOT DEFINED suitesparse_SOURCE_DIR)
    set(SUITESPARSE_ENABLE_PROJECTS "suitesparse_config;amd;colamd;camd;ccolamd;cholmod;btf" CACHE STRING "" FORCE)
    set(CHOLMOD_SUPERNODAL OFF CACHE BOOL "" FORCE)  # No BLAS needed for symbolic-only
    set(CHOLMOD_CAMD ON CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS_SAVED ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)  # Build SuiteSparse as static
    set(SUITESPARSE_USE_OPENMP OFF CACHE BOOL "" FORCE)
    set(SUITESPARSE_USE_CUDA OFF CACHE BOOL "" FORCE)
    set(SUITESPARSE_DEMOS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      suitesparse
      URL https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v7.8.3.zip
    )
    FetchContent_MakeAvailable(suitesparse)
    set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_SAVED})
  else()
    message("${Cyan}Using SuiteSparse from parent project:${ColourReset}")
  endif()

  message("* SuiteSparse Source = " ${suitesparse_SOURCE_DIR})

  # SuiteSparse creates targets like SuiteSparse::AMD, SuiteSparse::CHOLMOD, etc.
  # For static builds, the targets may be AMD_static, CHOLMOD_static, etc.
  if(TARGET SuiteSparse::AMD)
    set(SUITESPARSE_AMD_TARGET SuiteSparse::AMD)
  elseif(TARGET AMD_static)
    set(SUITESPARSE_AMD_TARGET AMD_static)
  elseif(TARGET AMD)
    set(SUITESPARSE_AMD_TARGET AMD)
  else()
    message(FATAL_ERROR "Could not find AMD target from SuiteSparse")
  endif()

  if(TARGET SuiteSparse::CHOLMOD)
    set(SUITESPARSE_CHOLMOD_TARGET SuiteSparse::CHOLMOD)
  elseif(TARGET CHOLMOD_static)
    set(SUITESPARSE_CHOLMOD_TARGET CHOLMOD_static)
  elseif(TARGET CHOLMOD)
    set(SUITESPARSE_CHOLMOD_TARGET CHOLMOD)
  else()
    message(FATAL_ERROR "Could not find CHOLMOD target from SuiteSparse")
  endif()

  if(TARGET SuiteSparse::BTF)
    set(SUITESPARSE_BTF_TARGET SuiteSparse::BTF)
  elseif(TARGET BTF_static)
    set(SUITESPARSE_BTF_TARGET BTF_static)
  elseif(TARGET BTF)
    set(SUITESPARSE_BTF_TARGET BTF)
  else()
    message(FATAL_ERROR "Could not find BTF target from SuiteSparse")
  endif()
endif()

# AMD from SuiteSparse is now always available (vendored or system)
set(HAVE_SUITESPARSE_AMD 1)
add_compile_definitions(SPRUX_USE_SUITESPARSE_AMD)
message("* Using SuiteSparse AMD for fill-reducing ordering")

# CHOLMOD is now always available
set(HAVE_CHOLMOD 1)
add_compile_definitions(SPRUX_HAVE_CHOLMOD)
message("* Using CHOLMOD for fast symbolic analysis")

# BTF is now always available (for MC64 preprocessing)
set(HAVE_BTF 1)
add_compile_definitions(SPRUX_HAVE_BTF)
message("* Using BTF for maximum transversal (LU preprocessing)")

if(SPRUX_BUILD_EXAMPLES)

  # cuDSS (for GPU sparse direct solver comparison)
  if(SPRUX_USE_CUBLAS)
    message("${Cyan}==============================[ cuDSS ]=================================${ColourReset}")
    find_package(cuDSS)

    if(cuDSS_FOUND)
      set(HAVE_CUDSS 1)
      message("* cuDSS Include = " ${CUDSS_INCLUDE_DIRS})
      message("* cuDSS Libraries = " ${CUDSS_LIBRARIES})
      add_compile_definitions(SPRUX_HAVE_CUDSS)
    else()
      set(HAVE_CUDSS 0)
      message("* cuDSS not found! cuDSS benchmarking comparison will not be available")
    endif()
  endif()

  # UMFPACK (for LU benchmarking comparison)
  message("${Cyan}============================[ UMFPACK ]=================================${ColourReset}")
  find_path(UMFPACK_INCLUDE_DIR NAMES umfpack.h
    PATHS ${INCLUDE_INSTALL_DIR}
    PATH_SUFFIXES suitesparse ufsparse)
  find_library(UMFPACK_LIBRARIES umfpack
    PATHS ${LIB_INSTALL_DIR})

  if((UMFPACK_INCLUDE_DIR STREQUAL "UMFPACK_INCLUDE_DIR-NOTFOUND")
    OR(UMFPACK_LIBRARIES STREQUAL "UMFPACK_LIBRARIES-NOTFOUND"))
    set(HAVE_UMFPACK 0)
    message("* UMFPACK not found! LU benchmarking comparison will not be available")
  else()
    set(HAVE_UMFPACK 1)
    message("* UMFPACK Source = " ${UMFPACK_INCLUDE_DIR})
    message("* UMFPACK Libraries = " ${UMFPACK_LIBRARIES})
    add_compile_definitions(SPRUX_HAVE_UMFPACK)
    include_directories(${UMFPACK_INCLUDE_DIR})
  endif()
endif()

# dispenso for multithreading
# currently in facebookincubator, soon to be opensourced
if(NOT DEFINED dispenso_SOURCE_DIR)
  message("${Cyan}============================[ dispenso ]================================${ColourReset}")
  FetchContent_Declare(
    dispenso
    URL https://github.com/facebookincubator/dispenso/archive/refs/tags/v1.4.0.zip
  )
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  FetchContent_MakeAvailable(dispenso)
  unset(CMAKE_POSITION_INDEPENDENT_CODE)
else()
  message("${Cyan}Using Dispenso from parent project:${ColourReset}")
endif()

include_directories(${dispenso_SOURCE_DIR} ${dispenso_SOURCE_DIR}/dispenso/third-party/moodycamel)
message("* Dispenso Source = " ${dispenso_SOURCE_DIR})
message("* Dispenso Library Dir = " ${dispenso_BINARY_DIR})

# gtest
# NOTE: Check for target existence, not just gtest_SOURCE_DIR variable.
# SuiteSparse FetchContent can set gtest_SOURCE_DIR as a cache variable
# without actually creating the gtest targets.
if(SPRUX_BUILD_TESTS)
  if(NOT TARGET gtest_main)
    message("${Cyan}=============================[ GTest ]==================================${ColourReset}")
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  else()
    message("${Cyan}Using GoogleTest from parent project:${ColourReset}")
  endif()

  message("* GTest Source = " ${gtest_SOURCE_DIR})
  message("* GTest Library Dir = " ${gtest_BINARY_DIR})
  message("* GMock Source = " ${gmock_SOURCE_DIR})
  message("* GMock Library Dir = " ${gmock_BINARY_DIR})
  include(GoogleTest)
endif()

# eigen
if(NOT DEFINED eigen_SOURCE_DIR)
  message("${Cyan}=============================[ Eigen ]==================================${ColourReset}")
  FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
    EXCLUDE_FROM_ALL
  )
  FetchContent_GetProperties(eigen)

  if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
  endif()
else()
  message("${Cyan}Using Eigen from parent project:${ColourReset}")
endif()

message("* Eigen Source = " ${eigen_SOURCE_DIR})
message("* Eigen Library Dir = " ${eigen_BINARY_DIR})
include_directories(${eigen_SOURCE_DIR})

# Sophus (only needed for BA problem in examples/benchmarks)
if(SPRUX_BUILD_EXAMPLES)
  if(NOT DEFINED sophus_SOURCE_DIR)
    message("${Cyan}=============================[ Sophus ]=================================${ColourReset}")
    FetchContent_Declare(
      sophus
      URL https://github.com/strasdat/Sophus/archive/refs/tags/1.24.6.zip
      EXCLUDE_FROM_ALL
    )
    FetchContent_GetProperties(sophus)

    if(NOT sophus_POPULATED)
      FetchContent_Populate(sophus)
    endif()
  else()
    message("${Cyan}Using Sophus from parent project:${ColourReset}")
  endif()

  message("* Sophus Source = " ${sophus_SOURCE_DIR})
  message("* Sophus Library Dir = " ${sophus_BINARY_DIR})
  include_directories(${sophus_SOURCE_DIR})
  add_compile_definitions(SOPHUS_USE_BASIC_LOGGING=1)
endif()

# add root to include path
include_directories("${PROJECT_SOURCE_DIR}")

# basic lib name depends on shared/static, so that
# `Sprux` is the bundled library if static
if(BUILD_SHARED_LIBS)
  set(SPRUX_LIBRARY Sprux)
else()
  set(SPRUX_LIBRARY Sprux_static)
endif()

# unit tests
if(SPRUX_BUILD_TESTS)
  enable_testing()

  function(add_sprux_test TARGET)
    add_executable(
      "${TARGET}"
      "${ARGN}"
    )
    target_link_libraries(
      "${TARGET}" ${SPRUX_LIBRARY} testing gtest_main gmock
    )
    gtest_discover_tests("${TARGET}")
  endfunction()
endif()

# add subfolders
add_subdirectory(baspacho)

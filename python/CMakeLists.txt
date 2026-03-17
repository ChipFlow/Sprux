# Copyright (c) Robert Taylor, 2026. All rights reserved.
# Licensed under the MIT license found in the LICENSE file.
#
# Sprux Python Bindings
#
# This builds a Python extension module using pybind11 that provides
# access to Sprux's sparse LU solver from Python.

cmake_minimum_required(VERSION 3.18)

# Find pybind11
find_package(pybind11 REQUIRED)

# Create the Python module
pybind11_add_module(sprux_py sprux_bindings.cpp)

# Link against Sprux library
target_link_libraries(sprux_py PRIVATE ${SPRUX_LIBRARY})

# Include directories
target_include_directories(sprux_py PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/sprux
)

# Propagate backend definitions
if(SPRUX_USE_METAL)
    target_compile_definitions(sprux_py PRIVATE SPRUX_USE_METAL)
endif()

if(SPRUX_USE_CUDA)
    target_compile_definitions(sprux_py PRIVATE SPRUX_USE_CUDA)
endif()

if(SPRUX_USE_OPENCL)
    target_compile_definitions(sprux_py PRIVATE SPRUX_USE_OPENCL)
endif()

# Install the module
install(TARGETS sprux_py
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

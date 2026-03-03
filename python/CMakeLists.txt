# BaSpaCho Python Bindings
#
# This builds a Python extension module using pybind11 that provides
# access to BaSpaCho's sparse LU solver from Python.

cmake_minimum_required(VERSION 3.18)

# Find pybind11
find_package(pybind11 REQUIRED)

# Create the Python module
pybind11_add_module(baspacho_py baspacho_bindings.cpp)

# Link against baspacho library
target_link_libraries(baspacho_py PRIVATE baspacho)

# Include directories
target_include_directories(baspacho_py PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/baspacho
)

# Propagate backend definitions
if(BASPACHO_USE_METAL)
    target_compile_definitions(baspacho_py PRIVATE BASPACHO_USE_METAL)
endif()

if(BASPACHO_USE_CUDA)
    target_compile_definitions(baspacho_py PRIVATE BASPACHO_USE_CUDA)
endif()

if(BASPACHO_USE_OPENCL)
    target_compile_definitions(baspacho_py PRIVATE BASPACHO_USE_OPENCL)
endif()

# Install the module
install(TARGETS baspacho_py
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

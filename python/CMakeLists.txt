# Copyright (c) Robert Taylor, 2026. All rights reserved.
# Licensed under the MIT license found in the LICENSE file.
#
# Sprux Python Bindings
#
# This builds a Python extension module using pybind11 that provides
# access to Sprux's sparse direct solver from Python.

# Create the Python module (pybind11::pybind11 is provided by parent CMakeLists.txt)
pybind11_add_module(sprux_py sprux_bindings.cpp)

# Link against Sprux library
target_link_libraries(sprux_py PRIVATE ${SPRUX_LIBRARY})

# Include directories
target_include_directories(sprux_py PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/sprux
)

# Propagate backend compile definitions (passed from parent CMakeLists.txt)
if(_SPRUX_PY_DEFS)
    target_compile_definitions(sprux_py PRIVATE ${_SPRUX_PY_DEFS})
endif()

# Install the module
install(TARGETS sprux_py
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

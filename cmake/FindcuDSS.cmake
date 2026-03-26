#
# FindcuDSS.cmake - Find NVIDIA cuDSS (CUDA Direct Sparse Solver) library
#
# This module finds the cuDSS library, which is distributed separately from
# the CUDA Toolkit.
#
# Search order:
#   1. CUDSS_DIR cmake variable or environment variable
#   2. CUDA Toolkit paths (CUDAToolkit_LIBRARY_DIR, etc.)
#   3. Standard system paths
#
# Result variables:
#   CUDSS_FOUND        - True if cuDSS was found
#   CUDSS_INCLUDE_DIRS - cuDSS include directories
#   CUDSS_LIBRARIES    - cuDSS libraries to link
#
# Imported targets:
#   cuDSS::cuDSS       - cuDSS library target
#

# Build search paths from CUDSS_DIR (cmake var or env var)
set(_cudss_search_paths "")
if(CUDSS_DIR)
  list(APPEND _cudss_search_paths "${CUDSS_DIR}")
endif()
if(DEFINED ENV{CUDSS_DIR})
  list(APPEND _cudss_search_paths "$ENV{CUDSS_DIR}")
endif()

# Also check CUDA toolkit paths as fallback
if(CUDAToolkit_LIBRARY_DIR)
  get_filename_component(_cuda_root "${CUDAToolkit_LIBRARY_DIR}" DIRECTORY)
  list(APPEND _cudss_search_paths "${_cuda_root}")
endif()

find_path(CUDSS_INCLUDE_DIR
  NAMES cudss.h
  PATHS ${_cudss_search_paths}
  PATH_SUFFIXES include
)

find_library(CUDSS_LIBRARY
  NAMES cudss
  PATHS ${_cudss_search_paths}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuDSS
  REQUIRED_VARS CUDSS_LIBRARY CUDSS_INCLUDE_DIR
)

if(cuDSS_FOUND)
  set(CUDSS_INCLUDE_DIRS "${CUDSS_INCLUDE_DIR}")
  set(CUDSS_LIBRARIES "${CUDSS_LIBRARY}")

  if(NOT TARGET cuDSS::cuDSS)
    add_library(cuDSS::cuDSS UNKNOWN IMPORTED)
    set_target_properties(cuDSS::cuDSS PROPERTIES
      IMPORTED_LOCATION "${CUDSS_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CUDSS_INCLUDE_DIR}"
    )
  endif()
endif()

mark_as_advanced(CUDSS_INCLUDE_DIR CUDSS_LIBRARY)


set(Sprux_sources
    CoalescedBlockMatrix.cpp
    ComputationModel.cpp
    CsrSolver.cpp
    EliminationTree.cpp
    MatOpsFast.cpp
    MatOpsRef.cpp
    Preprocessing.cpp
    Solver.cpp
    SparseStructure.cpp
    SupernodeMerger.cpp
    Utils.cpp)

if(SPRUX_USE_CUBLAS)
    # separate recent and legacy cuda archs (which need atomicAdd's workaround)
    set(SPRUX_LEGACY_CUDA_ARCHITECTURES "")
    set(SPRUX_RECENT_CUDA_ARCHITECTURES "")
    foreach(ARCH IN LISTS SPRUX_CUDA_ARCHITECTURES)
        if(ARCH LESS 60)
            list(APPEND SPRUX_LEGACY_CUDA_ARCHITECTURES "${ARCH}")
        else()
            list(APPEND SPRUX_RECENT_CUDA_ARCHITECTURES "${ARCH}")
        endif()
    endforeach()

    list(APPEND Sprux_sources CublasError.cpp)
    if(SPRUX_RECENT_CUDA_ARCHITECTURES)
        list(APPEND Sprux_sources MatOpsCuda.cu)
    endif()
endif()

if(SPRUX_USE_METAL)
    # Metal backend uses Objective-C++ (.mm files)
    # We compile these as CXX with explicit -x objective-c++ flag
    # instead of using enable_language(OBJCXX), which would affect
    # the entire project and break embedding in projects with .m files.
    set(SPRUX_METAL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/MetalDefs.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/MatOpsMetal.mm)

    # Set .mm files to be treated as CXX with Objective-C++ mode
    set_source_files_properties(${SPRUX_METAL_SOURCES}
        PROPERTIES
        LANGUAGE CXX
        COMPILE_FLAGS "-x objective-c++ -std=c++17")

    list(APPEND Sprux_sources
         MetalDefs.mm
         MatOpsMetal.mm)
endif()

if(SPRUX_USE_OPENCL)
    list(APPEND Sprux_sources
         OpenCLDefs.cpp
         MatOpsOpenCL.cpp)
endif()

add_library(${SPRUX_LIBRARY}  ${Sprux_sources})
set_property(TARGET ${SPRUX_LIBRARY} PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${SPRUX_LIBRARY}
                      dispenso
                      ${BLAS_LIBRARIES})

if(SPRUX_USE_CUBLAS)
    # if compiling for legacy, create a target for Cuda code with enabled workaround
    if(SPRUX_LEGACY_CUDA_ARCHITECTURES)
        add_library(Sprux_legacy_cuda_archs OBJECT MatOpsCuda.cu)
        set_target_properties(Sprux_legacy_cuda_archs PROPERTIES CUDA_ARCHITECTURES "${SPRUX_LEGACY_CUDA_ARCHITECTURES}")
        target_compile_options(Sprux_legacy_cuda_archs PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${SPRUX_CXX_FLAGS}>)
        target_compile_options(Sprux_legacy_cuda_archs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
        target_compile_definitions(Sprux_legacy_cuda_archs PRIVATE CUDA_DOUBLE_ATOMIC_ADD_WORKAROUND)
        target_link_libraries(${SPRUX_LIBRARY} Sprux_legacy_cuda_archs)
    endif()

    target_link_libraries(${SPRUX_LIBRARY}
                          CUDA::cudart
                          CUDA::cublas
                          CUDA::cusolver)
    if(SPRUX_RECENT_CUDA_ARCHITECTURES)
        set_target_properties(${SPRUX_LIBRARY} PROPERTIES CUDA_ARCHITECTURES "${SPRUX_RECENT_CUDA_ARCHITECTURES}")
    endif()
    target_compile_options(${SPRUX_LIBRARY} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
endif()

if(SPRUX_USE_METAL)
    # Compile Metal shaders to metallib
    set(METAL_SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/MetalKernels.metal")
    set(METAL_SHADER_AIR "${CMAKE_CURRENT_BINARY_DIR}/MetalKernels.air")
    set(METAL_SHADER_LIB "${CMAKE_CURRENT_BINARY_DIR}/MetalKernels.metallib")

    add_custom_command(
        OUTPUT "${METAL_SHADER_AIR}"
        COMMAND xcrun -sdk macosx metal -c "${METAL_SHADER_SOURCE}" -o "${METAL_SHADER_AIR}" -std=macos-metal2.4
        DEPENDS "${METAL_SHADER_SOURCE}"
        COMMENT "Compiling Metal shaders to AIR"
    )

    add_custom_command(
        OUTPUT "${METAL_SHADER_LIB}"
        COMMAND xcrun -sdk macosx metallib "${METAL_SHADER_AIR}" -o "${METAL_SHADER_LIB}"
        DEPENDS "${METAL_SHADER_AIR}"
        COMMENT "Linking Metal shader library"
    )

    add_custom_target(metal_shaders ALL DEPENDS "${METAL_SHADER_LIB}")
    add_dependencies(${SPRUX_LIBRARY} metal_shaders)

    # Link Metal frameworks
    target_link_libraries(${SPRUX_LIBRARY}
                          ${METAL_FRAMEWORK}
                          ${MPS_FRAMEWORK}
                          ${FOUNDATION_FRAMEWORK})

    # Add ARC flag to the Metal source files
    # Note: -x objective-c++ and -std=c++17 are already set above
    set_source_files_properties(${SPRUX_METAL_SOURCES}
        PROPERTIES COMPILE_FLAGS "-x objective-c++ -std=c++17 -fobjc-arc")

    # Define path to compiled metallib for runtime loading
    target_compile_definitions(${SPRUX_LIBRARY} PRIVATE
        SPRUX_METAL_LIBRARY_PATH="${METAL_SHADER_LIB}")
endif()

if(SPRUX_USE_OPENCL)
    # Link OpenCL and CLBlast
    target_link_libraries(${SPRUX_LIBRARY}
                          ${OpenCL_LIBRARIES}
                          ${CLBlast_LIBRARIES})
    target_include_directories(${SPRUX_LIBRARY} PRIVATE
                               ${CLBlast_INCLUDE_DIRS})

    # Embed OpenCL kernel source for runtime compilation
    set(OPENCL_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/OpenCLKernels.cl")
    target_compile_definitions(${SPRUX_LIBRARY} PRIVATE
        SPRUX_OPENCL_KERNEL_PATH="${OPENCL_KERNEL_SOURCE}")
endif()

target_compile_options(${SPRUX_LIBRARY} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${SPRUX_CXX_FLAGS}>)
                     
# SuiteSparse: AMD for fill-reducing ordering, CHOLMOD for fast symbolic analysis, BTF for max transversal
target_link_libraries(${SPRUX_LIBRARY}
                      ${SUITESPARSE_AMD_TARGET}
                      ${SUITESPARSE_CHOLMOD_TARGET}
                      ${SUITESPARSE_BTF_TARGET})

if(NOT BUILD_SHARED_LIBS)
    bundle_static_library(${SPRUX_LIBRARY} Sprux)
endif()

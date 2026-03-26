
add_library(BAL_utils BaAtLarge.cpp)

set(BAL_bench_sources BaAtLargeBench.cpp)
set(BAL_opt_sources BaAtLargeOptimizer.cpp)
set(bench_sources Bench.cpp)

if(HAVE_CHOLMOD)
    list(APPEND BAL_bench_sources
         BenchCholmod.cpp)
    list(APPEND bench_sources
         BenchCholmod.cpp)
endif()

if(HAVE_UMFPACK)
    list(APPEND bench_sources
         BenchUmfpack.cpp)
endif()

if(HAVE_CUDSS)
    list(APPEND bench_sources
         BenchCudss.cpp)
endif()

add_executable(BAL_bench ${BAL_bench_sources})
target_link_libraries (BAL_bench ${SPRUX_LIBRARY} testing BAL_utils)

add_executable(BAL_opt ${BAL_opt_sources})
target_link_libraries (BAL_opt ${SPRUX_LIBRARY} testing BAL_utils)

add_executable(bench ${bench_sources})
target_link_libraries (bench ${SPRUX_LIBRARY} testing)

add_executable(lu_bench LUBench.cpp)
target_link_libraries (lu_bench ${SPRUX_LIBRARY} testing)

add_executable(profile_chol_grid ProfileCholGrid.cpp)
target_link_libraries (profile_chol_grid ${SPRUX_LIBRARY} testing)

if(HAVE_CHOLMOD)
  target_link_libraries (BAL_bench ${SUITESPARSE_CHOLMOD_TARGET})
  target_link_libraries (bench ${SUITESPARSE_CHOLMOD_TARGET})
endif()

if(HAVE_UMFPACK)
  target_link_libraries (bench ${UMFPACK_LIBRARIES})
endif()

if(HAVE_CUDSS)
  target_link_libraries (bench cuDSS::cuDSS CUDA::cudart)
  target_link_libraries (lu_bench cuDSS::cuDSS CUDA::cudart)
endif()

add_subdirectory(sprux)

if(SPRUX_BUILD_TESTS OR SPRUX_BUILD_EXAMPLES)
  add_subdirectory(testing)
endif()

if(SPRUX_BUILD_EXAMPLES)
  add_subdirectory(benchmarking)
  add_subdirectory(examples)
endif()

if(SPRUX_BUILD_TESTS)
  add_subdirectory(tests)
endif()
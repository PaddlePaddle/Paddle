set(FLUID_CORE_MODULES proto_desc paddle_memory executor prune init io)

# Merge all modules into a single static library
cc_library(paddle_fluid DEPS ${FLUID_CORE_MODULES} ${GLOB_OP_LIB})

# install library & headers
if(NOT WITH_C_API AND WITH_FLUID)
  install(TARGETS paddle_fluid DESTINATION lib)
endif()

add_executable(example example.cc)
if(APPLE)
  set(OPTIONAL_LINK_FLAGS)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(OPTIONAL_LINK_FLAGS "-undefined dynamic_lookup")
  endif()
  target_link_libraries(example
      -Wl,-force_load paddle_fluid
      ${OPTIONAL_LINK_FLAGS}
      ${PTOOLS_LIB})
else()
  target_link_libraries(example
      -Wl,--start-group -Wl,--whole-archive paddle_fluid
      -Wl,--no-whole-archive -Wl,--end-group
      ${PTOOLS_LIB})
endif()

set(FLUID_CORE_MODULES
    backward proto_desc paddle_memory executor prune init ${GLOB_OP_LIB})

cc_library(paddle_fluid_api
    SRCS inference.cc
    DEPS ${FLUID_CORE_MODULES})

# Merge all modules into a simgle static library
cc_library(paddle_fluid DEPS paddle_fluid_api ${FLUID_CORE_MODULES})

# ptools
# just for testing, we may need to change the storing format for inference_model
# and move the dependent of pickle.
# download from http://www.picklingtools.com/
# build in the C++ sub-directory, using command
#     make -f Makefile.Linux libptools.so
set(PTOOLS_LIB)
set(PTOOLS_ROOT $ENV{PTOOLS_ROOT} CACHE PATH "Folder contains PicklingTools")
find_path(PTOOLS_INC_DIR chooseser.h PATHS ${PTOOLS_ROOT}/C++)
find_library(PTOOLS_SHARED_LIB NAMES ptools PATHS ${PTOOLS_ROOT}/C++)
if(PTOOLS_INC_DIR AND PTOOLS_SHARED_LIB)
  add_definitions(-DPADDLE_USE_PTOOLS)
  set(PTOOLS_LIB ptools)
  message(STATUS "Found PicklingTools: ${PTOOLS_SHARED_LIB}")
  add_library(${PTOOLS_LIB} SHARED IMPORTED GLOBAL)
  set_property(TARGET ${PTOOLS_LIB} PROPERTY IMPORTED_LOCATION ${PTOOLS_SHARED_LIB})
  include_directories(${PTOOLS_ROOT}/C++)
  include_directories(${PTOOLS_ROOT}/C++/opencontainers_1_8_5/include)
  add_definitions(-DOC_NEW_STYLE_INCLUDES) # used in ptools
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

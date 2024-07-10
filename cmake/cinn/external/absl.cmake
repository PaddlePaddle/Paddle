include(ExternalProject)

set(ABSL_SOURCES_DIR ${PADDLE_SOURCE_DIR}/third_party/absl)
set(ABSL_INSTALL_DIR ${THIRD_PARTY_PATH}/install/absl)
set(ABSL_PREFIX_DIR ${THIRD_PARTY_PATH}/absl)
set(ABSL_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

set(ABSL_REPOSITORY "${GIT_URL}/abseil/abseil-cpp.git")
set(ABSL_TAG "20210324.2")

set(OPTIONAL_ARGS
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
    "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
    "-DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}"
    "-DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}"
    "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
    "-DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}"
    "-DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}")

ExternalProject_Add(
  external_absl
  ${EXTERNAL_PROJECT_LOG_ARGS}
  DEPENDS gflags
  PREFIX ${ABSL_PREFIX_DIR}
  SOURCE_DIR ${ABSL_SOURCES_DIR}
  UPDATE_COMMAND ""
  CMAKE_ARGS ${OPTIONAL_ARGS}
             -DCMAKE_INSTALL_PREFIX=${ABSL_INSTALL_DIR}
             -DCMAKE_INSTALL_LIBDIR=${ABSL_INSTALL_DIR}/lib
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DWITH_GFLAGS=ON
             -Dgflags_DIR=${GFLAGS_INSTALL_DIR}/lib/cmake/gflags
             -DBUILD_TESTING=OFF
             -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
             ${EXTERNAL_OPTIONAL_ARGS}
  CMAKE_CACHE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=${ABSL_INSTALL_DIR}
    -DCMAKE_INSTALL_LIBDIR:PATH=${ABSL_INSTALL_DIR}/lib
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
  BUILD_BYPRODUCTS ${ABSL_INSTALL_DIR}/lib/libabsl_base.a
  BUILD_BYPRODUCTS ${ABSL_INSTALL_DIR}/lib/libabsl_hash.a
  BUILD_BYPRODUCTS ${ABSL_INSTALL_DIR}/lib/libabsl_wyhash.a
  BUILD_BYPRODUCTS ${ABSL_INSTALL_DIR}/lib/libabsl_city.a
  BUILD_BYPRODUCTS ${ABSL_INSTALL_DIR}/lib/libabsl_strings.a
  BUILD_BYPRODUCTS ${ABSL_INSTALL_DIR}/lib/libabsl_throw_delegate.a
  BUILD_BYPRODUCTS ${ABSL_INSTALL_DIR}/lib/libabsl_bad_any_cast_impl.a
  BUILD_BYPRODUCTS ${ABSL_INSTALL_DIR}/lib/libabsl_bad_optional_access.a
  BUILD_BYPRODUCTS ${ABSL_INSTALL_DIR}/lib/libabsl_bad_variant_access.a
  BUILD_BYPRODUCTS ${ABSL_INSTALL_DIR}/lib/libabsl_raw_hash_set.a)

# It may be more convenient if we just include all absl libs
set(ABSL_LIB_NAMES
    hash
    wyhash
    city
    strings
    throw_delegate
    bad_any_cast_impl
    bad_optional_access
    bad_variant_access
    raw_hash_set)
set(ABSL_LIBS "")

if(WITH_ROCM)
  list(APPEND ABSL_LIB_NAMES strings_internal raw_logging_internal)
endif()

add_library(absl STATIC IMPORTED GLOBAL)
set_property(TARGET absl PROPERTY IMPORTED_LOCATION
                                  ${ABSL_INSTALL_DIR}/lib/libabsl_base.a)

if(NOT USE_PREBUILD_EXTERNAL)
  add_dependencies(absl external_absl)
endif()
foreach(lib_name ${ABSL_LIB_NAMES})
  target_link_libraries(absl
                        INTERFACE ${ABSL_INSTALL_DIR}/lib/libabsl_${lib_name}.a)
endforeach()
include_directories(${ABSL_INSTALL_DIR}/include)

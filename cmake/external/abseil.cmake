INCLUDE(ExternalProject)


SET(ABSEIL_PREFIX_DIR   ${THIRD_PARTY_PATH}/absl)
SET(ABSEIL_INSTALL_DIR  ${THIRD_PARTY_PATH}/install/absl)
SET(ABSEIL_LIB_DIR  ${THIRD_PARTY_PATH}/install/absl/lib)
SET(ABSEIL_INCLUDE_DIR  "${ABSEIL_INSTALL_DIR}/include" CACHE PATH "abseil include directory." FORCE)
set(ABSEIL_REPOSITORY   https://github.com/abseil/abseil-cpp.git)
set(ABSEIL_TAG          ad904b6cd3906ddf79878003d92b7bc08d7786ae)

INCLUDE_DIRECTORIES(${ABSEIL_INCLUDE_DIR})


cache_third_party(extern_abseil
    REPOSITORY   ${ABSEIL_REPOSITORY}
    TAG          ${ABSEIL_TAG})


ExternalProject_Add(
    extern_abseil
    ${EXTERNAL_PROJECT_LOG_ARGS}
    ${SHALLOW_CLONE}
    "${ABSEIL_DOWNLOAD_CMD}"
    PREFIX          ${ABSEIL_PREFIX_DIR}
    SOURCE_DIR      ${ABSEIL_SOURCE_DIR}
    UPDATE_COMMAND  ""
    CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                    -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
                    -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
                    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                    -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
                    -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
                    -DCMAKE_INSTALL_PREFIX=${ABSEIL_INSTALL_DIR}
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                    ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${ABSEIL_INSTALL_DIR}
                     -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                     -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
)

add_library(absl_time INTERFACE)
target_link_libraries(absl_time INTERFACE ${ABSEIL_LIB_DIR}/libabsl_time.a
                                          ${ABSEIL_LIB_DIR}/libabsl_int128.a 
                                          ${ABSEIL_LIB_DIR}/libabsl_time_zone.a)
ADD_DEPENDENCIES(absl_time extern_abseil)

add_library(absl_hash INTERFACE)
target_link_libraries(absl_hash INTERFACE ${ABSEIL_LIB_DIR}/libabsl_hash.a
                                          ${ABSEIL_LIB_DIR}/libabsl_city.a)
ADD_DEPENDENCIES(absl_hash extern_abseil)

add_library(absl_synchronization INTERFACE)
target_link_libraries(absl_synchronization INTERFACE ${ABSEIL_LIB_DIR}/libabsl_synchronization.a
                                                     ${ABSEIL_LIB_DIR}/libabsl_symbolize.a 
                                                     ${ABSEIL_LIB_DIR}/libabsl_demangle_internal.a
                                                     ${ABSEIL_LIB_DIR}/libabsl_stacktrace.a 
                                                     ${ABSEIL_LIB_DIR}/libabsl_debugging_internal.a
                                                     ${ABSEIL_LIB_DIR}/libabsl_malloc_internal.a 
                                                     ${ABSEIL_LIB_DIR}/libabsl_spinlock_wait.a 
                                                     absl_time)
ADD_DEPENDENCIES(absl_synchronization extern_abseil)

add_library(absl_flat_hash_map INTERFACE)
target_link_libraries(absl_flat_hash_map INTERFACE  ${ABSEIL_LIB_DIR}/libabsl_hashtablez_sampler.a  
                                                    absl_hash
                                                    absl_synchronization
                                                    ${ABSEIL_LIB_DIR}/libabsl_throw_delegate.a
                                                    ${ABSEIL_LIB_DIR}/libabsl_raw_logging_internal.a
                                                    ${ABSEIL_LIB_DIR}/libabsl_base.a 
                                                    ${ABSEIL_LIB_DIR}/libabsl_exponential_biased.a 
                                                    )
ADD_DEPENDENCIES(absl_flat_hash_map extern_abseil)



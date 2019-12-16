
INCLUDE(ExternalProject)


# SET(ABSEIL_PREFIX_DIR   ${THIRD_PARTY_PATH}/absl)
# SET(ABSEIL_INSTALL_DIR  ${THIRD_PARTY_PATH}/install/absl)
# SET(ABSEIL_INCLUDE_DIR  "${ABSEIL_INSTALL_DIR}/include" CACHE PATH "abseil include directory." FORCE)
# set(ABSEIL_REPOSITORY   https://github.com/abseil/abseil-cpp.git)
# set(ABSEIL_TAG          39d68a422a2c08bc7bea56487aded3da8cbf0a19)
# 
# 
# cache_third_party(extern_abseil
#     REPOSITORY   ${ABSEIL_REPOSITORY}
#     TAG          ${ABSEIL_TAG})
# 
# 
# ExternalProject_Add(
#     extern_abseil
#     ${EXTERNAL_PROJECT_LOG_ARGS}
#     ${SHALLOW_CLONE}
#     "${ABSEIL_DOWNLOAD_CMD}"
#     PREFIX          ${ABSEIL_PREFIX_DIR}
#     SOURCE_DIR      ${ABSEIL_SOURCE_DIR}
#     UPDATE_COMMAND  ""
#     CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
#                     -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
#                     -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
#                     -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
#                     -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
#                     -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
#                     -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
#                     -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
#                     -DCMAKE_INSTALL_PREFIX=${ABSEIL_INSTALL_DIR}
#                     -DCMAKE_POSITION_INDEPENDENT_CODE=ON
#                     -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
#                     ${EXTERNAL_OPTIONAL_ARGS}
#     CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${ABSEIL_INSTALL_DIR}
#                      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
#                      -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
# )


find_package(absl)


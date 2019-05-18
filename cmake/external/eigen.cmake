INCLUDE(ExternalProject)

SET(EIGEN_SOURCE_DIR ${THIRD_PARTY_PATH}/eigen3)
SET(EIGEN_INCLUDE_DIR ${EIGEN_SOURCE_DIR}/src/extern_eigen3)
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})
if(NOT WITH_FAST_MATH)
  # EIGEN_FAST_MATH: https://eigen.tuxfamily.org/dox/TopicPreprocessorDirectives.html
  # enables some optimizations which might affect the accuracy of the result. 
  # This currently enables the SSE vectorization of sin() and cos(), 
  # and speedups sqrt() for single precision.
  # Defined to 1 by default. Define it to 0 to disable.
  add_definitions(-DEIGEN_FAST_MATH=0)
endif()

if(WITH_AMD_GPU)
    ExternalProject_Add(
        extern_eigen3
        ${EXTERNAL_PROJECT_LOG_ARGS}
        GIT_REPOSITORY  "https://github.com/sabreshao/hipeigen.git"
        GIT_TAG         7cb2b6e5a4b4a1efe658abb215cd866c6fb2275e
        PREFIX          ${EIGEN_SOURCE_DIR}
        UPDATE_COMMAND  ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
    )
else()
    ExternalProject_Add(
        extern_eigen3
        ${EXTERNAL_PROJECT_LOG_ARGS}
        GIT_REPOSITORY  "https://github.com/eigenteam/eigen-git-mirror"
        # eigen on cuda9.1 missing header of math_funtions.hpp
        # https://stackoverflow.com/questions/43113508/math-functions-hpp-not-found-when-using-cuda-with-eigen
        GIT_TAG         917060c364181f33a735dc023818d5a54f60e54c
        PREFIX          ${EIGEN_SOURCE_DIR}
        DOWNLOAD_NAME   "eigen"
        UPDATE_COMMAND  ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
    )
endif()

if (${CMAKE_VERSION} VERSION_LESS "3.3.0")
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/eigen3_dummy.c)
    file(WRITE ${dummyfile} "const char *dummy_eigen3 = \"${dummyfile}\";")
    add_library(eigen3 STATIC ${dummyfile})
else()
    add_library(eigen3 INTERFACE)
endif()

add_dependencies(eigen3 extern_eigen3)

LIST(APPEND external_project_dependencies eigen3)

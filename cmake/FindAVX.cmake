# This file is use to check all support level of AVX on your machine
# so that PaddlePaddle can unleash the vectorization power of muticore.

INCLUDE(CheckCXXSourceRuns)

SET(FIND_AVX_10)
SET(FIND_AVX_20)
SET(AVX_FLAGS)
SET(AVX_FOUND)

# Check AVX 2
SET(CMAKE_REQUIRED_FLAGS)
IF(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  SET(CMAKE_REQUIRED_FLAGS "-mavx2")
ELSEIF(MSVC AND NOT CMAKE_CL_64)  # reserve for WINDOWS
  SET(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
ENDIF()

CHECK_CXX_SOURCE_RUNS("
#include <immintrin.h>
int main()
{
    __m256i a = _mm256_set_epi32 (-1, 2, -3, 4, -1, 2, -3, 4);
    __m256i result = _mm256_abs_epi32 (a);
    return 0;
}" FIND_AVX_20)

# Check AVX
SET(CMAKE_REQUIRED_FLAGS)
IF(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    SET(CMAKE_REQUIRED_FLAGS "-mavx")
ELSEIF(MSVC AND NOT CMAKE_CL_64)
    SET(CMAKE_REQUIRED_FLAGS "/arch:AVX")
endif()

CHECK_CXX_SOURCE_RUNS("
#include <immintrin.h>
int main()
{
    __m256 a = _mm256_set_ps (-1.0f, 2.0f, -3.0f, 4.0f, -1.0f, 2.0f, -3.0f, 4.0f);
    __m256 b = _mm256_set_ps (1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f);
    __m256 result = _mm256_add_ps (a, b);
    return 0;
}" FIND_AVX_10)

IF(${FIND_AVX_20})
    IF(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        SET(AVX_FLAGS "${AVX_FLAGS} -mavx2")
    ELSEIF(MSVC)
        SET(AVX_FLAGS "${AVX_FLAGS} /arch:AVX2")
    ENDIF()
ENDIF()

IF(${FIND_AVX_10})
    IF(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        SET(AVX_FLAGS "${AVX_FLAGS} -mavx")
    ELSEIF(MSVC)
        SET(AVX_FLAGS "${AVX_FLAGS} /arch:AVX")
    ENDIF()
ENDIF()

IF(${FIND_AVX_10})
    SET(AVX_FOUND TRUE)
    MESSAGE(STATUS "Find CPU supports ${AVX_FLAGS}.")
ENDIF()

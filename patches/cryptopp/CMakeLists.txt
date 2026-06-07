# Please ensure your changes or patch meets minimum requirements.
#   The minimum requirements are 2.8.6. It roughly equates to
#   Ubuntu 14.05 LTS or Solaris 11.3. Please do not check in something
#   for 3.5.0 or higher because it will break LTS operating systems
#   and a number of developer boards used for testing. To test your
#   changes, please set up a Ubuntu 14.05 LTS system.

# Should we be setting things like this? We are not a C project
# so nothing should be done with the C compiler. But there is
# no reliable way to tell CMake we are C++.
# Cannot set this... Breaks Linux PowerPC with Clang:
# SET(CMAKE_C_COMPILER ${CMAKE_CXX_COMPILER})
# # error "The CMAKE_C_COMPILER is set to a C++ compiler"

if(NOT DEFINED cryptopp_DISPLAY_CMAKE_SUPPORT_WARNING)
  set(cryptopp_DISPLAY_CMAKE_SUPPORT_WARNING 1)
endif()
if(cryptopp_DISPLAY_CMAKE_SUPPORT_WARNING)
  message( STATUS
"*************************************************************************\n"
"The Crypto++ library does not officially support CMake. CMake support is a\n"
"community effort, and the library works with the folks using CMake to help\n"
"improve it. If you find an issue then please fix it or report it at\n"
"https://github.com/noloader/cryptopp-cmake.\n"
"-- *************************************************************************"
)
endif()

# Print useful information
message( STATUS "CMake version ${CMAKE_VERSION}" )
message( STATUS "System ${CMAKE_SYSTEM_NAME}" )
message( STATUS "Processor ${CMAKE_SYSTEM_PROCESSOR}" )

cmake_minimum_required(VERSION 2.8.6)
if (${CMAKE_VERSION} VERSION_LESS "3.0.0")
  project(cryptopp)
  set(cryptopp_VERSION_MAJOR 8)
  set(cryptopp_VERSION_MINOR 6)
  set(cryptopp_VERSION_PATCH 0)
else ()
  cmake_policy(SET CMP0048 NEW)
  project(cryptopp VERSION 8.6.0)
  if (NOT ${CMAKE_VERSION} VERSION_LESS "3.1.0")
    cmake_policy(SET CMP0054 NEW)
  endif ()
endif ()

# Need to set SRC_DIR manually after removing the Python library code.
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Make RelWithDebInfo the default (it does e.g. add '-O2 -g -DNDEBUG' for GNU)
#   If not in multi-configuration environments, no explicit build type or CXX
#   flags are set by the user and if we are the root CMakeLists.txt file.
if (NOT CMAKE_CONFIGURATION_TYPES AND
    NOT CMAKE_NO_BUILD_TYPE AND
    NOT CMAKE_BUILD_TYPE AND
    NOT CMAKE_CXX_FLAGS AND
    CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif ()

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

# We now carry around test programs. test_cxx.cpp is the default C++ one.
# Also see https://github.com/weidai11/cryptopp/issues/741.
set(TEST_PROG_DIR ${SRC_DIR}/TestPrograms)
set(TEST_CXX_FILE ${TEST_PROG_DIR}/test_cxx.cpp)

# https://github.com/noloader/cryptopp-cmake/issues/56
# https://stackoverflow.com/a/40152725
if (APPLE)
  set(USE_INTERMEDIATE_OBJECTS_TARGET OFF)
endif()

#============================================================================
# Settable options
#============================================================================

option(BUILD_STATIC "Build static library" ON)
option(BUILD_SHARED "Build shared library" ON)
option(BUILD_TESTING "Build library tests" ON)
option(BUILD_DOCUMENTATION "Use Doxygen to create the HTML based API documentation" OFF)
option(USE_INTERMEDIATE_OBJECTS_TARGET "Use a common intermediate objects target for the static and shared library targets" ON)

if (${CMAKE_VERSION} VERSION_GREATER "3.1")
  option(USE_OPENMP "Enable OpenMP to parallelize some of the algorithms. Note that this isn't always faster, see https://www.cryptopp.com/wiki/OpenMP" OFF)
endif()


# These are IA-32 options.
option(DISABLE_ASM "Disable ASM" OFF)
option(DISABLE_SSSE3 "Disable SSSE3" OFF)
option(DISABLE_SSE4 "Disable SSE4" OFF)
option(DISABLE_AESNI "Disable AES-NI" OFF)
option(DISABLE_CLMUL "Disable CLMUL" OFF)
option(DISABLE_SHA "Disable SHA" OFF)
option(DISABLE_AVX "Disable AVX" OFF)
option(DISABLE_AVX2 "Disable AVX2" OFF)

# These are ARM A-32 options
option(DISABLE_ARM_NEON "Disable NEON" OFF)

# These are Aarch64 options
option(DISABLE_ARM_AES "Disable ASIMD" OFF)
option(DISABLE_ARM_AES "Disable AES" OFF)
option(DISABLE_ARM_PMULL "Disable PMULL" OFF)
option(DISABLE_ARM_SHA "Disable SHA" OFF)

# These are PowerPC options
option(DISABLE_ALTIVEC "Disable Altivec" OFF)
option(DISABLE_POWER7 "Disable POWER7" OFF)
option(DISABLE_POWER8 "Disable POWER8" OFF)
option(DISABLE_POWER9 "Disable POWER9" OFF)

set(CRYPTOPP_DATA_DIR "" CACHE PATH "Crypto++ test data directory")

#============================================================================
# Compiler options
#============================================================================

set(CRYPTOPP_COMPILE_DEFINITIONS)
set(CRYPTOPP_COMPILE_OPTIONS)

# Stop hiding the damn output...
# set(CMAKE_VERBOSE_MAKEFILE ON)

# Stop CMake complaining...
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(MACOSX_RPATH FALSE)
endif()

# Always 1 ahead in Master. Also see http://groups.google.com/forum/#!topic/cryptopp-users/SFhqLDTQPG4
set(LIB_VER ${cryptopp_VERSION_MAJOR}${cryptopp_VERSION_MINOR}${cryptopp_VERSION_PATCH})

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  list(APPEND CRYPTOPP_COMPILE_OPTIONS -wd68 -wd186 -wd279 -wd327 -wd161 -wd3180)
endif ()

# Also see http://github.com/weidai11/cryptopp/issues/395
if (DISABLE_ASM)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_ASM)
endif ()
if (DISABLE_SSSE3)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_SSSE3)
endif ()
if (DISABLE_SSE4)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_SSSE4)
endif ()
if (DISABLE_CLMUL)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_CLMUL)
endif ()
if (DISABLE_AESNI)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_AESNI)
endif ()
if (DISABLE_RDRAND)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_RDRAND)
endif ()
if (DISABLE_RDSEED)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_RDSEED)
endif ()
if (DISABLE_AVX)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_AVX)
endif ()
if (DISABLE_AVX2)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_AVX2)
endif ()
if (DISABLE_SHA)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_SHA)
endif ()
if (DISABLE_ARM_NEON)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_ARM_NEON)
endif ()
if (DISABLE_ARM_ASIMD)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_ARM_ASIMD)
endif ()
if (DISABLE_ARM_AES)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_ARM_AES)
endif ()
if (DISABLE_ARM_PMULL)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_ARM_PMULL)
endif ()
if (DISABLE_ARM_SHA)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_ARM_SHA)
endif ()
if (DISABLE_ALTIVEC)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_ALTIVEC)
endif ()
if (DISABLE_POWER7)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_POWER7)
endif ()
if (DISABLE_POWER8)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_POWER8)
endif ()
if (DISABLE_POWER9)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_POWER9)
endif ()
if (NOT CRYPTOPP_DATA_DIR STREQUAL "")
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS "CRYPTOPP_DATA_DIR=${CRYPTOPP_DATA_DIR}")
endif ()

###############################################################################

# Try to find a Posix compatible grep and sed. Solaris, Digital Unix,
#   Tru64, HP-UX and a few others need tweaking

if (EXISTS /usr/xpg4/bin/grep)
  set(GREP_CMD /usr/xpg4/bin/grep)
elseif (EXISTS /usr/gnu/bin/grep)
  set(GREP_CMD /usr/gnu/bin/grep)
elseif (EXISTS /usr/linux/bin/grep)
  set(GREP_CMD /usr/linux/bin/grep)
else ()
  set(GREP_CMD grep)
endif ()

if (EXISTS /usr/xpg4/bin/sed)
  set(SED_CMD /usr/xpg4/bin/sed)
elseif (EXISTS /usr/gnu/bin/sed)
  set(SED_CMD /usr/gnu/bin/sed)
elseif (EXISTS /usr/linux/bin/sed)
  set(SED_CMD /usr/linux/bin/sed)
else ()
  set(SED_CMD sed)
endif ()

###############################################################################

function(CheckCompileOption opt var)

  if (MSVC)

    # TODO: improve this...
    CHECK_CXX_COMPILER_FLAG(${opt} ${var})

  elseif (CMAKE_CXX_COMPILER_ID MATCHES "SunPro")

    message(STATUS "Performing Test ${var}")
    execute_process(
      COMMAND sh -c "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${opt} -c ${TEST_CXX_FILE} 2>&1"
      COMMAND ${GREP_CMD} -i -c -E "illegal value ignored"
      RESULT_VARIABLE COMMAND_RESULT
      OUTPUT_VARIABLE COMMAND_OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    # No dereference below. Thanks for the warning, CMake (not!).
    if (COMMAND_RESULT AND NOT COMMAND_OUTPUT)
      set(${var} 1 PARENT_SCOPE)
      message(STATUS "Performing Test ${var} - Success")
    else ()
      set(${var} 0 PARENT_SCOPE)
      message(STATUS "Performing Test ${var} - Failed")
    endif ()

  # Must use CMAKE_CXX_COMPILER here due to XLC 13.1 and LLVM front-end.
  elseif (CMAKE_CXX_COMPILER MATCHES "xlC")

    message(STATUS "Performing Test ${var}")
    execute_process(
      COMMAND sh -c "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${opt} -c ${TEST_CXX_FILE} 2>&1"
      COMMAND ${GREP_CMD} -i -c -E "Unrecognized value"
      RESULT_VARIABLE COMMAND_RESULT
      OUTPUT_VARIABLE COMMAND_OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    # No dereference below. Thanks for the warning, CMake (not!).
    if (COMMAND_RESULT AND NOT COMMAND_OUTPUT)
      set(${var} 1 PARENT_SCOPE)
      message(STATUS "Performing Test ${var} - Success")
    else ()
      set(${var} 0 PARENT_SCOPE)
      message(STATUS "Performing Test ${var} - Failed")
    endif ()

  else ()

    CHECK_CXX_COMPILER_FLAG(${opt} ${var})

  endif ()

endfunction(CheckCompileOption)

function(CheckCompileLinkOption opt var prog)

  if (MSVC)

    # TODO: improve this...
    CHECK_CXX_COMPILER_FLAG(${opt} ${var})

  elseif (APPLE)

    message(STATUS "Performing Test ${var}")
    try_compile(COMMAND_SUCCESS ${CMAKE_BINARY_DIR} ${prog} COMPILE_DEFINITIONS ${opt})
    if (COMMAND_SUCCESS)
      set(${var} 1 PARENT_SCOPE)
      message(STATUS "Performing Test ${var} - Success")
    else ()
      set(${var} 0 PARENT_SCOPE)
      message(STATUS "Performing Test ${var} - Failed")
    endif ()

  else ()

    message(STATUS "Performing Test ${var}")
    try_compile(COMMAND_SUCCESS ${CMAKE_BINARY_DIR} ${prog} COMPILE_DEFINITIONS ${opt})
    if (COMMAND_SUCCESS)
        set(${var} 1 PARENT_SCOPE)
        message(STATUS "Performing Test ${var} - Success")
    else ()
      set(${var} 0 PARENT_SCOPE)
      message(STATUS "Performing Test ${var} - Failed")
    endif ()

  endif ()

endfunction(CheckCompileLinkOption)

function(AddCompileOption opt)

    if ("${COMMAND_OUTPUT}" NOT STREQUAL "")
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "${opt}")
    endif ()

endfunction(AddCompileOption)

###############################################################################

function(DumpMachine output pattern)

  if (MSVC)

    # CMake does not provide a generic shell/terminal mechanism
    #  and Microsoft environments don't know what 'sh' is.
    set(${output} 0 PARENT_SCOPE)

  else ()
    execute_process(
      COMMAND sh -c "${CMAKE_CXX_COMPILER} -dumpmachine 2>&1"
      COMMAND ${GREP_CMD} -i -c -E "${pattern}"
      OUTPUT_VARIABLE ${output}
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(${output} "${${output}}" PARENT_SCOPE)
  endif()

endfunction(DumpMachine)

# Thanks to Anonimal for MinGW; see http://github.com/weidai11/cryptopp/issues/466
DumpMachine(CRYPTOPP_AMD64 "(x86_64|AMD64|amd64)")
DumpMachine(CRYPTOPP_I386 "^i.86$")
DumpMachine(CRYPTOPP_MINGW32 "^mingw32")
DumpMachine(CRYPTOPP_MINGW64 "(w64-mingw32|mingw64)")
DumpMachine(CRYPTOPP_ARMV8 "(armv8|arm64|aarch32|aarch64)")
DumpMachine(CRYPTOPP_ARM32 "(arm|armhf|arm7l|eabihf)")
DumpMachine(CRYPTOPP_PPC32 "^(powerpc|ppc)")
DumpMachine(CRYPTOPP_PPC64 "^ppc64")

# Cleanup 32/64 bit
if (CRYPTOPP_AMD64)
  set (CRYPTOPP_I386 0)
endif ()

if (CRYPTOPP_ARMV8)
  set (CRYPTOPP_ARM32 0)
endif ()

if (CRYPTOPP_PPC64)
  set (CRYPTOPP_PPC32 0)
endif ()


###############################################################################

# Test SunCC for a string like 'CC: Sun C++ 5.13 SunOS_i386'
if (NOT CRYPTOPP_SOLARIS AND NOT MSVC)
  execute_process(COMMAND sh -c "${CMAKE_CXX_COMPILER} -V 2>&1"
    COMMAND ${GREP_CMD} -i -c "SunOS"
    OUTPUT_VARIABLE CRYPTOPP_SOLARIS
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif ()

# Test GCC for a string like 'i386-pc-solaris2.11'
if (NOT CRYPTOPP_SOLARIS AND NOT MSVC)
  execute_process(COMMAND sh -c "${CMAKE_CXX_COMPILER} -dumpmachine 2>&1"
    COMMAND ${GREP_CMD} -i -c "Solaris"
    OUTPUT_VARIABLE CRYPTOPP_SOLARIS
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif ()

# Fixup PowerPC. If both 32-bit and 64-bit use 64-bit.
if (CRYPTOPP_PPC32 AND CRYPTOPP_PPC64)
  unset(CRYPTOPP_PPC32)
endif ()

# Fixup for xlC compiler. -dumpmachine fails so we miss PowerPC
# TODO: something better than proxying the platform via compiler
# Must use CMAKE_CXX_COMPILER here due to XLC 13.1 and LLVM front-end.
if (CMAKE_CXX_COMPILER MATCHES "xlC")
  message ("-- Fixing platform due to IBM xlC")
  set(CRYPTOPP_PPC64 1)
endif ()

# DumpMachine SunCC style
if (CMAKE_CXX_COMPILER_ID STREQUAL "SunPro")

  # SunCC is 32-bit, but it builds both 32 and 64 bit. Use
  execute_process(COMMAND sh -c "${CMAKE_CXX_COMPILER} -V 2>&1"
    COMMAND ${GREP_CMD} -i -c "Sparc"
    OUTPUT_VARIABLE CRYPTOPP_SPARC
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(COMMAND sh -c "${CMAKE_CXX_COMPILER} -V 2>&1"
    COMMAND ${GREP_CMD} -i -c -E "i386|i86"
    OUTPUT_VARIABLE CRYPTOPP_I386
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(COMMAND isainfo -k
    COMMAND ${GREP_CMD} -i -c "i386"
    OUTPUT_VARIABLE KERNEL_I386
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(COMMAND isainfo -k
    COMMAND ${GREP_CMD} -i -c "amd64"
    OUTPUT_VARIABLE KERNEL_AMD64
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(COMMAND isainfo -k
    COMMAND ${GREP_CMD} -i -c "Sparc"
    OUTPUT_VARIABLE KERNEL_SPARC
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(COMMAND isainfo -k
    COMMAND ${GREP_CMD} -i -c -E "UltraSarc|Sparc64|SparcV9"
    OUTPUT_VARIABLE KERNEL_SPARC64
    OUTPUT_STRIP_TRAILING_WHITESPACE)

endif ()

###############################################################################

# TODO: what about ICC and LLVM on Windows?
if (MSVC)
  if (CMAKE_SYSTEM_VERSION MATCHES "10\\.0.*")
    # https://docs.microsoft.com/en-us/cpp/porting/modifying-winver-and-win32-winnt
    list(APPEND CRYPTOPP_COMPILE_DEFINITIONS "_WIN32_WINNT=0x0A00")
  endif ()
  # winapifamily.h is missing on AppVeyor machines
  INCLUDE (CheckIncludeFiles)
  CHECK_INCLUDE_FILES (winapifamily.h HAVE_WINAPIFAMILY_H)
  if (HAVE_WINAPIFAMILY_H)
    if ("${CMAKE_GENERATOR}" STREQUAL "Ninja")
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "/nologo")
    else()
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "/FIwinapifamily.h")
    endif()
  endif()
endif ()

# Enable PIC for all target machines except 32-bit i386 due to register pressures.
if (NOT CRYPTOPP_I386)
  SET(CMAKE_POSITION_INDEPENDENT_CODE 1)
endif ()

# IBM XLC compiler options for AIX and Linux.
# Must use CMAKE_CXX_COMPILER here due to XLC 13.1 and LLVM front-end.
if (CMAKE_CXX_COMPILER MATCHES "xlC")

  #CheckCompileLinkOption("-qxlcompatmacros" CRYPTOPP_XLC_COMPAT "${TEST_CXX_FILE}")
  #if (CRYPTOPP_XLC_COMPAT)
  #  list(APPEND CRYPTOPP_COMPILE_OPTIONS "-qxlcompatmacros")
  #endif ()

  CheckCompileLinkOption("-qrtti" CRYPTOPP_PPC_RTTI "${TEST_CXX_FILE}")
  if (CRYPTOPP_PPC_RTTI)
    list(APPEND CRYPTOPP_COMPILE_OPTIONS "-qrtti")
  endif ()

  CheckCompileLinkOption("-qmaxmem=-1" CRYPTOPP_PPC_MAXMEM "${TEST_CXX_FILE}")
  if (CRYPTOPP_PPC_MAXMEM)
    list(APPEND CRYPTOPP_COMPILE_OPTIONS "-qmaxmem=-1")
  endif ()

  CheckCompileLinkOption("-qthreaded" CRYPTOPP_PPC_THREADED "${TEST_CXX_FILE}")
  if (CRYPTOPP_PPC_THREADED)
    list(APPEND CRYPTOPP_COMPILE_OPTIONS "-qthreaded")
  endif ()
endif ()

# Solaris specific
if (CRYPTOPP_SOLARIS)

  # SunCC needs -template=no%extdef
  if (CMAKE_CXX_COMPILER_ID STREQUAL "SunPro")
    list(APPEND CRYPTOPP_COMPILE_OPTIONS "-template=no%extdef")
  endif ()

  # SunCC needs -xregs=no%appl on Sparc (not x86) for libraries (not test program)
  # TODO: wire this up properly
  if (CMAKE_CXX_COMPILER_ID STREQUAL "SunPro" AND (CRYPTOPP_SPARC OR CRYPTOPP_SPARC64))
    list(APPEND CRYPTOPP_COMPILE_OPTIONS "-xregs=no%appl")
  endif ()

  # GCC needs to enable use of '/' for division in the assembler
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND CRYPTOPP_COMPILE_OPTIONS "-Wa,--divide")
  endif ()

endif ()

#============================================================================
# Sources & headers
#============================================================================

# Library headers
file(GLOB cryptopp_HEADERS ${SRC_DIR}/*.h)

# Remove headers used to build test suite
list(REMOVE_ITEM cryptopp_HEADERS
    ${SRC_DIR}/bench.h
    ${SRC_DIR}/validate.h
    )

# Test sources. You can use the GNUmakefile to generate the list: `make sources`.
set(cryptopp_SOURCES_TEST
    ${SRC_DIR}/test.cpp
    ${SRC_DIR}/bench1.cpp
    ${SRC_DIR}/bench2.cpp
    ${SRC_DIR}/bench3.cpp
    ${SRC_DIR}/validat0.cpp
    ${SRC_DIR}/validat1.cpp
    ${SRC_DIR}/validat2.cpp
    ${SRC_DIR}/validat3.cpp
    ${SRC_DIR}/validat4.cpp
    ${SRC_DIR}/validat5.cpp
    ${SRC_DIR}/validat6.cpp
    ${SRC_DIR}/validat7.cpp
    ${SRC_DIR}/validat8.cpp
    ${SRC_DIR}/validat9.cpp
    ${SRC_DIR}/validat10.cpp
    ${SRC_DIR}/regtest1.cpp
    ${SRC_DIR}/regtest2.cpp
    ${SRC_DIR}/regtest3.cpp
    ${SRC_DIR}/regtest4.cpp
    ${SRC_DIR}/datatest.cpp
    ${SRC_DIR}/fipsalgt.cpp
    ${SRC_DIR}/fipstest.cpp
    ${SRC_DIR}/dlltest.cpp
    #${SRC_DIR}/adhoc.cpp
    )

# Library sources. You can use the GNUmakefile to generate the list: `make sources`.
# Makefile sorted them at http://github.com/weidai11/cryptopp/pull/426.
file(GLOB cryptopp_SOURCES ${SRC_DIR}/*.cpp)
list(SORT cryptopp_SOURCES)
list(REMOVE_ITEM cryptopp_SOURCES
    ${SRC_DIR}/cryptlib.cpp
    ${SRC_DIR}/cpu.cpp
    ${SRC_DIR}/integer.cpp
    ${SRC_DIR}/pch.cpp
    ${SRC_DIR}/simple.cpp
    ${SRC_DIR}/adhoc.cpp
    ${cryptopp_SOURCES_TEST}
    )
set(cryptopp_SOURCES
    ${SRC_DIR}/cryptlib.cpp
    ${SRC_DIR}/cpu.cpp
    ${SRC_DIR}/integer.cpp
    ${cryptopp_SOURCES}
    )

if(ANDROID)
    include_directories(${ANDROID_NDK}/sources/android/cpufeatures)
    list(APPEND cryptopp_SOURCES ${ANDROID_NDK}/sources/android/cpufeatures/cpu-features.c)
endif()

set(cryptopp_SOURCES_ASM)

if (MSVC AND NOT DISABLE_ASM)
  if (${CMAKE_GENERATOR} MATCHES ".*ARM")
    message(STATUS "Disabling ASM because ARM is specified as target platform.")
  else ()
    enable_language(ASM_MASM)
    list(APPEND cryptopp_SOURCES_ASM
      ${SRC_DIR}/rdrand.asm
      ${SRC_DIR}/rdseed.asm
      )
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
      list(APPEND cryptopp_SOURCES_ASM
        ${SRC_DIR}/x64dll.asm
        ${SRC_DIR}/x64masm.asm
        )
      set_source_files_properties(${cryptopp_SOURCES_ASM} PROPERTIES COMPILE_DEFINITIONS "_M_X64")
    else ()
      set_source_files_properties(${cryptopp_SOURCES_ASM} PROPERTIES COMPILE_DEFINITIONS "_M_X86" COMPILE_FLAGS "/safeseh")
    endif ()
    set_source_files_properties(${cryptopp_SOURCES_ASM} PROPERTIES LANGUAGE ASM_MASM)
  endif ()
endif ()

#============================================================================
# Architecture flags
#============================================================================

# TODO: Android, AIX, IBM xlC, iOS and a few other profiles are missing.

# New as of Pull Request 461, http://github.com/weidai11/cryptopp/pull/461.
# Must use CMAKE_CXX_COMPILER here due to XLC 13.1 and LLVM front-end.
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR CMAKE_CXX_COMPILER MATCHES "xlC")

  if (CRYPTOPP_AMD64 OR CRYPTOPP_I386)

    # For Darwin and a GCC port compiler, we need to check for -Wa,-q first. -Wa,-q
    # is a GCC option, and it tells GCC to use the Clang Integrated Assembler. We
    # need LLVM's assembler because GAS is too old on Apple platforms. GAS will
    # not assemble modern ISA, like AVX or AVX2.
	if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
      CheckCompileLinkOption("-Wa,-q" CRYPTOPP_X86_WAQ
                             "${TEST_PROG_DIR}/test_x86_sse2.cpp")

      if (CRYPTOPP_X86_WAQ)
        list(APPEND CRYPTOPP_COMPILE_OPTIONS "-Wa,-q")
      endif ()
	endif ()

    # Now we can move on to normal feature testing.
    CheckCompileLinkOption("-msse2" CRYPTOPP_X86_SSE2
                           "${TEST_PROG_DIR}/test_x86_sse2.cpp")
    CheckCompileLinkOption("-mssse3" CRYPTOPP_X86_SSSE3
                           "${TEST_PROG_DIR}/test_x86_ssse3.cpp")
    CheckCompileLinkOption("-msse4.1" CRYPTOPP_X86_SSE41
                           "${TEST_PROG_DIR}/test_x86_sse41.cpp")
    CheckCompileLinkOption("-msse4.2" CRYPTOPP_X86_SSE42
                           "${TEST_PROG_DIR}/test_x86_sse42.cpp")
    CheckCompileLinkOption("-mssse3 -mpclmul" CRYPTOPP_X86_CLMUL
                           "${TEST_PROG_DIR}/test_x86_clmul.cpp")
    CheckCompileLinkOption("-msse4.1 -maes" CRYPTOPP_X86_AES
                           "${TEST_PROG_DIR}/test_x86_aes.cpp")
    CheckCompileLinkOption("-mavx" CRYPTOPP_X86_AVX
                           "${TEST_PROG_DIR}/test_x86_avx.cpp")
    CheckCompileLinkOption("-mavx2" CRYPTOPP_X86_AVX2
                           "${TEST_PROG_DIR}/test_x86_avx2.cpp")
    CheckCompileLinkOption("-msse4.2 -msha" CRYPTOPP_X86_SHA
                           "${TEST_PROG_DIR}/test_x86_sha.cpp")
    if (EXISTS "${TEST_PROG_DIR}/test_asm_mixed.cpp")
      CheckCompileLinkOption("" CRYPTOPP_MIXED_ASM
                             "${TEST_PROG_DIR}/test_asm_mixed.cpp")
    else ()
      CheckCompileLinkOption("" CRYPTOPP_MIXED_ASM
                             "${TEST_PROG_DIR}/test_mixed_asm.cpp")
    endif ()

    # https://github.com/weidai11/cryptopp/issues/756
    if (NOT CRYPTOPP_MIXED_ASM)
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_MIXED_ASM")
    endif ()

    if (NOT CRYPTOPP_X86_SSE2 AND NOT DISABLE_ASM)
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_ASM")
    elseif (CRYPTOPP_X86_SSE2 AND NOT DISABLE_ASM)
      set_source_files_properties(${SRC_DIR}/sse_simd.cpp PROPERTIES COMPILE_FLAGS "-msse2")
      set_source_files_properties(${SRC_DIR}/chacha_simd.cpp PROPERTIES COMPILE_FLAGS "-msse2")
      set_source_files_properties(${SRC_DIR}/donna_sse.cpp PROPERTIES COMPILE_FLAGS "-msse2")
    endif ()
    if (NOT CRYPTOPP_X86_SSSE3 AND NOT DISABLE_SSSE3)
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_SSSE3")
    elseif (CRYPTOPP_X86_SSSE3 AND NOT DISABLE_SSSE3)
      set_source_files_properties(${SRC_DIR}/aria_simd.cpp PROPERTIES COMPILE_FLAGS "-mssse3")
      set_source_files_properties(${SRC_DIR}/cham_simd.cpp PROPERTIES COMPILE_FLAGS "-mssse3")
      set_source_files_properties(${SRC_DIR}/keccak_simd.cpp PROPERTIES COMPILE_FLAGS "-mssse3")
      set_source_files_properties(${SRC_DIR}/lea_simd.cpp PROPERTIES COMPILE_FLAGS "-mssse3")
      set_source_files_properties(${SRC_DIR}/lsh256_sse.cpp PROPERTIES COMPILE_FLAGS "-mssse3")
      set_source_files_properties(${SRC_DIR}/lsh512_sse.cpp PROPERTIES COMPILE_FLAGS "-mssse3")
      set_source_files_properties(${SRC_DIR}/simon128_simd.cpp PROPERTIES COMPILE_FLAGS "-mssse3")
      set_source_files_properties(${SRC_DIR}/speck128_simd.cpp PROPERTIES COMPILE_FLAGS "-mssse3")
      if (NOT CRYPTOPP_X86_SSE41 AND NOT DISABLE_SSE4)
        list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_SSE4")
      elseif (CRYPTOPP_X86_SSE41 AND NOT DISABLE_SSE4)
        set_source_files_properties(${SRC_DIR}/blake2s_simd.cpp PROPERTIES COMPILE_FLAGS "-msse4.1")
        set_source_files_properties(${SRC_DIR}/blake2b_simd.cpp PROPERTIES COMPILE_FLAGS "-msse4.1")
      endif ()
      if (NOT CRYPTOPP_X86_SSE42 AND NOT DISABLE_SSE4)
        list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_SSE4")
      elseif (CRYPTOPP_X86_SSE42 AND NOT DISABLE_SSE4)
        set_source_files_properties(${SRC_DIR}/crc_simd.cpp PROPERTIES COMPILE_FLAGS "-msse4.2")
        if (NOT CRYPTOPP_X86_CLMUL AND NOT DISABLE_CLMUL)
          list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_CLMUL")
        elseif (CRYPTOPP_X86_CLMUL AND NOT DISABLE_CLMUL)
          set_source_files_properties(${SRC_DIR}/gcm_simd.cpp PROPERTIES COMPILE_FLAGS "-mssse3 -mpclmul")
          set_source_files_properties(${SRC_DIR}/gf2n_simd.cpp PROPERTIES COMPILE_FLAGS "-mpclmul")
        endif ()
        if (NOT CRYPTOPP_X86_AES AND NOT DISABLE_AES)
          list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_AESNI")
        elseif (CRYPTOPP_X86_AES AND NOT DISABLE_AES)
          set_source_files_properties(${SRC_DIR}/rijndael_simd.cpp PROPERTIES COMPILE_FLAGS "-msse4.1 -maes")
          set_source_files_properties(${SRC_DIR}/sm4_simd.cpp PROPERTIES COMPILE_FLAGS "-mssse3 -maes")
        endif ()
        if (NOT CRYPTOPP_X86_AVX2 AND NOT DISABLE_AVX2)
          list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_AVX2")
        elseif (CRYPTOPP_X86_AVX2 AND NOT DISABLE_AVX2)
          set_source_files_properties(${SRC_DIR}/chacha_avx.cpp PROPERTIES COMPILE_FLAGS "-mavx2")
          set_source_files_properties(${SRC_DIR}/lsh256_avx.cpp PROPERTIES COMPILE_FLAGS "-mavx2")
          set_source_files_properties(${SRC_DIR}/lsh512_avx.cpp PROPERTIES COMPILE_FLAGS "-mavx2")
        endif ()
        if (NOT CRYPTOPP_X86_SHA AND NOT DISABLE_SHA)
          list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_SHANI")
        elseif (CRYPTOPP_X86_SHA AND NOT DISABLE_SHA)
          set_source_files_properties(${SRC_DIR}/sha_simd.cpp PROPERTIES COMPILE_FLAGS "-msse4.2 -msha")
          set_source_files_properties(${SRC_DIR}/shacal2_simd.cpp PROPERTIES COMPILE_FLAGS "-msse4.2 -msha")
        endif ()
      endif ()
    endif ()

  elseif (CRYPTOPP_ARMV8)

    # This checks for <arm_acle.h>
    CheckCompileLinkOption("-march=armv8-a" CRYPTOPP_ARM_ACLE_HEADER
                           "${TEST_PROG_DIR}/test_arm_acle_header.cpp")

    # Use <arm_acle.h> if available
    if (CRYPTOPP_ARM_NEON_HEADER)
      CheckCompileOption("-march=armv8-a -DCRYPTOPP_ARM_ACLE_HEADER=1" CRYPTOPP_ARMV8A_ASIMD)
      CheckCompileOption("-march=armv8-a+crc -DCRYPTOPP_ARM_ACLE_HEADER=1" CRYPTOPP_ARMV8A_CRC)
      CheckCompileOption("-march=armv8-a+crypto -DCRYPTOPP_ARM_ACLE_HEADER=1" CRYPTOPP_ARMV8A_CRYPTO)
    else ()
      CheckCompileOption("-march=armv8-a" CRYPTOPP_ARMV8A_ASIMD)
      CheckCompileOption("-march=armv8-a+crc" CRYPTOPP_ARMV8A_CRC)
      CheckCompileOption("-march=armv8-a+crypto" CRYPTOPP_ARMV8A_CRYPTO)
    endif ()

    if (CRYPTOPP_ARMV8A_ASIMD)
      set_source_files_properties(${SRC_DIR}/aria_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
      set_source_files_properties(${SRC_DIR}/blake2s_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
      set_source_files_properties(${SRC_DIR}/blake2b_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
      set_source_files_properties(${SRC_DIR}/chacha_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
      set_source_files_properties(${SRC_DIR}/cham_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
      set_source_files_properties(${SRC_DIR}/lea_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
      set_source_files_properties(${SRC_DIR}/neon_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
      set_source_files_properties(${SRC_DIR}/simon128_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
      set_source_files_properties(${SRC_DIR}/speck128_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
    else ()
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_ARM_ASIMD")
    endif ()
    if (CRYPTOPP_ARMV8A_CRC)
      set_source_files_properties(${SRC_DIR}/crc_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a+crc")
    else ()
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_ARM_CRC32")
    endif ()
    if (CRYPTOPP_ARMV8A_CRYPTO)
      set_source_files_properties(${SRC_DIR}/gcm_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a+crypto")
      set_source_files_properties(${SRC_DIR}/gf2n_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a+crypto")
      set_source_files_properties(${SRC_DIR}/rijndael_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a+crypto")
      set_source_files_properties(${SRC_DIR}/sha_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a+crypto")
      set_source_files_properties(${SRC_DIR}/shacal2_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a+crypto")
    else ()
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_ARM_AES")
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_ARM_PMULL")
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_ARM_SHA")
    endif ()

  elseif (CRYPTOPP_ARM32)

    # This checks for <arm_neon.h>
    CheckCompileLinkOption("-march=armv7-a -mfpu=neon" CRYPTOPP_ARM_NEON_HEADER
                           "${TEST_PROG_DIR}/test_arm_neon_header.cpp")

    # Use <arm_neon.h> if available
    if (CRYPTOPP_ARM_NEON_HEADER)
      CheckCompileLinkOption("-march=armv7-a -mfpu=neon -DCRYPTOPP_ARM_NEON_HEADER=1" CRYPTOPP_ARMV7A_NEON
                             "${TEST_PROG_DIR}/test_arm_neon.cpp")
    else ()
      CheckCompileLinkOption("-march=armv7-a -mfpu=neon" CRYPTOPP_ARMV7A_NEON
                             "${TEST_PROG_DIR}/test_arm_neon.cpp")
    endif ()

    if (CRYPTOPP_ARMV7A_NEON)

      # Add Cryptogams ASM files for ARM on Linux. Linux is required due to GNU Assembler.
      # AES requires -mthumb under Clang. Do not add -mthumb for SHA for any files.
      if (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Android")
        list(APPEND cryptopp_SOURCES ${SRC_DIR}/aes_armv4.S)
        list(APPEND cryptopp_SOURCES ${SRC_DIR}/sha1_armv4.S)
        list(APPEND cryptopp_SOURCES ${SRC_DIR}/sha256_armv4.S)
        list(APPEND cryptopp_SOURCES ${SRC_DIR}/sha512_armv4.S)

        set_source_files_properties(${SRC_DIR}/aes_armv4.S PROPERTIES LANGUAGE CXX)
        set_source_files_properties(${SRC_DIR}/sha1_armv4.S PROPERTIES LANGUAGE CXX)
        set_source_files_properties(${SRC_DIR}/sha256_armv4.S PROPERTIES LANGUAGE CXX)
        set_source_files_properties(${SRC_DIR}/sha512_armv4.S PROPERTIES LANGUAGE CXX)

        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
          set_source_files_properties(${SRC_DIR}/aes_armv4.S PROPERTIES COMPILE_FLAGS "-march=armv7-a -mthumb -mfpu=neon -Wa,--noexecstack")
        else ()
          set_source_files_properties(${SRC_DIR}/aes_armv4.S PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon -Wa,--noexecstack")
        endif ()

        set_source_files_properties(${SRC_DIR}/sha1_armv4.S PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon -Wa,--noexecstack")
        set_source_files_properties(${SRC_DIR}/sha256_armv4.S PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon -Wa,--noexecstack")
        set_source_files_properties(${SRC_DIR}/sha512_armv4.S PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon -Wa,--noexecstack")
      endif ()

      set_source_files_properties(${SRC_DIR}/aria_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/blake2s_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/blake2b_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/chacha_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/cham_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/crc_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/lea_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/gcm_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/rijndael_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/neon_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/sha_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/simon128_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/speck128_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/sm4_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
    else ()
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_ARM_NEON")
    endif ()

  elseif (CRYPTOPP_PPC32 OR CRYPTOPP_PPC64)

    # XLC requires -qaltivec in addition to Arch or CPU option
    # Disable POWER9 due to https://github.com/weidai11/cryptopp/issues/986.
    if (CMAKE_CXX_COMPILER MATCHES "xlC")
      set(CRYPTOPP_ALTIVEC_FLAGS "-qaltivec")
      set(CRYPTOPP_POWER4_FLAGS "-qarch=pwr4 -qaltivec")
      set(CRYPTOPP_POWER5_FLAGS "-qarch=pwr5 -qaltivec")
      set(CRYPTOPP_POWER6_FLAGS "-qarch=pwr6 -qaltivec")
      set(CRYPTOPP_POWER7_VSX_FLAG  "-qarch=pwr7 -qvsx -qaltivec")
      set(CRYPTOPP_POWER7_PWR_FLAGS "-qarch=pwr7 -qaltivec")
      set(CRYPTOPP_POWER8_FLAGS "-qarch=pwr8 -qaltivec")
      #set(CRYPTOPP_POWER9_FLAGS "-qarch=pwr9 -qaltivec")
    else ()
      set(CRYPTOPP_ALTIVEC_FLAGS "-maltivec")
      set(CRYPTOPP_POWER7_VSX_FLAGS "-mcpu=power7 -mvsx")
      set(CRYPTOPP_POWER7_PWR_FLAGS "-mcpu=power7")
      set(CRYPTOPP_POWER8_FLAGS "-mcpu=power8")
      #set(CRYPTOPP_POWER9_FLAGS "-mcpu=power9")
    endif ()

    CheckCompileLinkOption("${CRYPTOPP_ALTIVEC_FLAGS}" PPC_ALTIVEC_FLAG
                           "${TEST_PROG_DIR}/test_ppc_altivec.cpp")

    # Hack for XLC. Find the lowest PWR architecture.
    if (CMAKE_CXX_COMPILER MATCHES "xlC")
      if (NOT PPC_ALTIVEC_FLAG)
        CheckCompileLinkOption("${CRYPTOPP_POWER4_FLAGS}" PPC_POWER4_FLAG
                               "${TEST_PROG_DIR}/test_ppc_altivec.cpp")
        if (PPC_POWER4_FLAG)
          set(PPC_ALTIVEC_FLAG 1)
          set(CRYPTOPP_ALTIVEC_FLAGS "${CRYPTOPP_POWER4_FLAGS}")
        endif ()
      endif ()
      if (NOT PPC_ALTIVEC_FLAG)
        CheckCompileLinkOption("${CRYPTOPP_POWER5_FLAGS}" PPC_POWER5_FLAG
                               "${TEST_PROG_DIR}/test_ppc_altivec.cpp")
        if (PPC_POWER5_FLAG)
          set(PPC_ALTIVEC_FLAG 1)
          set(CRYPTOPP_ALTIVEC_FLAGS "${CRYPTOPP_POWER5_FLAGS}")
        endif ()
      endif ()
      if (NOT PPC_ALTIVEC_FLAG)
        CheckCompileLinkOption("${CRYPTOPP_POWER6_FLAGS}" PPC_POWER6_FLAG
                               "${TEST_PROG_DIR}/test_ppc_altivec.cpp")
        if (PPC_POWER6_FLAG)
          set(PPC_ALTIVEC_FLAG 1)
          set(CRYPTOPP_ALTIVEC_FLAGS "${CRYPTOPP_POWER6_FLAGS}")
        endif ()
      endif ()
    endif ()

    # Hack for XLC and GCC. Find the right combination for PWR7 and the VSX unit.
    CheckCompileLinkOption("${CRYPTOPP_POWER7_VSX_FLAGS}" PPC_POWER7_FLAG
                           "${TEST_PROG_DIR}/test_ppc_power7.cpp")
    if (PPC_POWER7_FLAG)
      set (CRYPTOPP_POWER7_FLAGS "${CRYPTOPP_POWER7_VSX_FLAGS}")
    else ()
      CheckCompileLinkOption("${CRYPTOPP_POWER7_PWR_FLAGS}" PPC_POWER7_FLAG
                             "${TEST_PROG_DIR}/test_ppc_power7.cpp")
      if (PPC_POWER7_FLAG)
        set (CRYPTOPP_POWER7_FLAGS "${CRYPTOPP_POWER7_PWR_FLAGS}")
      endif ()
    endif ()

    CheckCompileLinkOption("${CRYPTOPP_POWER8_FLAGS}" PPC_POWER8_FLAG
                           "${TEST_PROG_DIR}/test_ppc_power8.cpp")

    # Disable POWER9 due to https://github.com/weidai11/cryptopp/issues/986.
    #CheckCompileLinkOption("${CRYPTOPP_POWER9_FLAGS}" PPC_POWER9_FLAG
    #                       "${TEST_PROG_DIR}/test_ppc_power9.cpp")

    #if (PPC_POWER9_FLAG AND NOT DISABLE_POWER9)
    #  set_source_files_properties(${SRC_DIR}/ppc_power9.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER9_FLAGS})
    #endif ()

    if (PPC_POWER8_FLAG AND NOT DISABLE_POWER8)
      set_source_files_properties(${SRC_DIR}/ppc_power8.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      #set_source_files_properties(${SRC_DIR}/aria_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/blake2b_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/cham_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      #set_source_files_properties(${SRC_DIR}/crc_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/gcm_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/gf2n_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/lea_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/rijndael_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/sha_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/shacal2_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/simon128_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/speck128_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
    endif ()

    if (PPC_POWER7_FLAG AND NOT DISABLE_POWER7)
      set_source_files_properties(${SRC_DIR}/ppc_power7.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER7_FLAGS})
    endif ()

    if (PPC_ALTIVEC_FLAG AND NOT DISABLE_ALTIVEC)
      set_source_files_properties(${SRC_DIR}/ppc_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_ALTIVEC_FLAGS})
      set_source_files_properties(${SRC_DIR}/blake2s_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_ALTIVEC_FLAGS})
      set_source_files_properties(${SRC_DIR}/chacha_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_ALTIVEC_FLAGS})
    endif ()

    # Drop to Altivec if Power8 unavailable
    if (NOT PPC_POWER8_FLAG)
      if (PPC_ALTIVEC_FLAG)
        set_source_files_properties(${SRC_DIR}/gcm_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_ALTIVEC_FLAGS})
      endif ()
    endif ()

    if (NOT PPC_ALTIVEC_FLAG)
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_ALTIVEC")
    elseif (NOT PPC_POWER7_FLAG)
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_POWER7")
    elseif (NOT PPC_POWER8_FLAG)
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_POWER8")
    #elseif (NOT PPC_POWER9_FLAG)
    #  list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_POWER9")
    endif ()

  endif ()
endif ()

# New as of Pull Request 461, http://github.com/weidai11/cryptopp/pull/461.
if (CMAKE_CXX_COMPILER_ID STREQUAL "SunPro")

  if (CRYPTOPP_AMD64 OR CRYPTOPP_I386)

    CheckCompileLinkOption("-xarch=sse2" CRYPTOPP_X86_SSE2
                           "${TEST_PROG_DIR}/test_x86_sse2.cpp")
    CheckCompileLinkOption("-xarch=ssse3" CRYPTOPP_X86_SSSE3
                           "${TEST_PROG_DIR}/test_x86_ssse3.cpp")
    CheckCompileLinkOption("-xarch=sse4_1" CRYPTOPP_X86_SSE41
                           "${TEST_PROG_DIR}/test_x86_sse41.cpp")
    CheckCompileLinkOption("-xarch=sse4_2" CRYPTOPP_X86_SSE42
                           "${TEST_PROG_DIR}/test_x86_sse42.cpp")
    CheckCompileLinkOption("-xarch=aes" CRYPTOPP_X86_CLMUL
                           "${TEST_PROG_DIR}/test_x86_clmul.cpp")
    CheckCompileLinkOption("-xarch=aes" CRYPTOPP_X86_AES
                           "${TEST_PROG_DIR}/test_x86_aes.cpp")
    CheckCompileLinkOption("-xarch=avx" CRYPTOPP_X86_AVX
                           "${TEST_PROG_DIR}/test_x86_avx.cpp")
    CheckCompileLinkOption("-xarch=avx2" CRYPTOPP_X86_AVX2
                           "${TEST_PROG_DIR}/test_x86_avx2.cpp")
    CheckCompileLinkOption("-xarch=sha" CRYPTOPP_X86_SHA
                           "${TEST_PROG_DIR}/test_x86_sha.cpp")

    # Each -xarch=XXX options must be added to LDFLAGS if the option is used during a compile.
    set(XARCH_LDFLAGS "")

    if (CRYPTOPP_X86_SSE2 AND NOT DISABLE_ASM)
      set_source_files_properties(${SRC_DIR}/sse_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=sse2")
      set_source_files_properties(${SRC_DIR}/chacha_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=sse2")
      set(XARCH_LDFLAGS "-xarch=sse2")
    endif ()
    if (CRYPTOPP_X86_SSSE3 AND NOT DISABLE_SSSE3)
      set_source_files_properties(${SRC_DIR}/aria_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=ssse3")
      set_source_files_properties(${SRC_DIR}/cham_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=ssse3")
      set_source_files_properties(${SRC_DIR}/lea_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=ssse3")
      set_source_files_properties(${SRC_DIR}/simon128_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=ssse3")
      set_source_files_properties(${SRC_DIR}/speck128_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=ssse3")
      set(XARCH_LDFLAGS "${XARCH_LDFLAGS} -xarch=ssse3")
      if (CRYPTOPP_X86_SSE41 AND NOT DISABLE_SSE4)
        set_source_files_properties(${SRC_DIR}/blake2s_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=sse4_1")
        set_source_files_properties(${SRC_DIR}/blake2b_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=sse4_1")
        set(XARCH_LDFLAGS "${XARCH_LDFLAGS} -xarch=sse4_1")
      endif ()
      if (CRYPTOPP_X86_SSE42 AND NOT DISABLE_SSE4)
        set_source_files_properties(${SRC_DIR}/crc_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=sse4_2")
        set(XARCH_LDFLAGS "${XARCH_LDFLAGS} -xarch=sse4_2")
        if (CRYPTOPP_X86_CLMUL AND NOT DISABLE_CLMUL)
          set_source_files_properties(${SRC_DIR}/gcm_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=aes")
          set_source_files_properties(${SRC_DIR}/gf2n_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=aes")
        endif ()
        if (CRYPTOPP_X86_AES AND NOT DISABLE_AES)
          set_source_files_properties(${SRC_DIR}/rijndael_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=aes")
          set_source_files_properties(${SRC_DIR}/sm4_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=aes")
          set(XARCH_LDFLAGS "${XARCH_LDFLAGS} -xarch=aes")
        endif ()
        #if (CRYPTOPP_X86_AVX AND NOT DISABLE_AVX)
        #  set_source_files_properties(${SRC_DIR}/XXX_avx.cpp PROPERTIES COMPILE_FLAGS "-xarch=avx2")
        #  set(XARCH_LDFLAGS "${XARCH_LDFLAGS} -xarch=avx")
        #endif ()
        if (CRYPTOPP_X86_AVX2 AND NOT DISABLE_AVX2)
          set_source_files_properties(${SRC_DIR}/chacha_avx.cpp PROPERTIES COMPILE_FLAGS "-xarch=avx2")
          set(XARCH_LDFLAGS "${XARCH_LDFLAGS} -xarch=avx2")
        endif ()
        if (CRYPTOPP_X86_SHA AND NOT DISABLE_SHA)
          set_source_files_properties(${SRC_DIR}/sha_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=sha")
          set_source_files_properties(${SRC_DIR}/shacal2_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=sha")
          set(XARCH_LDFLAGS "${XARCH_LDFLAGS} -xarch=sha")
        endif ()
      endif ()
    endif ()

    # https://stackoverflow.com/a/6088646/608639
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${XARCH_LDFLAGS} -M${SRC_DIR}/cryptopp.mapfile")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${XARCH_LDFLAGS} -M${SRC_DIR}/cryptopp.mapfile")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${XARCH_LDFLAGS} -M${SRC_DIR}/cryptopp.mapfile")

  # elseif (CRYPTOPP_SPARC OR CRYPTOPP_SPARC64)

  endif ()
endif ()

#============================================================================
# Compile targets
#============================================================================

# Work around the archaic versions of cmake that do not support
# target_compile_xxxx commands
# !!! DO NOT try to use the old way for newer version - it does not work !!!
function(cryptopp_target_compile_properties target)
  # CMake >= 2.8.11
  if (NOT ${CMAKE_VERSION} VERSION_LESS "2.8.11")
    target_compile_definitions(${target} PUBLIC ${CRYPTOPP_COMPILE_DEFINITIONS})
  else()
    string (REPLACE ";" " " PROP_STR "${CRYPTOPP_COMPILE_DEFINITIONS}")
    set_target_properties(${target} PROPERTIES COMPILE_DEFINITIONS "${CRYPTOPP_COMPILE_DEFINITIONS}")
  endif()
  # CMake >= 2.8.12
  if (NOT ${CMAKE_VERSION} VERSION_LESS "2.8.12")
    target_compile_options(${target} PUBLIC ${CRYPTOPP_COMPILE_OPTIONS})
  else()
    string (REPLACE ";" " " PROP_STR "${CRYPTOPP_COMPILE_OPTIONS}")
    set_target_properties(${target} PROPERTIES COMPILE_FLAGS "${PROP_STR}")
  endif()
endfunction()

set(cryptopp_LIBRARY_SOURCES ${cryptopp_SOURCES_ASM})
if (USE_INTERMEDIATE_OBJECTS_TARGET AND NOT ${CMAKE_VERSION} VERSION_LESS "2.8.8")
  add_library(cryptopp-object OBJECT ${cryptopp_SOURCES})
  cryptopp_target_compile_properties(cryptopp-object)

  list(APPEND cryptopp_LIBRARY_SOURCES
    $<TARGET_OBJECTS:cryptopp-object>
    )
else ()
  list(APPEND cryptopp_LIBRARY_SOURCES
    ${cryptopp_SOURCES}
    )
endif ()

if (BUILD_STATIC)
  add_library(cryptopp-static STATIC ${cryptopp_LIBRARY_SOURCES})
  cryptopp_target_compile_properties(cryptopp-static)
  # CMake >= 2.8.11
  if (NOT ${CMAKE_VERSION} VERSION_LESS "2.8.11")
    target_include_directories(cryptopp-static PUBLIC $<BUILD_INTERFACE:${SRC_DIR}> $<INSTALL_INTERFACE:include>)
  else ()
    set_target_properties(cryptopp-static PROPERTIES INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${SRC_DIR}> $<INSTALL_INTERFACE:include>")
  endif ()
endif ()

if (BUILD_SHARED)
  add_library(cryptopp-shared SHARED ${cryptopp_LIBRARY_SOURCES})
  cryptopp_target_compile_properties(cryptopp-shared)
  if (NOT ${CMAKE_VERSION} VERSION_LESS "2.8.11")
    target_include_directories(cryptopp-shared PUBLIC $<BUILD_INTERFACE:${SRC_DIR}> $<INSTALL_INTERFACE:include>)
  else ()
    set_target_properties(cryptopp-shared PROPERTIES INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${SRC_DIR}> $<INSTALL_INTERFACE:include>")
  endif ()
endif ()

# Set filenames for targets to be "cryptopp"
if (NOT MSVC)
  set(COMPAT_VERSION ${cryptopp_VERSION_MAJOR}.${cryptopp_VERSION_MINOR})

  if (BUILD_STATIC)
    set_target_properties(cryptopp-static
        PROPERTIES
        OUTPUT_NAME cryptopp)
  endif ()
  if (BUILD_SHARED)
    set_target_properties(cryptopp-shared
        PROPERTIES
        SOVERSION ${COMPAT_VERSION}
        OUTPUT_NAME cryptopp)
  endif ()
endif ()

# Add alternate ways to invoke the build for the shared library that are
# similar to how the crypto++ 'make' tool works.
# see https://github.com/noloader/cryptopp-cmake/issues/32
if (BUILD_STATIC)
  add_custom_target(static DEPENDS cryptopp-static)
endif ()
if (BUILD_SHARED)
  add_custom_target(shared DEPENDS cryptopp-shared)
  add_custom_target(dynamic DEPENDS cryptopp-shared)
endif ()

#============================================================================
# Third-party libraries
#============================================================================

if (WIN32)
  if (BUILD_STATIC)
    target_link_libraries(cryptopp-static ws2_32)
  endif ()
  if (BUILD_SHARED)
    target_link_libraries(cryptopp-shared ws2_32)
  endif ()
endif ()

# This may need to be expanded to "Solaris"
if (CRYPTOPP_SOLARIS)
  if (BUILD_STATIC)
    target_link_libraries(cryptopp-static nsl socket)
  endif ()
  if (BUILD_SHARED)
    target_link_libraries(cryptopp-shared nsl socket)
  endif ()
endif ()

find_package(Threads)
if (BUILD_STATIC)
  target_link_libraries(cryptopp-static ${CMAKE_THREAD_LIBS_INIT})
endif ()
if (BUILD_SHARED)
  target_link_libraries(cryptopp-shared ${CMAKE_THREAD_LIBS_INIT})
endif ()


#============================================================================
# Setup OpenMP
#============================================================================
if (${CMAKE_VERSION} VERSION_GREATER "3.1" AND USE_OPENMP)
  find_package(OpenMP)

  if (OPENMP_FOUND OR OPENMP_CXX_FOUND)
      message(STATUS "OpenMP: Found libomp without any special flags")
  endif()

  # If OpenMP wasn't found, try if we can find it in the default Macports location
  if((NOT OPENMP_FOUND) AND (NOT OPENMP_CXX_FOUND) AND EXISTS "/opt/local/lib/libomp/libomp.dylib") # older cmake uses OPENMP_FOUND, newer cmake also sets OPENMP_CXX_FOUND, homebrew installations seem only to get the latter set.
      set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/local/include/libomp/")
      set(OpenMP_CXX_LIB_NAMES omp)
      set(OpenMP_omp_LIBRARY /opt/local/lib/libomp/libomp.dylib)

      find_package(OpenMP)
      if (OPENMP_FOUND OR OPENMP_CXX_FOUND)
          message(STATUS "OpenMP: Found libomp in macports default location.")
      else()
          message(FATAL_ERROR "OpenMP: Didn't find libomp. Tried macports default location but also didn't find it.")
      endif()
  endif()

  # If OpenMP wasn't found, try if we can find it in the default Homebrew location
  if((NOT OPENMP_FOUND) AND (NOT OPENMP_CXX_FOUND) AND EXISTS "/usr/local/opt/libomp/lib/libomp.dylib")
      set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
      set(OpenMP_CXX_LIB_NAMES omp)
      set(OpenMP_omp_LIBRARY /usr/local/opt/libomp/lib/libomp.dylib)

      find_package(OpenMP)
      if (OPENMP_FOUND OR OPENMP_CXX_FOUND)
          message(STATUS "OpenMP: Found libomp in homebrew default location.")
      else()
          message(FATAL_ERROR "OpenMP: Didn't find libomp. Tried homebrew default location but also didn't find it.")
      endif()
  endif()

  set(Additional_OpenMP_Libraries_Workaround "")

  # Workaround because older cmake on apple doesn't support FindOpenMP
  if((NOT OPENMP_FOUND) AND (NOT OPENMP_CXX_FOUND))
      if((APPLE AND ((CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")))
              AND ((CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "7.0") AND (CMAKE_VERSION VERSION_LESS "3.12.0")))
          message(STATUS "OpenMP: Applying workaround for OSX OpenMP with old cmake that doesn't have FindOpenMP")
          set(OpenMP_CXX_FLAGS "-Xclang -fopenmp")
          set(Additional_OpenMP_Libraries_Workaround "-lomp")
      else()
          message(FATAL_ERROR "OpenMP: Did not find OpenMP. Build without USE_OPENMP if you want to allow this.")
      endif()
  endif()

  if(NOT TARGET OpenMP::OpenMP_CXX)
      # We're on cmake < 3.9, handle behavior of the old FindOpenMP implementation
      message(STATUS "OpenMP: Applying workaround for old CMake that doesn't define FindOpenMP using targets")
      add_library(OpenMP_TARGET INTERFACE)
      add_library(OpenMP::OpenMP_CXX ALIAS OpenMP_TARGET)
      target_compile_options(OpenMP_TARGET INTERFACE ${OpenMP_CXX_FLAGS}) # add to all targets depending on this
      find_package(Threads REQUIRED)
      target_link_libraries(OpenMP_TARGET INTERFACE Threads::Threads)
      target_link_libraries(OpenMP_TARGET INTERFACE ${OpenMP_CXX_FLAGS} ${Additional_OpenMP_Libraries_Workaround})
  endif()

  if (BUILD_STATIC)
    target_link_libraries(cryptopp-static ${OpenMP_CXX_FLAGS}) # Workaround for Ubuntu 18.04 that otherwise doesn't set -fopenmp for linking
    target_link_libraries(cryptopp-static OpenMP::OpenMP_CXX)
  endif()
  if (BUILD_SHARED)
    target_link_libraries(cryptopp-shared ${OpenMP_CXX_FLAGS}) # Workaround for Ubuntu 18.04 that otherwise doesn't set -fopenmp for linking
    target_link_libraries(cryptopp-shared OpenMP::OpenMP_CXX)
  endif()
endif()

#============================================================================
# Tests
#============================================================================

enable_testing()
if (BUILD_TESTING)
  add_executable(cryptest ${cryptopp_SOURCES_TEST})
  target_link_libraries(cryptest cryptopp-static)

  # Setting "cryptest" binary name to "cryptest.exe"
  if (NOT (WIN32 OR CYGWIN))
    set_target_properties(cryptest PROPERTIES OUTPUT_NAME cryptest.exe)
  endif ()
  if (NOT TARGET cryptest.exe)
    add_custom_target(cryptest.exe)
    add_dependencies(cryptest.exe cryptest)
  endif ()

  file(COPY ${SRC_DIR}/TestData DESTINATION ${PROJECT_BINARY_DIR})
  file(COPY ${SRC_DIR}/TestVectors DESTINATION ${PROJECT_BINARY_DIR})

  add_test(NAME build_cryptest COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target cryptest)
  add_test(NAME cryptest COMMAND $<TARGET_FILE:cryptest> v)
  set_tests_properties(cryptest PROPERTIES DEPENDS build_cryptest)
endif ()

#============================================================================
# Doxygen documentation
#============================================================================

if (BUILD_DOCUMENTATION)
  find_package(Doxygen REQUIRED)

  set(in_source_DOCS_DIR "${SRC_DIR}/html-docs")
  set(out_source_DOCS_DIR "${PROJECT_BINARY_DIR}/html-docs")

  add_custom_target(docs ALL
      COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile -d CRYPTOPP_DOXYGEN_PROCESSING
      WORKING_DIRECTORY ${SRC_DIR}
      SOURCES ${SRC_DIR}/Doxyfile
      )

  if (NOT ${in_source_DOCS_DIR} STREQUAL ${out_source_DOCS_DIR})
    add_custom_command(
        TARGET docs POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${in_source_DOCS_DIR}" "${out_source_DOCS_DIR}"
        COMMAND ${CMAKE_COMMAND} -E remove_directory "${in_source_DOCS_DIR}"
    )
  endif ()
endif ()

#============================================================================
# Install
#============================================================================

set(export_name "cryptopp-targets")

# Runtime package
if (BUILD_SHARED)
  export(TARGETS cryptopp-shared FILE ${export_name}.cmake )
  install(
      TARGETS cryptopp-shared
      EXPORT ${export_name}
      DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif ()

# Development package
if (BUILD_STATIC)
  export(TARGETS cryptopp-static FILE ${export_name}.cmake )
  install(TARGETS cryptopp-static EXPORT ${export_name} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif ()
install(FILES ${cryptopp_HEADERS} DESTINATION include/cryptopp)

# CMake Package
if (NOT CMAKE_VERSION VERSION_LESS 2.8.8)
  include(CMakePackageConfigHelpers)
  write_basic_package_version_file("${PROJECT_BINARY_DIR}/cryptopp-config-version.cmake" VERSION ${cryptopp_VERSION_MAJOR}.${cryptopp_VERSION_MINOR}.${cryptopp_VERSION_PATCH} COMPATIBILITY SameMajorVersion)
  install(FILES cryptopp-config.cmake ${PROJECT_BINARY_DIR}/cryptopp-config-version.cmake DESTINATION "lib/cmake/cryptopp")
  install(EXPORT ${export_name} DESTINATION "lib/cmake/cryptopp")
endif ()

# Tests
if (BUILD_TESTING)
  install(TARGETS cryptest DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(DIRECTORY ${SRC_DIR}/TestData DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cryptopp)
  install(DIRECTORY ${SRC_DIR}/TestVectors DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cryptopp)
endif ()

# Documentation
if (BUILD_DOCUMENTATION)
  install(DIRECTORY "${out_source_DOCS_DIR}" DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif ()

# Print a configuration summary. We want CXX and CXXFLAGS, but they are not includd in ALL.
if (CRYPTOPP_I386)
  message(STATUS "Platform: i386/i686")
elseif (CRYPTOPP_AMD64)
  message(STATUS "Platform: x86_64")
elseif (CRYPTOPP_ARM32)
  message(STATUS "Platform: ARM-32")
elseif (CRYPTOPP_ARMV8)
  message(STATUS "Platform: ARMv8")
elseif (CRYPTOPP_SPARC)
  message(STATUS "Platform: Sparc")
elseif (CRYPTOPP_SPARC64)
  message(STATUS "Platform: Sparc64")
elseif (CRYPTOPP_PPC32)
  message(STATUS "Platform: PowerPC")
elseif (CRYPTOPP_PPC64)
  message(STATUS "Platform: PowerPC-64")
elseif (CRYPTOPP_MINGW32)
  message(STATUS "Platform: MinGW-32")
elseif (CRYPTOPP_MINGW64)
  message(STATUS "Platform: MinGW-64")
endif ()
if (CRYPTOPP_ARMV7A_NEON)
  message(STATUS "NEON: TRUE")
endif ()
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler options: ${CMAKE_CXX_FLAGS} ${CRYPTOPP_COMPILE_OPTIONS}")
message(STATUS "Compiler definitions: ${CRYPTOPP_COMPILE_DEFINITIONS}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# The home path of ISL
# Required!
set(ISL_HOME "")

if(WITH_XPU_CADA)
  # The xtrans/clang toolchain's OpenMP flag (-fopenmp=libomp) is not
  # recognized by the plain system gcc used to configure some
  # ExternalProject dependencies (e.g. yaml-cpp), which leaves
  # CMAKE_CXX_FLAGS/CMAKE_C_FLAGS clean. Skip CINN's own OpenMP detection
  # for WITH_XPU_CADA builds to avoid polluting those flags.
  set(USE_OPENMP "")
else()
  set(USE_OPENMP "intel")
endif()

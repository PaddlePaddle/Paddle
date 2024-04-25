add_subdirectory(string)

# if(NOT WITH_GFLAGS)
#   cc_library(paddle_flags SRCS flags_native.cc)
#   cc_test(
#     flags_native_test
#     SRCS flags_native_test.cc
#     DEPS paddle_flags)
# endif()

if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
  cc_library(
    pybind_util
    SRCS pybind.cc
    DEPS phi common)
endif()

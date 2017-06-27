# ddim lib
cc_library(ddim SRCS ddim.cc)
cc_test(ddim_test SRCS ddim_test.cc DEPS ddim)

nv_test(dim_test SRCS dim_test.cu DEPS ddim)

cc_test(variable_test SRCS variable_test.cc)

# scope lib
cc_library(scope SRCS scope.cc)
cc_test(scope_test SRCS scope_test.cc DEPS scope)

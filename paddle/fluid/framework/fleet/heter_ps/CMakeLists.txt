nv_library(heter_comm SRCS heter_comm.h feature_value.h heter_resource.cc
heter_resource.h hashtable.h DEPS cub device_context)
nv_test(test_heter_comm SRCS test_heter_comm.cu feature_value.h DEPS
heter_comm)

nv_library(heter_ps SRCS heter_ps.cu DEPS heter_comm)

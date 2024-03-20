core_gather_headers()

gather_srcs(cinnapi_src SRCS
        hip_util.cc
        hip_module.cc
        hip_intrinsics.cc
        hip_intrinsics_reduce.cc
        hip_backend_api.cc
)

# 
# cinn_rocm_test(test_hip_module SRCS hip_module_test.cc DEPS cinncore)
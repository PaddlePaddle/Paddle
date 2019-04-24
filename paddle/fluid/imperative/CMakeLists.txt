if(WITH_PYTHON)
cc_library(layer SRCS layer.cc DEPS proto_desc operator device_context blas pybind profiler)
cc_library(tracer SRCS tracer.cc DEPS proto_desc device_context pybind profiler)
cc_library(engine SRCS engine.cc)
cc_library(imperative_profiler SRCS profiler.cc)
cc_library(nccl_context SRCS nccl_context.cc DEPS device_context)

cc_test(nccl_context_test SRCS nccl_context_test.cc  DEPS nccl_context)
endif()

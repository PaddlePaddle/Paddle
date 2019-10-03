cc_library(imperative_flag SRCS flags.cc DEPS gflags) 

cc_library(prepared_operator SRCS prepared_operator.cc DEPS proto_desc operator device_context lod_tensor selected_rows var_type_traits op_kernel_type data_transform)
cc_library(layer SRCS layer.cc DEPS prepared_operator math_function imperative_flag variable_helper op_registry)
cc_library(gradient_accumulator SRCS gradient_accumulator.cc DEPS blas operator lod_tensor selected_rows var_type_traits layer)
cc_library(tracer SRCS tracer.cc DEPS layer engine)
cc_library(engine SRCS engine.cc DEPS layer gradient_accumulator)
cc_library(imperative_profiler SRCS profiler.cc)
cc_library(nccl_context SRCS nccl_context.cc DEPS device_context)

add_subdirectory(tests)

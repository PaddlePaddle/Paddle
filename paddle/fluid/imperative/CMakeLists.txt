cc_library(imperative_flag SRCS flags.cc DEPS gflags) 

cc_library(prepared_operator SRCS prepared_operator.cc DEPS proto_desc operator device_context lod_tensor selected_rows var_type_traits op_registry pybind)
cc_library(layer SRCS layer.cc DEPS prepared_operator math_function imperative_flag variable_helper op_registry proto_desc operator device_context blas pybind profiler)
cc_library(gradient_accumulator SRCS gradient_accumulator.cc DEPS blas operator lod_tensor selected_rows var_type_traits layer)
cc_library(tracer SRCS tracer.cc DEPS layer engine pybind op_registry)
cc_library(engine SRCS engine.cc DEPS layer gradient_accumulator)
cc_library(imperative_profiler SRCS profiler.cc)
cc_library(nccl_context SRCS nccl_context.cc DEPS device_context)
cc_binary(tracer_test SRCS tracer_test.cc DEPS tracer layer  op_registry prepared_operator python pybind ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS}  profiler)

add_subdirectory(tests)

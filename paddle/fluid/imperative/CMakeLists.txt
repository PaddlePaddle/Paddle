cc_library(imperative_flag SRCS flags.cc DEPS gflags) 

cc_library(prepared_operator SRCS prepared_operator.cc DEPS proto_desc operator device_context lod_tensor selected_rows var_type_traits op_kernel_type data_transform)
cc_library(layer SRCS layer.cc DEPS prepared_operator math_function imperative_flag variable_helper op_registry)
cc_library(gradient_accumulator SRCS gradient_accumulator.cc DEPS blas operator lod_tensor selected_rows selected_rows_functor var_type_traits layer)
add_subdirectory(jit)

cc_library(tracer SRCS tracer.cc DEPS layer engine program_desc_tracer)
cc_library(basic_engine SRCS basic_engine.cc DEPS layer gradient_accumulator)
cc_library(engine SRCS basic_engine.cc partial_grad_engine.cc DEPS layer gradient_accumulator)
cc_library(imperative_profiler SRCS profiler.cc)
if(NOT WIN32)
    if(WITH_NCCL)
        cc_library(nccl_context SRCS nccl_context.cc DEPS collective_helper device_context)
    endif()
    cc_library(data_loader SRCS data_loader.cc DEPS enforce)
endif(NOT WIN32)

add_subdirectory(tests)

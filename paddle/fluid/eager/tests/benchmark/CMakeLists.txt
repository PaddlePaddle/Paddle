cc_test(benchmark_eager SRCS benchmark_eager.cc DEPS eager_api function_api backward autograd_meta grad_node_info top)
cc_test(benchmark_fluid SRCS benchmark_fluid.cc DEPS tracer layer proto_desc operator op_registry variable_helper scale_op memcpy)

cc_library(op_graph_view SRCS op_graph_view.cc DEPS op_handle_base)
cc_library(while_op_eager_deletion_pass SRCS while_op_eager_deletion_pass.cc DEPS while_op_helper graph_helper pass computation_op_handle)
cc_library(reference_count_pass_helper SRCS reference_count_pass_helper.cc DEPS garbage_collector computation_op_handle var_handle)
cc_library(reference_count_pass SRCS reference_count_pass.cc DEPS computation_op_handle graph graph_helper pass op_graph_view reference_count_pass_helper)

if(WITH_GPU)
    cc_library(memory_optimize_helper SRCS memory_optimize_helper.cc DEPS graph graph_helper gpu_info)
else()
    cc_library(memory_optimize_helper SRCS memory_optimize_helper.cc DEPS graph graph_helper cpu_info)
endif()

cc_library(memory_optimize_pass SRCS memory_optimize_pass.cc DEPS memory_optimize_helper pass)
cc_library(inplace_op_pass SRCS inplace_op_pass.cc DEPS memory_optimize_pass op_info)

cc_test(memory_optimize_helper_test SRCS memory_optimize_helper_test.cc memory_optimize_helper.cc DEPS framework_proto graph graph_helper op_registry)

cc_library(eager_deletion_pass SRCS eager_deletion_pass.cc DEPS computation_op_handle eager_deletion_op_handle graph graph_helper pass while_op_eager_deletion_pass reference_count_pass_helper)
cc_library(record_skip_memory_opt_vars_pass SRCS record_skip_memory_opt_vars_pass.cc DEPS graph graph_helper)

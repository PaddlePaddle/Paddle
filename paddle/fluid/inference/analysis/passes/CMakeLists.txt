cc_library(
  ir_graph_build_pass
  SRCS ir_graph_build_pass.cc
  DEPS analysis_pass argument ir_pass_manager)
cc_library(
  ir_analysis_pass
  SRCS ir_analysis_pass.cc
  DEPS analysis_pass argument ir_pass_manager)
cc_library(
  memory_optim_pass
  SRCS memory_optimize_pass.cc
  DEPS analysis_pass zero_copy_tensor)
cc_library(
  convert_to_mixed_precision
  SRCS convert_to_mixed_precision.cc
  DEPS analysis_pass ir_graph_build_pass auto_mixed_precision_pass)
cc_library(
  ir_params_sync_among_devices_pass
  SRCS ir_params_sync_among_devices_pass.cc
  DEPS analysis_pass argument ir_pass_manager)
cc_library(
  ir_graph_to_program_pass
  SRCS ir_graph_to_program_pass.cc
  DEPS analysis_pass graph_to_program_pass)
cc_library(
  adjust_cudnn_workspace_size_pass
  SRCS adjust_cudnn_workspace_size_pass.cc
  DEPS analysis_pass graph_to_program_pass)
cc_library(
  inference_op_replace_pass
  SRCS inference_op_replace_pass.cc
  DEPS analysis_pass graph_to_program_pass)

cc_library(
  analysis_passes
  SRCS passes.cc
  DEPS ir_graph_build_pass
       ir_analysis_pass
       ir_params_sync_among_devices_pass
       adjust_cudnn_workspace_size_pass
       memory_optim_pass
       convert_to_mixed_precision
       inference_op_replace_pass
       ir_graph_to_program_pass)

set(analysis_deps
    ${analysis_deps} analysis_passes subgraph_detector
    CACHE INTERNAL "")

set(parsed_yaml_path ${PADDLE_BINARY_DIR}/paddle/phi/ops/yaml)
set(fwd_path ${parsed_yaml_path}/ops.parsed.yaml)
set(rev_path ${parsed_yaml_path}/backward.parsed.yaml)
set(fwd_pd_op_path ${parsed_yaml_path}/inconsistent/static_ops.parsed.yaml)
set(update_fwd_pd_op_path
    ${parsed_yaml_path}/inconsistent/update_ops.parsed.yaml)
set(rev_pd_op_path ${parsed_yaml_path}/inconsistent/static_backward.parsed.yaml)
set(fused_op_path ${parsed_yaml_path}/fused_ops.parsed.yaml)
set(fused_rev_op_path ${parsed_yaml_path}/fused_backward.parsed.yaml)

set(sparse_op_path ${parsed_yaml_path}/sparse_ops.parsed.yaml)
set(sparse_rev_op_path ${parsed_yaml_path}/sparse_backward.parsed.yaml)
set(prim_path
    ${PADDLE_SOURCE_DIR}/paddle/fluid/primitive/primitive/primitive.yaml)
set(templates_dir
    ${PADDLE_SOURCE_DIR}/paddle/fluid/primitive/codegen/templates/)
set(compat_path ${PADDLE_SOURCE_DIR}/paddle/phi/ops/yaml/op_compat.yaml)
set(destination_dir ${PADDLE_BINARY_DIR}/paddle/fluid/primitive/)
set(decomp_vjp_destination_dir
    ${PADDLE_BINARY_DIR}/paddle/fluid/pir/dialect/operator/ir/op_decomp_vjp.cc)
set(scripts
    ${PADDLE_SOURCE_DIR}/paddle/fluid/primitive/codegen/decomp_vjp_gen.py)

message("Automatic code generation for paddle/fluid/primitive")
execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/primitive/codegen
  COMMAND
    ${PYTHON_EXECUTABLE}
    ${PADDLE_SOURCE_DIR}/paddle/fluid/primitive/codegen/decomp_vjp_gen.py
    --fwd_path ${fwd_path} --rev_path ${rev_path} --fwd_pd_op_path
    ${fwd_pd_op_path} --update_fwd_pd_op_path ${update_fwd_pd_op_path}
    --rev_pd_op_path ${rev_pd_op_path} --prim_path ${prim_path} --fused_op_path
    ${fused_op_path} --fused_rev_op_path ${fused_rev_op_path} --sparse_op_path
    ${sparse_op_path} --sparse_rev_op_path ${sparse_rev_op_path} --templates_dir
    ${templates_dir} --compat_path ${compat_path} --destination_dir
    ${destination_dir} --decomp_vjp_destination_dir
    ${decomp_vjp_destination_dir}
  RESULT_VARIABLE _result)
if(${_result})
  message(
    FATAL_ERROR
      "Automatic code generation for paddle/fluid/primitive failed, exiting.")
endif()
message("Automatic code generation for paddle/fluid/primitive succeed.")
execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/primitive/codegen
  COMMAND
    ${PYTHON_EXECUTABLE}
    ${PADDLE_SOURCE_DIR}/paddle/fluid/primitive/codegen/decomp_rule_gen.py
    --fwd_path ${fwd_path} --fwd_pd_op_path ${fwd_pd_op_path} --templates_dir
    ${templates_dir} --compat_path ${compat_path} --destination_dir
    ${PADDLE_BINARY_DIR}/paddle/fluid/pir/dialect/operator/ir/op_decomp_rule.cc
  RESULT_VARIABLE _result)
if(${_result})
  message(
    FATAL_ERROR
      "Automatic code generation for build/paddle/fluid/pir/dialect/operator/ir/op_decomp_rule.cc failed."
  )
endif()
message("Automatic code generation for decomp interface succeed.")

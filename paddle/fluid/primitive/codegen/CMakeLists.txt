set(parsed_yaml_path
    "${PADDLE_BINARY_DIR}/paddle/fluid/operators/generator/parsed_ops")
set(fwd_path ${parsed_yaml_path}/ops.parsed.yaml)
set(rev_path ${parsed_yaml_path}/backward_ops.parsed.yaml)
set(fwd_pd_op_path
    ${PADDLE_BINARY_DIR}/paddle/fluid/pir/dialect/operator/ir/generated/ops.parsed.yaml
)
set(update_fwd_pd_op_path
    ${PADDLE_BINARY_DIR}/paddle/fluid/pir/dialect/operator/ir/generated/update_ops.parsed.yaml
)
set(rev_pd_op_path
    ${PADDLE_BINARY_DIR}/paddle/fluid/pir/dialect/operator/ir/generated/ops_backward.parsed.yaml
)
set(prim_path "${PADDLE_SOURCE_DIR}/paddle/fluid/primitive/primitive.yaml")
set(templates_dir
    "${PADDLE_SOURCE_DIR}/paddle/fluid/primitive/codegen/templates/")
set(compat_path "${PADDLE_SOURCE_DIR}/paddle/phi/api/yaml/op_compat.yaml")
set(destination_dir "${PADDLE_BINARY_DIR}/paddle/fluid/primitive/")
set(scripts "${PADDLE_SOURCE_DIR}/paddle/fluid/primitive/codegen/gen.py")

message("Automatic code generation for paddle/fluid/primitive")
execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/primitive/codegen
  COMMAND
    ${PYTHON_EXECUTABLE} ${scripts} --fwd_path ${fwd_path} --rev_path
    ${rev_path} --fwd_pd_op_path ${fwd_pd_op_path} --update_fwd_pd_op_path
    ${update_fwd_pd_op_path} --rev_pd_op_path ${rev_pd_op_path} --prim_path
    ${prim_path} --templates_dir ${templates_dir} --compat_path ${compat_path}
    --destination_dir ${destination_dir}
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
    ${PADDLE_SOURCE_DIR}/paddle/fluid/primitive/codegen/decomp_gen.py --fwd_path
    ${fwd_path} --fwd_pd_op_path ${fwd_pd_op_path} --templates_dir
    ${templates_dir} --compat_path ${compat_path} --destination_dir
    ${PADDLE_BINARY_DIR}/paddle/fluid/pir/dialect/operator/ir/op_decomp.cc
  RESULT_VARIABLE _result)
if(${_result})
  message(
    FATAL_ERROR
      "Automatic code generation for build/paddle/fluid/pir/dialect/operator/ir/op_decomp.cc failed."
  )
endif()
message("Automatic code generation for decomp interface succeed.")

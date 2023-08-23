set(fwd_path
    "${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parsed_ops/ops.parsed.yaml"
)
set(fwd_legacy_path
    "${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parsed_ops/legacy_ops.parsed.yaml"
)
set(rev_path
    "${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parsed_ops/backward_ops.parsed.yaml"
)
set(rev_legacy_path
    "${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parsed_ops/legacy_backward_ops.parsed.yaml"
)
set(prim_path "${PADDLE_SOURCE_DIR}/paddle/fluid/primitive/primitive.yaml")
set(templates_dir
    "${PADDLE_SOURCE_DIR}/paddle/fluid/primitive/codegen/templates/")
set(destination_dir "${PADDLE_SOURCE_DIR}/paddle/fluid/primitive/")
set(scripts "${PADDLE_SOURCE_DIR}/paddle/fluid/primitive/codegen/gen.py")

message("Automatic code generation for paddle/fluid/primitive")
execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/primitive/codegen
  COMMAND
    ${PYTHON_EXECUTABLE} ${scripts} --fwd_path ${fwd_path} --fwd_legacy_path
    ${fwd_legacy_path} --rev_path ${rev_path} --rev_legacy_path
    ${rev_legacy_path} --prim_path ${prim_path} --templates_dir ${templates_dir}
    --destination_dir ${destination_dir}
  RESULT_VARIABLE _result)
if(${_result})
  message(
    FATAL_ERROR
      "Automatic code generation for paddle/fluid/primitive failed, exiting.")
endif()
message("Automatic code generation for paddle/fluid/primitive succeed.")

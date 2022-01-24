set(api_yaml_path "${PADDLE_SOURCE_DIR}/python/paddle/utils/code_gen/api.yaml")
set(backward_yaml_path "${PADDLE_SOURCE_DIR}/python/paddle/utils/code_gen/backward.yaml")
set(tmp_forwards_cc_path "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/eager_generated/forwards/tmp_dygraph_functions.cc")
set(tmp_forwards_h_path "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/eager_generated/forwards/tmp_dygraph_functions.h")
set(tmp_nodes_cc_path "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/eager_generated/backwards/tmp_node.cc")
set(tmp_nodes_h_path "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/eager_generated/backwards/tmp_node.h")
set(forwards_cc_path "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.cc")
set(forwards_h_path "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h")
set(nodes_cc_path "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/eager_generated/backwards/node.cc")
set(nodes_h_path "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/eager_generated/backwards/node.h")

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/auto_code_generator/final_state_generator/eager_gen.py" 
            "--api_yaml_path=${api_yaml_path}"
            "--backward_yaml_path=${backward_yaml_path}"
            "--forwards_cc_path=${tmp_forwards_cc_path}" 
            "--forwards_h_path=${tmp_forwards_h_path}"
            "--nodes_cc_path=${tmp_nodes_cc_path}" 
            "--nodes_h_path=${tmp_nodes_h_path}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_forwards_cc_path} ${forwards_cc_path}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_forwards_h_path} ${forwards_h_path}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_nodes_cc_path} ${nodes_cc_path}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_nodes_h_path} ${nodes_h_path}
)

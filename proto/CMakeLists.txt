set(proto_filenames
    DataConfig.proto
    DataFormat.proto
    ModelConfig.proto
    ParameterConfig.proto
    ParameterService.proto
    TrainerConfig.proto
    OptimizerConfig.proto
    ParameterServerConfig.proto)

set(PROTO_GEN)
set(PROTO_GEN_PY)

foreach(filename ${proto_filenames})
    get_filename_component(base_filename ${filename} NAME_WE)
    set(CUR_PROTO_GEN_PY
        ${PROJ_ROOT}/paddle/python/paddle/proto/${base_filename}_pb2.py)
    set(PROTO_GEN_PY
        ${CUR_PROTO_GEN_PY}
        ${PROTO_GEN_PY})
    add_custom_command(OUTPUT ${CUR_PROTO_GEN_PY}
        COMMAND env ${py_env} ${PROTOBUF_PROTOC_EXECUTABLE} --python_out ${PROJ_ROOT}/python/paddle/proto
    --proto_path ${PROJ_ROOT}/proto ${PROJ_ROOT}/proto/${filename}
        DEPENDS ${filename} ${external_project_dependencies})
endforeach()

add_custom_target(gen_proto_py ALL DEPENDS ${PROTO_GEN_PY})

pb_cc_library(paddle_proto SRCS ${proto_filenames})

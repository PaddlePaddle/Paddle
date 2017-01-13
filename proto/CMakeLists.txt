set(proto_filenames
    DataConfig.proto
    DataFormat.proto
    ModelConfig.proto
    ParameterConfig.proto
    ParameterService.proto
    TrainerConfig.proto
    ParameterServerConfig.proto)

set(PROTO_GEN)
set(PROTO_GEN_PY)

foreach(filename ${proto_filenames})
    get_filename_component(base_filename ${filename} NAME_WE)
    set(CUR_PROTO_GEN
        ${CMAKE_CURRENT_BINARY_DIR}/${base_filename}.pb.h
        ${CMAKE_CURRENT_BINARY_DIR}/${base_filename}.pb.cc)
    set(PROTO_GEN
        ${PROTO_GEN}
        ${CUR_PROTO_GEN})
    add_custom_command(OUTPUT ${CUR_PROTO_GEN}
        COMMAND env ${py_env} ${PROTOBUF_PROTOC_EXECUTABLE} 
                  --cpp_out ${CMAKE_CURRENT_BINARY_DIR}
          --proto_path ${PROJ_ROOT}/proto ${PROJ_ROOT}/proto/${filename}
        DEPENDS ${filename} ${external_project_dependencies})

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

include_directories(${CMAKE_CURRENT_BINARY_DIR}/proto)

add_custom_target(gen_proto_cpp ALL DEPENDS ${PROTO_GEN})
add_custom_target(gen_proto_py ALL DEPENDS ${PROTO_GEN_PY})
add_library(paddle_proto STATIC
    ${PROTO_GEN})
target_include_directories(paddle_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

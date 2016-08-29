set(proto_filenames
    DataConfig.proto
    DataFormat.proto
    ModelConfig.proto
    ParameterConfig.proto
    ParameterService.proto
    TrainerConfig.proto)

set(real_proto_files)

# TODO(yuyang18): Some internal proto will also be depended on.
#                 Find a way to automatically calculate all depends.
foreach(filename ${proto_filenames})
    add_custom_command(OUTPUT ${filename}
        COMMAND ${M4_EXECUTABLE} -Dreal=${ACCURACY} -I '${INTERNAL_PROTO_PATH}'
              ${PROJ_ROOT}/proto/${filename}.m4 > ${filename}
        DEPENDS ${PROJ_ROOT}/proto/${filename}.m4
        COMMENT "Generate ${filename}")
endforeach()

add_custom_target(proto_accuracy ALL
                    DEPENDS ${proto_filenames})

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
        COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} 
                  --cpp_out ${CMAKE_CURRENT_BINARY_DIR}
                  --proto_path ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${filename}
        DEPENDS proto_accuracy
                ${PROJ_ROOT}/proto/${filename}.m4)

    set(CUR_PROTO_GEN_PY
        ${PROJ_ROOT}/paddle/python/paddle/proto/${base_filename}_pb2.py)
    set(PROTO_GEN_PY
        ${CUR_PROTO_GEN_PY}
        ${PROTO_GEN_PY})
    add_custom_command(OUTPUT ${CUR_PROTO_GEN_PY}
        COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --python_out ${PROJ_ROOT}/python/paddle/proto
                  --proto_path ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${filename}
        DEPENDS proto_accuracy
                ${PROJ_ROOT}/proto/${filename}.m4)
endforeach()

include_directories(${CMAKE_CURRENT_BINARY_DIR}/proto)

add_custom_target(gen_proto_cpp ALL DEPENDS ${PROTO_GEN})
add_custom_target(gen_proto_py ALL DEPENDS ${PROTO_GEN_PY})
add_library(paddle_proto STATIC
    ${PROTO_GEN})
add_dependencies(paddle_proto proto_accuracy)
target_include_directories(paddle_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

file(GLOB proto_filenames . *.proto)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
proto_library(paddle_proto SRCS ${proto_filenames})

set(PROTO_GEN)
set(PROTO_GEN_PY)

foreach(filename ${proto_filenames})
    get_filename_component(ABS_FIL ${filename} ABSOLUTE)
    get_filename_component(FIL_WE ${filename} NAME_WE)
    set(CUR_PROTO_GEN_PY
            ${PADDLE_SOURCE_DIR}/paddle/python/paddle/proto/${FIL_WE}_pb2.py)
    set(PROTO_GEN_PY
            ${CUR_PROTO_GEN_PY}
            ${PROTO_GEN_PY})
    add_custom_command(OUTPUT ${CUR_PROTO_GEN_PY}
            COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
            ARGS "--python_out=${PADDLE_SOURCE_DIR}/python/paddle/proto"
            "-I" ${CMAKE_CURRENT_SOURCE_DIR} ${ABS_FIL}
            DEPENDS ${ABS_FIL} protoc)
endforeach()

add_custom_target(gen_proto_py ALL DEPENDS ${PROTO_GEN_PY})

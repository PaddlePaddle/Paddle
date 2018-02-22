if (MOBILE_INFERENCE)
    file(GLOB proto_filenames . ModelConfig.proto ParameterConfig.proto
         TrainerConfig.proto DataConfig.proto)
else()
    file(GLOB proto_filenames . *.proto)
endif()

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


if (WITH_GOLANG)
    add_custom_target(protoc-gen-go)
    add_custom_command(TARGET protoc-gen-go
            COMMAND go 
            ARGS "get" "-u" "github.com/golang/protobuf/protoc-gen-go")

    set(PROTO_GEN_GO)
    file(GLOB proto_filenames . OptimizerConfig.proto)
    foreach(filename ${proto_filenames})
        message(STATUS ${filename})
        get_filename_component(ABS_FIL ${filename} ABSOLUTE)
        get_filename_component(FIL_WE ${filename} NAME_WE)
        set(CUR_PROTO_GEN_GO
                ${PADDLE_SOURCE_DIR}/paddle/go/proto/${FIL_WE}.pb.go)
        set(PROTO_GEN_GO
                ${CUR_PROTO_GEN_GO}
                ${PROTO_GEN_GO})
        add_custom_command(OUTPUT ${CUR_PROTO_GEN_GO}
                COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
                ARGS "--go_out=${PADDLE_SOURCE_DIR}/go/proto"
                "-I" ${CMAKE_CURRENT_SOURCE_DIR} ${ABS_FIL}
                DEPENDS ${ABS_FIL} protoc protoc-gen-go)
    endforeach()
    add_custom_target(gen_proto_go ALL DEPENDS ${PROTO_GEN_GO})
endif()

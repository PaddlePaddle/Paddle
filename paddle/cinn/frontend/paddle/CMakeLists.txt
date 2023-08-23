cinn_proto_library(cinn_framework_proto SRCS framework.proto)

add_subdirectory(cpp)
add_subdirectory(pb)

set(srcs model_parser.cc compatible_pb.cc)

cinn_cc_test(
  test_model_parser
  SRCS
  model_parser_test.cc
  DEPS
  cinncore
  ARGS
  --model_dir=${THIRD_PARTY_PATH}/model/lite_naive_model)

foreach(cpp ${srcs})
  set(cinnapi_src
      "${cinnapi_src};paddle/cinn/frontend/paddle/${cpp}"
      CACHE INTERNAL "")
endforeach()

file(
  GLOB includes
  LIST_DIRECTORIES false
  RELATIVE ${CMAKE_SOURCE_DIR}
  *.h)

foreach(header ${includes})
  set(core_includes
      "${core_includes};${header}"
      CACHE INTERNAL "")
endforeach()

foreach(header ${cinn_framework_proto_HDRS})
  set(core_proto_includes
      "${core_proto_includes};${header}"
      CACHE INTERNAL "")
endforeach()

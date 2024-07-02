file(GLOB_RECURSE SERIALIZE_DESERIALIZE_CPP_SOURCES "*.cc")

include_directories(${PADDLE_BINARY_DIR}/third_party/install/yaml-cpp/include/)
include_directories(${PADDLE_SOURCE_DIR}/third_party/nlohmann_json/include/)
link_directories(${PADDLE_BINARY_DIR}/third_party/install/yaml-cpp/include/)
link_directories(${PADDLE_SOURCE_DIR}/third_party/nlohmann_json/include/)

cc_library(
  pir_save_load
  SRCS ${SERIALIZE_DESERIALIZE_CPP_SOURCES}
  DEPS op_dialect phi json yaml)

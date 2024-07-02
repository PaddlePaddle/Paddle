file(GLOB_RECURSE SERIALIZE_DESERIALIZE_CPP_SOURCES "*.cc")

include_directories(pir_save_load PRIVATE
                    ${PADDLE_SOURCE_DIR}/third_party/nlohmann_json/include/)
include_directories(pir_save_load PRIVATE
                    ${PADDLE_BINARY_DIR}/third_party/install/yaml-cpp/include/)

cc_library(
  pir_save_load
  SRCS ${SERIALIZE_DESERIALIZE_CPP_SOURCES}
  DEPS op_dialect phi json yaml)

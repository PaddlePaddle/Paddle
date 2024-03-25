add_subdirectory(nlohmann_json)


file(
  GLOB_RECURSE
  SERIALIZE_DESERIALIZE_CPP_SOURCES
  "*.cc"
)

#include_directories(${PADDLE_SOURCE_DIR}/third_party/nlohmann_json/include)


#target_include_directories(serialize_deserialize PRIVATE ${PADDLE_SOURCE_DIR}/third_party/nlohmann_json/include/)
cc_library(
  serialize_deserialize
  SRCS ${SERIALIZE_DESERIALIZE_CPP_SOURCES}
  DEPS op_dialect nlohmann_json::nlohmann_json)

file(GLOB_RECURSE SERIALIZE_DESERIALIZE_CPP_SOURCES "*.cc")

cc_library(
  serialize_deserialize
  SRCS ${SERIALIZE_DESERIALIZE_CPP_SOURCES}
  DEPS phi lod_tensor)

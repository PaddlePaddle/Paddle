#add_subdirectory(nlohmann_json)

add_definitions(-DIR_LIBRARY)
set_property(GLOBAL PROPERTY SAVE_TARGETS "")

file(GLOB_RECURSE SERIALIZE_DESERIALIZE_CPP_SOURCES "*.cc")

include_directories(serialize_deserialize PRIVATE
                    ${PADDLE_SOURCE_DIR}/third_party/nlohmann_json/include/)
if(WITH_SHARED_IR)
  add_library(serialize_deserialize SHARED ${SERIALIZE_DESERIALIZE_CPP_SOURCES})
  target_link_libraries(serialize_deserialize op_dialect json)
else()
  cc_library(
    serialize_deserialize
    SRCS ${SERIALIZE_DESERIALIZE_CPP_SOURCES}
    DEPS op_dialect json)

  get_property(save_targets GLOBAL PROPERTY SAVE_TARGETS)
  set(save_targets serialize_deserialize)
  set_property(GLOBAL PROPERTY SAVE_TARGETS "${save_targets}")
endif()

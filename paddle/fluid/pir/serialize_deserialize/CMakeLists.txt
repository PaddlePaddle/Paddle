add_definitions(-DIR_LIBRARY)
set_property(GLOBAL PROPERTY SAVE_TARGETS "")

file(GLOB_RECURSE pir_save_load_CPP_SOURCES "*.cc")

include_directories(pir_save_load PRIVATE
                    ${PADDLE_SOURCE_DIR}/third_party/nlohmann_json/include/)
if(WITH_SHARED_IR)
  add_library(pir_save_load SHARED ${pir_save_load_CPP_SOURCES})
  target_link_libraries(pir_save_load op_dialect json)
else()
  cc_library(
    pir_save_load
    SRCS ${pir_save_load_CPP_SOURCES}
    DEPS op_dialect json)

  get_property(save_targets GLOBAL PROPERTY SAVE_TARGETS)
  set(save_targets pir_save_load)
  set_property(GLOBAL PROPERTY SAVE_TARGETS "${save_targets}")
endif()

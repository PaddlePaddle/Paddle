add_subdirectory(profiler)
add_subdirectory(lib)
if(WIN32)
  file(GLOB YAML_FILE "${CMAKE_CURRENT_SOURCE_DIR}/yaml/*.yaml")
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS ${YAML_FILE})
endif()

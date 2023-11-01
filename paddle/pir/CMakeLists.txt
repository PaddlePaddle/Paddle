add_definitions(-DIR_LIBRARY)
set_property(GLOBAL PROPERTY IR_TARGETS "")

function(ir_static_library TARGET_NAME)
  set(options STATIC static SHARED shared INTERFACE interface)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(ir_library "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  set(OBJ_LIB ir_${TARGET_NAME})
  add_library(${OBJ_LIB} OBJECT ${ir_library_SRCS})
  cc_library(
    static_${TARGET_NAME}
    SRCS $<TARGET_OBJECTS:${OBJ_LIB}>
    DEPS ${ir_library_DEPS})
  set_property(GLOBAL APPEND PROPERTY IR_MODULES $<TARGET_OBJECTS:${OBJ_LIB}>)

  get_property(ir_targets GLOBAL PROPERTY IR_TARGETS)
  set(ir_targets ${ir_targets} static_${TARGET_NAME})
  set_property(GLOBAL PROPERTY IR_TARGETS "${ir_targets}")
endfunction()
file(GLOB_RECURSE PIR_CPP_SOURCES "*.cc")

if(WIN32)
  if(WITH_SHARED_IR)
    set(IR_NAME
        pir.dll
        CACHE INTERNAL "" FORCE)
  else()
    set(IR_NAME
        pir.lib
        CACHE INTERNAL "" FORCE)
  endif()
elseif(APPLE)
  if(WITH_SHARED_IR)
    set(IR_NAME
        libpir.dylib
        CACHE INTERNAL "" FORCE)
  else()
    set(IR_NAME
        libpir.a
        CACHE INTERNAL "" FORCE)
  endif()
else()
  if(WITH_SHARED_IR)
    set(IR_NAME
        libpir.so
        CACHE INTERNAL "" FORCE)
  else()
    set(IR_NAME
        libpir.a
        CACHE INTERNAL "" FORCE)
  endif()
endif()

set(IR_LIB
    "${CMAKE_CURRENT_BINARY_DIR}/${IR_NAME}"
    CACHE FILEPATH "IR Library" FORCE)

if(WITH_SHARED_IR)
  add_library(pir SHARED ${PIR_CPP_SOURCES})
  target_link_libraries(pir ddim)
else()
  ir_static_library(pir SRCS ${PIR_CPP_SOURCES} DEPS ddim)
endif()

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

set(core_srcs CACHE INTERNAL "" FORCE)
set(pass_srcs CACHE INTERNAL "" FORCE)
set(pattern_rewrite_srcs CACHE INTERNAL "" FORCE)
set(transforms_srcs CACHE INTERNAL "" FORCE)
set(contro_flow_srcs CACHE INTERNAL "" FORCE)
set(shape_srcs CACHE INTERNAL "" FORCE)

add_subdirectory(core)
add_subdirectory(pass)
add_subdirectory(pattern_rewrite)
add_subdirectory(transforms)
add_subdirectory(dialect)

set(PIR_SRCS ${core_srcs} ${pass_srcs} ${pattern_rewrite_srcs}
             ${transforms_srcs} ${contro_flow_srcs} ${shape_srcs})
set(PIR_DEPS ddim)
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
  if(NOT ON_INFER)
    ir_static_library(pir SRCS ${PIR_SRCS} DEPS ddim)
  endif()
  add_library(pir SHARED ${PIR_SRCS})
  target_link_libraries(pir ddim)
  set_property(GLOBAL PROPERTY IR_TARGETS pir)
else()
  add_library(pir STATIC ${PIR_SRCS})
  target_link_libraries(pir ddim)
  set_property(GLOBAL PROPERTY IR_TARGETS pir)
endif()

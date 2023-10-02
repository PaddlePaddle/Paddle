add_definitions(-DIR_LIBRARY)
set_property(GLOBAL PROPERTY IR_TARGETS "")
set_property(GLOBAL PROPERTY IR_MODULES "")
function(ir_library TARGET_NAME)
  set(options STATIC static SHARED shared INTERFACE interface)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(ir_library "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  set(OBJ_LIB ir_${TARGET_NAME})
  add_library(${OBJ_LIB} OBJECT ${ir_library_SRCS})

  if(ir_library_SHARED OR ir_library_shared) # build *.so
    cc_library(
      ${TARGET_NAME} SHARED
      SRCS $<TARGET_OBJECTS:${OBJ_LIB}>
      DEPS ${ir_library_DEPS})
  elseif(ir_library_INTERFACE OR ir_library_interface)
    cc_library(
      ${TARGET_NAME} INTERFACE
      SRCS $<TARGET_OBJECTS:${OBJ_LIB}>
      DEPS ${ir_library_DEPS})
  else()
    cc_library(
      ${TARGET_NAME}
      SRCS $<TARGET_OBJECTS:${OBJ_LIB}>
      DEPS ${ir_library_DEPS})
    set_property(GLOBAL APPEND PROPERTY IR_MODULES $<TARGET_OBJECTS:${OBJ_LIB}>)

    get_property(ir_targets GLOBAL PROPERTY IR_TARGETS)
    set(ir_targets ${ir_targets} ${TARGET_NAME})
    set_property(GLOBAL PROPERTY IR_TARGETS "${ir_targets}")

  endif()
endfunction()

add_subdirectory(core)
add_subdirectory(pass)
add_subdirectory(pattern_rewrite)
add_subdirectory(transforms)
add_subdirectory(dialect)

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

get_property(ir_modules GLOBAL PROPERTY IR_MODULES)
if(WITH_SHARED_IR)
  add_library(pir SHARED ${ir_modules})
else()
  add_library(pir STATIC ${ir_modules})
endif()

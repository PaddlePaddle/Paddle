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
    #get_property(ir_modules GLOBAL PROPERTY IR_MODULES)
    #set(ir_modules ${ir_modules} ${TARGET_NAME})
    set_property(GLOBAL APPEND PROPERTY IR_MODULES $<TARGET_OBJECTS:${OBJ_LIB}>)
    #set_property(GLOBAL PROPERTY IR_MODULES "${ir_modules}")
  endif()
endfunction()

if(WITH_SHARED_IR)
  add_definitions(-DIR_DLL_EXPORT)
endif()

add_subdirectory(core)
add_subdirectory(pass)
add_subdirectory(pattern_rewrite)

get_property(ir_modules GLOBAL PROPERTY IR_MODULES)
if(WITH_SHARED_IR)
  # set(target_SRCS ${CMAKE_CURRENT_BINARY_DIR}/ir_dummy.cc)
  # execute_process(COMMAND ${CMAKE_COMMAND} -E touch ${target_SRCS})
  # cc_library(ir SHARED SRCS ${target_SRCS})
  # target_link_libraries(ir -Wl,--whole-archive ${ir_modules} -Wl,--no-whole-archive)
  add_library(ir SHARED ${ir_modules})
else()
  # create_static_lib(ir ${ir_modules})
  add_library(ir STATIC ${ir_modules})
endif()

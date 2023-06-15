set_property(GLOBAL PROPERTY IR_MODULES "")
function(ir_library TARGET_NAME)
  set(options STATIC static SHARED shared INTERFACE interface)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(ir_library "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  if(ir_library_SHARED OR ir_library_shared) # build *.so
    cc_library(
      ${TARGET_NAME} SHARED
      SRCS ${ir_library_SRCS}
      DEPS ${ir_library_DEPS})
  elseif(ir_library_INTERFACE OR ir_library_interface)
    cc_library(
      ${TARGET_NAME} INTERFACE
      SRCS ${ir_library_SRCS}
      DEPS ${ir_library_DEPS})
  else()
    cc_library(
      ${TARGET_NAME}
      SRCS ${ir_library_SRCS}
      DEPS ${ir_library_DEPS})
    get_property(ir_modules GLOBAL PROPERTY IR_MODULES)
    set(ir_modules ${ir_modules} ${TARGET_NAME})
    set_property(GLOBAL PROPERTY IR_MODULES "${ir_modules}")
  endif()
endfunction()

add_subdirectory(core)
add_subdirectory(pass)
add_subdirectory(pattern_rewrite)

get_property(phi_modules GLOBAL PROPERTY PHI_MODULES)

if(WITH_SHARED_IR)
  add_definitions(-DIR_DLL_EXPORT)
  ir_library(ir SHARED DEPS ${ir_modules})
else()
  ir_library(ir DEPS ${ir_modules})
endif()

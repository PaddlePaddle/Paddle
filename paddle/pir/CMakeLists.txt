add_definitions(-DIR_LIBRARY)
set_property(GLOBAL PROPERTY IR_TARGETS "")

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
  target_link_libraries(pir common)
else()
  cc_library(
    pir
    SRCS ${PIR_CPP_SOURCES}
    DEPS common)
  get_property(ir_targets GLOBAL PROPERTY IR_TARGETS)
  set(ir_targets pir)
  set_property(GLOBAL PROPERTY IR_TARGETS "${ir_targets}")
endif()

if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
  set_target_properties(pir PROPERTIES COMPILE_FLAGS "-Wno-maybe-uninitialized")
endif()

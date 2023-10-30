add_definitions(-DIR_LIBRARY)
set_property(GLOBAL PROPERTY IR_TARGETS "")
set_property(GLOBAL PROPERTY IR_MODULES "")

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

# get_property(ir_modules GLOBAL PROPERTY IR_MODULES)
if(WITH_SHARED_IR)
  add_library(pir SHARED ${PIR_SRCS})
  target_link_libraries(pir ddim)
else()
  add_library(pir STATIC ${PIR_SRCS})
  target_link_libraries(pir ddim)
endif()

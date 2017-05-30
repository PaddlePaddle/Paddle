set(GOPATH "${CMAKE_CURRENT_BINARY_DIR}/go")
file(MAKE_DIRECTORY ${GOPATH})
set(PADDLE_IN_GOPATH "${GOPATH}/src/github.com/PaddlePaddle")
file(MAKE_DIRECTORY ${PADDLE_IN_GOPATH})

function(GO_LIBRARY NAME BUILD_TYPE)
  if(BUILD_TYPE STREQUAL "STATIC")
    set(BUILD_MODE -buildmode=c-archive)
    set(LIB_NAME "lib${NAME}.a")
  else()
    set(BUILD_MODE -buildmode=c-shared)
    if(APPLE)
      set(LIB_NAME "lib${NAME}.dylib")
    else()
      set(LIB_NAME "lib${NAME}.so")
    endif()
  endif()

  file(GLOB GO_SOURCE RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.go")
  file(RELATIVE_PATH rel ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

  # find Paddle directory.
  get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
  get_filename_component(PARENT_DIR ${PARENT_DIR} DIRECTORY)
  get_filename_component(PADDLE_DIR ${PARENT_DIR} DIRECTORY)

  # automatically get all dependencies specified in the source code
  # for given target.
  add_custom_target(goGet env GOPATH=${GOPATH} ${CMAKE_Go_COMPILER} get -d ${rel}/...)

  # make a symlink that references Paddle inside $GOPATH, so go get
  # will use the local changes in Paddle rather than checkout Paddle
  # in github.
  add_custom_target(copyPaddle
    COMMAND ln -sf ${PADDLE_DIR} ${PADDLE_IN_GOPATH})
  add_dependencies(goGet copyPaddle)

  add_custom_command(OUTPUT ${OUTPUT_DIR}/.timestamp
    COMMAND env GOPATH=${GOPATH} ${CMAKE_Go_COMPILER} build ${BUILD_MODE}
    -o "${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME}"
    ${CMAKE_GO_FLAGS} ${GO_SOURCE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

  add_custom_target(${NAME} ALL DEPENDS ${OUTPUT_DIR}/.timestamp ${ARGN})
  add_dependencies(${NAME} goGet)

  if(NOT BUILD_TYPE STREQUAL "STATIC")
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME} DESTINATION bin)
  endif()
endfunction(GO_LIBRARY)

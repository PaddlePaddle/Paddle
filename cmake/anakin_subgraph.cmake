set(ANAKIN_ROOT "/usr" CACHE PATH "ANAKIN ROOT")
find_path(ANAKIN_INCLUDE_DIR anakin_config.h
    PATHS ${ANAKIN_ROOT} ${ANAKIN_ROOT}/include
    $ENV{ANAKIN_ROOT} $ENV{ANAKIN_ROOT}/include
    NO_DEFAULT_PATH
)

find_library(ANAKIN_LIBRARY NAMES libanakin_saber_common.so libanakin.so
    PATHS ${ANAKIN_ROOT}
    $ENV{ANAKIN_ROOT} $ENV{ANAKIN_ROOT}/lib
    NO_DEFAULT_PATH
    DOC "Path to ANAKIN library.")

if(ANAKIN_INCLUDE_DIR AND ANAKIN_LIBRARY)
    set(ANAKIN_FOUND ON)
else()
    set(ANAKIN_FOUND OFF)
endif()

if(ANAKIN_FOUND)
    message(STATUS "Current ANAKIN header is ${ANAKIN_INCLUDE_DIR}/anakin_config.h. ")
    include_directories(${ANAKIN_ROOT})
    include_directories(${ANAKIN_ROOT}/include)
    include_directories(${ANAKIN_ROOT}/saber)
    link_directories(${ANAKIN_ROOT})
    add_definitions(-DPADDLE_WITH_ANAKIN)
endif()

if(ANAKIN_FOUND AND WITH_GPU AND WITH_DSO)
    message(STATUS "Compile with anakin subgraph.")
    set(ANAKIN_SUBGRAPH ON)
endif()

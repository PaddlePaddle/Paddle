# use opencv plugin

project(DeJpeg CXX C)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(PROJ_ROOT ${CMAKE_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
set(DEJPEG_LINKER_LIBS "")

# opencv
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND DEJPEG_LINKER_LIBS ${OpenCV_LIBS})
message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
add_definitions(-DUSE_OPENCV)

# boost-python
set(Boost_NO_SYSTEM_PATHS ON)
if (Boost_NO_SYSTEM_PATHS)
  set(BOOST_ROOT $ENV{BOOST_ROOT})
  set(Boost_DIR ${BOOST_ROOT})
  set(Boost_INCLUDE_DIR "${BOOST_ROOT}/include")
  set(Boost_LIBRARIES "${BOOST_ROOT}/lib/")
endif (Boost_NO_SYSTEM_PATHS)
find_package(Boost 1.46 COMPONENTS python)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})
link_directories(${Boost_INCLUDE_DIR})
message(STATUS "Boost found (${Boost_INCLUDE_DIR})")
message(STATUS "Boost found (${Boost_LIBRARIES})")
list(APPEND DEJPEG_LINKER_LIBS ${Boost_LIBRARIES})


file(GLOB DEJPEG_HEADER "${CMAKE_CURRENT_SOURCE_DIR}" "*.h")
file(GLOB DEJPEG_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")

set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -fPIC -Wno-unused-parameter")

add_library(DeJpeg SHARED ${DEJPEG_SOURCES})
target_link_libraries(DeJpeg ${DEJPEG_LINKER_LIBS})
set_target_properties(DeJpeg PROPERTIES PREFIX "")

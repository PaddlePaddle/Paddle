if(NOT CMAKE_Go_COMPILE_OBJECT)
  set(CMAKE_Go_COMPILE_OBJECT "go tool compile -l -N -o <OBJECT> <SOURCE> ")
endif()

if(NOT CMAKE_Go_LINK_EXECUTABLE)
  set(CMAKE_Go_LINK_EXECUTABLE "go tool link -o <TARGET> <OBJECTS>  ")
endif()

# Attention: cmake will append these flags to compile command automatically.
# So if you want to add global option, change this file rather than flags.cmake

# Linux
# DEBUG:  default: "-g"
# RELEASE:  default: "-O3 -DNDEBUG"
# RELWITHDEBINFO: default: "-O2 -g -DNDEBUG"
# MINSIZEREL: default: "-O2 -g -DNDEBUG"

if(NOT WIN32)
  set(CMAKE_C_FLAGS_DEBUG "-g")
  set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
  set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
  set(CMAKE_C_FLAGS_MINSIZEREL "-Os -DNDEBUG")

  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")

  if(WITH_GPU)
    set(CMAKE_CUDA_FLAGS_DEBUG "-g")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_MINSIZEREL "-O1 -DNDEBUG")
  endif()
else()
  set(CMAKE_C_FLAGS_DEBUG "/MDd /Zi /Ob0 /Od /RTC1")
  set(CMAKE_C_FLAGS_RELEASE "/MD /O2 /Ob2 /DNDEBUG")
  set(CMAKE_C_FLAGS_RELWITHDEBINFO "/MD /Zi /O2 /Ob1 /DNDEBUG")
  set(CMAKE_C_FLAGS_MINSIZEREL "/MD /O1 /Ob1 /DNDEBUG")

  set(CMAKE_CXX_FLAGS_DEBUG "/MDd /Zi /Ob0 /Od /RTC1")
  set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /Ob2 /DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MD /Zi /O2 /Ob1 /DNDEBUG")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "/MD /O1 /Ob1 /DNDEBUG")

  if(WITH_GPU)
    set(CMAKE_CUDA_FLAGS_DEBUG "-Xcompiler=\"-MDd -Zi -Ob0 -Od /RTC1\"")
    set(CMAKE_CUDA_FLAGS_RELEASE "-Xcompiler=\"-MD -O2 -Ob2\" -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO
        "-Xcompiler=\"-MD -Zi -O2 -Ob1\" -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_MINSIZEREL "-Xcompiler=\"-MD -O1 -Ob1\" -DNDEBUG")
  endif()

  # It can specify CUDA compile flag manually,
  # its use is to remove /Zi to reduce GPU static library size. But it's dangerous
  # because CUDA will update by nvidia, then error will occur.
  # Now, it's only used in VS2015 + CUDA:[10.0, 10.2]
  set(WIN_PROPS ${CMAKE_SOURCE_DIR}/cmake/paddle_win.props)
endif()

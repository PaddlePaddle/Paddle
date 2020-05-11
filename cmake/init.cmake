# Attention: cmake will append these flags to compile command automatically.
# So if you want to add global option, change this file rather than flags.cmake

# default: "-g"
set(CMAKE_C_FLAGS_DEBUG "-g")
# default: "-O3 -DNDEBUG"
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
# default: "-O2 -g -DNDEBUG"
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
# default: "-Os -DNDEBUG"
set(CMAKE_C_FLAGS_MINSIZEREL "-Os -DNDEBUG")

# default: "-g"
set(CMAKE_CXX_FLAGS_DEBUG "-g")
# default: "-O3 -DNDEBUG"
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
# default: "-O2 -g -DNDEBUG"
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
# default: "-Os -DNDEBUG"
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")

# default: "-g"
set(CMAKE_CUDA_FLAGS_DEBUG "-g")
# default: "-O3 -DNDEBUG"
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
# default: "-O2 -g -DNDEBUG"
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
# default: "-O1 -DNDEBUG"
set(CMAKE_CUDA_FLAGS_MINSIZEREL "-O1 -DNDEBUG")

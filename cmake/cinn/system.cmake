# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Detects the OS and sets appropriate variables.
# CMAKE_SYSTEM_NAME only give us a coarse-grained name of the OS CMake is
# building for, but the host processor name like centos is necessary
# in some scenes to distinguish system for customization.
#
# for instance, protobuf libs path is <install_dir>/lib64
# on CentOS, but <install_dir>/lib on other systems.

if(UNIX AND NOT APPLE)
  # except apple from nix*Os family
  set(LINUX TRUE)
endif()

if(WIN32)
  set(HOST_SYSTEM "win32")
else()
  if(APPLE)
    set(HOST_SYSTEM "macosx")
    exec_program(
      sw_vers ARGS
      -productVersion
      OUTPUT_VARIABLE HOST_SYSTEM_VERSION)
    string(REGEX MATCH "[0-9]+.[0-9]+" MACOS_VERSION "${HOST_SYSTEM_VERSION}")
    if(NOT DEFINED $ENV{MACOSX_DEPLOYMENT_TARGET})
      # Set cache variable - end user may change this during ccmake or cmake-gui configure.
      set(CMAKE_OSX_DEPLOYMENT_TARGET
          ${MACOS_VERSION}
          CACHE
            STRING
            "Minimum OS X version to target for deployment (at runtime); newer APIs weak linked. Set to empty string for default value."
      )
    endif()
    set(CMAKE_EXE_LINKER_FLAGS "-framework CoreFoundation -framework Security")
  else()

    if(EXISTS "/etc/issue")
      file(READ "/etc/issue" LINUX_ISSUE)
      if(LINUX_ISSUE MATCHES "CentOS")
        set(HOST_SYSTEM "centos")
      elseif(LINUX_ISSUE MATCHES "Debian")
        set(HOST_SYSTEM "debian")
      elseif(LINUX_ISSUE MATCHES "Ubuntu")
        set(HOST_SYSTEM "ubuntu")
      elseif(LINUX_ISSUE MATCHES "Red Hat")
        set(HOST_SYSTEM "redhat")
      elseif(LINUX_ISSUE MATCHES "Fedora")
        set(HOST_SYSTEM "fedora")
      endif()

      string(REGEX MATCH "(([0-9]+)\\.)+([0-9]+)" HOST_SYSTEM_VERSION
                   "${LINUX_ISSUE}")
    endif()

    if(EXISTS "/etc/redhat-release")
      file(READ "/etc/redhat-release" LINUX_ISSUE)
      if(LINUX_ISSUE MATCHES "CentOS")
        set(HOST_SYSTEM "centos")
      endif()
    endif()

    if(NOT HOST_SYSTEM)
      set(HOST_SYSTEM ${CMAKE_SYSTEM_NAME})
    endif()

  endif()
endif()

# query number of logical cores
cmake_host_system_information(RESULT CPU_CORES QUERY NUMBER_OF_LOGICAL_CORES)

mark_as_advanced(HOST_SYSTEM CPU_CORES)

message(
  STATUS
    "Found Paddle host system: ${HOST_SYSTEM}, version: ${HOST_SYSTEM_VERSION}")
message(STATUS "Found Paddle host system's CPU: ${CPU_CORES} cores")

# external dependencies log output
set(EXTERNAL_PROJECT_LOG_ARGS
    LOG_DOWNLOAD
    0 # Wrap download in script to log output
    LOG_UPDATE
    1 # Wrap update in script to log output
    LOG_CONFIGURE
    1 # Wrap configure in script to log output
    LOG_BUILD
    0 # Wrap build in script to log output
    LOG_TEST
    1 # Wrap test in script to log output
    LOG_INSTALL
    0 # Wrap install in script to log output
)

# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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

IF(WIN32)
    SET(HOST_SYSTEM "win32")
ELSE(WIN32)
    IF(APPLE)
        SET(HOST_SYSTEM "macosx")
        EXEC_PROGRAM(sw_vers ARGS -productVersion OUTPUT_VARIABLE HOST_SYSTEM_VERSION)
        STRING(REGEX MATCH "[0-9]+.[0-9]+" MACOS_VERSION "${HOST_SYSTEM_VERSION}")
        IF(NOT DEFINED $ENV{MACOSX_DEPLOYMENT_TARGET})
            # Set cache variable - end user may change this during ccmake or cmake-gui configure.
            SET(CMAKE_OSX_DEPLOYMENT_TARGET ${MACOS_VERSION} CACHE STRING
                "Minimum OS X version to target for deployment (at runtime); newer APIs weak linked. Set to empty string for default value.")
        ENDIF()
        set(CMAKE_EXE_LINKER_FLAGS "-framework CoreFoundation -framework Security")
    ELSE(APPLE)

        IF(EXISTS "/etc/issue")
            FILE(READ "/etc/issue" LINUX_ISSUE)
            IF(LINUX_ISSUE MATCHES "CentOS")
                SET(HOST_SYSTEM "centos")
            ELSEIF(LINUX_ISSUE MATCHES "Debian")
                SET(HOST_SYSTEM "debian")
            ELSEIF(LINUX_ISSUE MATCHES "Ubuntu")
                SET(HOST_SYSTEM "ubuntu")
            ELSEIF(LINUX_ISSUE MATCHES "Red Hat")
                SET(HOST_SYSTEM "redhat")
            ELSEIF(LINUX_ISSUE MATCHES "Fedora")
                SET(HOST_SYSTEM "fedora")
            ENDIF()

            STRING(REGEX MATCH "(([0-9]+)\\.)+([0-9]+)" HOST_SYSTEM_VERSION "${LINUX_ISSUE}")
        ENDIF(EXISTS "/etc/issue")

        IF(EXISTS "/etc/redhat-release")
            FILE(READ "/etc/redhat-release" LINUX_ISSUE)
            IF(LINUX_ISSUE MATCHES "CentOS")
                SET(HOST_SYSTEM "centos")
            ENDIF()
        ENDIF(EXISTS "/etc/redhat-release")

        IF(NOT HOST_SYSTEM)
            SET(HOST_SYSTEM ${CMAKE_SYSTEM_NAME})
        ENDIF()

    ENDIF(APPLE)
ENDIF(WIN32)

# query number of logical cores
CMAKE_HOST_SYSTEM_INFORMATION(RESULT CPU_CORES QUERY NUMBER_OF_LOGICAL_CORES)

MARK_AS_ADVANCED(HOST_SYSTEM CPU_CORES)

MESSAGE(STATUS "Found Paddle host system: ${HOST_SYSTEM}, version: ${HOST_SYSTEM_VERSION}")
MESSAGE(STATUS "Found Paddle host system's CPU: ${CPU_CORES} cores")

# external dependencies log output
SET(EXTERNAL_PROJECT_LOG_ARGS
    LOG_DOWNLOAD    0     # Wrap download in script to log output
    LOG_UPDATE      1     # Wrap update in script to log output
    LOG_CONFIGURE   1     # Wrap configure in script to log output
    LOG_BUILD       0     # Wrap build in script to log output
    LOG_TEST        1     # Wrap test in script to log output
    LOG_INSTALL     0     # Wrap install in script to log output
)

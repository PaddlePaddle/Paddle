#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Logic copied from PEP 513


def is_manylinux1_compatible():
    # Only Linux, and only x86-64 / i686
    from distutils.util import get_platform
    if get_platform() not in ["linux-x86_64", "linux-i686"]:
        return False

    # Check for presence of _manylinux module
    try:
        import _manylinux
        return bool(_manylinux.manylinux1_compatible)
    except (ImportError, AttributeError):
        # Fall through to heuristic check below
        pass

    # Check glibc version. CentOS 5 uses glibc 2.5.
    return have_compatible_glibc(2, 5)


def have_compatible_glibc(major, minimum_minor):
    import ctypes

    process_namespace = ctypes.CDLL(None)
    try:
        gnu_get_libc_version = process_namespace.gnu_get_libc_version
    except AttributeError:
        # Symbol doesn't exist -> therefore, we are not linked to
        # glibc.
        return False

    # Call gnu_get_libc_version, which returns a string like "2.5".
    gnu_get_libc_version.restype = ctypes.c_char_p
    version_str = gnu_get_libc_version()
    # py2 / py3 compatibility:
    if not isinstance(version_str, str):
        version_str = version_str.decode("ascii")

    # Parse string and check against requested version.
    version = [int(piece) for piece in version_str.split(".")]
    assert len(version) == 2
    if major != version[0]:
        return False
    if minimum_minor > version[1]:
        return False
    return True


import sys
if is_manylinux1_compatible():
    print("%s is manylinux1 compatible" % (sys.executable, ))
    sys.exit(0)
else:
    print("%s is NOT manylinux1 compatible" % (sys.executable, ))
    sys.exit(1)

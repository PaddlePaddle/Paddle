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

# cf. https://github.com/pypa/manylinux/issues/53

<<<<<<< HEAD
import sys
from urllib.request import urlopen

GOOD_SSL = "https://google.com"
BAD_SSL = "https://self-signed.badssl.com"

print("Testing SSL certificate checking for Python:", sys.version)

print("Connecting to %s should work" % (GOOD_SSL,))
urlopen(GOOD_SSL)
print("...it did, yay.")

print("Connecting to %s should fail" % (BAD_SSL,))
=======
GOOD_SSL = "https://google.com"
BAD_SSL = "https://self-signed.badssl.com"

import sys

print("Testing SSL certificate checking for Python:", sys.version)

if (sys.version_info[:2] < (2, 7) or sys.version_info[:2] < (3, 4)):
    print("This version never checks SSL certs; skipping tests")
    sys.exit(0)

if sys.version_info[0] >= 3:
    from urllib.request import urlopen
    EXC = OSError
else:
    from urllib import urlopen
    EXC = IOError

print("Connecting to %s should work" % (GOOD_SSL, ))
urlopen(GOOD_SSL)
print("...it did, yay.")

print("Connecting to %s should fail" % (BAD_SSL, ))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
try:
    urlopen(BAD_SSL)
    # If we get here then we failed:
    print("...it DIDN'T!!!!!11!!1one!")
    sys.exit(1)
<<<<<<< HEAD
except OSError:
=======
except EXC:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    print("...it did, yay.")

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

import sys
from urllib.request import urlopen

GOOD_SSL = "https://google.com"
BAD_SSL = "https://self-signed.badssl.com"

print("Testing SSL certificate checking for Python:", sys.version)

print("Connecting to %s should work" % (GOOD_SSL,))
urlopen(GOOD_SSL)
print("...it did, yay.")

print("Connecting to %s should fail" % (BAD_SSL,))
try:
    urlopen(BAD_SSL)
    # If we get here then we failed:
    print("...it DIDN'T!!!!!11!!1one!")
    sys.exit(1)
except OSError:
    print("...it did, yay.")

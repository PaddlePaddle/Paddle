import os
import re
import sys

res = sys.argv[1]
out = sys.argv[2]
var = re.sub(r'[ .-]', '_', os.path.basename(res))

open(out, "w").write("const unsigned char " + var + "[] = {" + ",".join([
    "0x%02x" % ord(c) for c in open(res).read()
]) + ",0};\n" + "const unsigned " + var + "_size = sizeof(" + var + ");\n")

# Utility script to print the python tag + the abi tag for a Python
# See PEP 425 for exactly what these are, but an example would be:
#   cp27-cp27mu

from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag

print("{0}{1}-{2}".format(get_abbr_impl(), get_impl_ver(), get_abi_tag()))

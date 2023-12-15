# The home path of ISL
# Required!
set(ISL_HOME "")

set(USE_OPENMP "intel")

# Whether enable SYCL runtime
#
# Possible values:
# - ON: enable SYCL with cmake's auto search.
# - OFF: disable SYCL
# - /path/to/sycl: use specific path to sycl root
set(CINN_WITH_SYCL OFF)
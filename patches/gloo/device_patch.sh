sed -i ':a;N;$!ba;s/#include <string.h>\n\n/#include <string.h>\n#include <array>\n\n/' \
"$1/gloo/transport/tcp/device.cc"

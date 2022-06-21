#include <unistd.h>
#include <netinet/ip.h>
#include <sys/socket.h>
#include <stdio.h>
#include <errno.h>
#include "paddle/fluid/distributed/store/socket.h"

namespace paddle {
namespace distributed {

#ifdef _WIN32
static int _get_sockname_of_win(int sock, char* out, int out_len){
    snprintf(out, outlen, "not support win now");
    return 0;
}
#else
static int _get_sockname(int sock, char* out, int out_len){
    struct sockaddr_in  addr;
    socklen_t       s_len = sizeof(addr);

    if(::getsockname(sock, (void *)&addr, &s_len)){
        ::snprintf(out, out_len, "can't getsocketname of %d, errno:%d", sock, errno)
        return -1;
    }

    char ip[128];
    int port=0;

    // deal with both IPv4 and IPv6:
    if (addr.ss_family == AF_INET) {
        struct sockaddr_in *s = (struct sockaddr_in *)&addr;
        port = ntohs(s->sin_port);
        ::inet_ntop(AF_INET, &s->sin_addr, ip, sizeof(ip));
    } else { // AF_INET6
        struct sockaddr_in6 *s = (struct sockaddr_in6 *)&addr;
        port = ntohs(s->sin6_port);
        ::inet_ntop(AF_INET6, &s->sin6_addr, ip, sizeof(ip));
    }

    ::snprintf(out, out_len, "%s:%d", ip, port);    
    return 0;
}
#endif

int GetSockName(int sock, char* out, int out_len){
#ifdef _WIN32
    return _get_sockname_of_win(sock, out, out_len)
#else
    return _get_sockname(sock, out, out_len)
#endif    
}

std::string GetSocketName(int fd){
    char out[256];
    GetSocketName(fd, out, sizeof(out));
    return std::string(out);
}
};
};

if [ "$1" == "0" ]; then
    export http_proxy=http://10.162.37.16:8128
    export https_proxy=http://10.162.37.16:8128
    echo "reset http_proxy http://10.162.37.16:8128"
else
    export http_proxy=http://10.24.0.156:13128
    export https_proxy=http://10.24.0.156:13128
    echo "reset http_proxy http://10.24.0.156:3128"
    #export https_proxy=http://172.19.57.45:3128
    #export http_proxy=http://172.19.57.45:3128
    #echo "reset http_proxy http://172.19.57.45:3128"
fi
#export no_proxy="localhost,127.0.0.1,localaddress,.localdomain.com,.cdn.bcebos.com,.baidu.com"
export no_proxy="localhost,127.0.0.1,localaddress,.localdomain.com"


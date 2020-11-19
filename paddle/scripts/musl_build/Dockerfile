FROM python:3.7-alpine3.10

WORKDIR /root

RUN apk update

RUN apk add --no-cache \
    g++ gfortran make cmake patchelf git ccache

ARG package

RUN if [ "$package" ]; then \
        pkgs=$(echo "$package" | base64 -d -); \
        echo ">>> decode package:"; \
        echo "$pkgs"; \
        for nm in $pkgs; do \
            echo ">>> intall package: $nm"; \
            apk add --no-cache --force-overwrite "$nm"; \
        done; \
    fi

ARG requirement
ARG pip_index

RUN if [ "$requirement" ]; then \
        echo $requirement| base64 -d - > "requirement.txt"; \
        echo ">>> decode requirement:"; \
        cat "requirement.txt"; \
        echo ">>> install python requirement:"; \
        PIP_ARGS=""; \
        if [ "$pip_index" ]; then \
            PIP_DOMAIN=$(echo "$pip_index" | awk -F/ '{print $3}'); \
            PIP_ARGS="-i $pip_index --trusted-host $PIP_DOMAIN"; \
            echo ">>> pip index: $pip_index"; \
        fi; \
        pip3 install $PIP_ARGS --timeout 300 --no-cache-dir -r "requirement.txt"; \
        rm -f "requirement.txt"; \
    fi

ENTRYPOINT [ "/bin/sh" ]

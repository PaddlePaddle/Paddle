FROM centos:7.4.1708

# anakin install basic tools
RUN yum -y install zip unzip vim wget git make glibc-devel libstdc++-devel epel-release gcc gcc-c++ libstdc++ && rm -rf /var/cache/yum/*

# uncomment if needed.
#RUN yum -y install python-pip && rm -rf /var/cache/yum/*
#RUN pip install --no-cache-dir --upgrade pip && \
#    pip install --no-cache-dir \
#        flask numpy pyyaml scipy pandas


# set env
#ENV LD_LIBRARY_PATH /usr/lib64:$LD_LIBRARY_PATH

# install cmake
RUN wget https://cmake.org/files/v3.2/cmake-3.2.0.tar.gz && tar xzf cmake-3.2.0.tar.gz && \
        cd cmake-3.2.0 && ./bootstrap && \
        make -j4 && make install && cd .. && rm -f cmake-3.2.0.tar.gz

# install pre-requires before installing protobuf.
RUN yum -y install autoconf automake libtool

# install protobuf on Host.
RUN wget --no-check-certificate https://mirror.sobukus.de/files/src/protobuf/protobuf-cpp-3.4.0.tar.gz \
    && tar -xvf protobuf-cpp-3.4.0.tar.gz \
    && cd protobuf-3.4.0 && ./configure \
    && make -j4 && make install && make distclean && cd .. \
    && rm -f protobuf-cpp-3.4.0.tar.gz

# Clone Anakin source code(arm branch).
RUN git clone --branch arm --recursive "https://github.com/PaddlePaddle/Anakin.git" 

#Copy  /usr/local/include/google to 
# Anakin/third-party/arm-android/protobuf/
RUN if [ ! -d "Anakin/third-party/arm-android/protobuf/include" ]; then mkdir -p Anakin/third-party/arm-android/protobuf/include; fi
RUN cp -r /usr/local/include/google Anakin/third-party/arm-android/protobuf/include

# Download Android NDK r14 (Linux version)
# uncompress it and produce android-ndk-r14b directory.
RUN wget https://dl.google.com/android/repository/android-ndk-r14b-linux-x86_64.zip
RUN unzip -q android-ndk-r14b-linux-x86_64.zip && rm -f android-ndk-r14b-linux-x86_64.zip

# set cross-compile environment.
ENV ANDROID_NDK /android-ndk-r14b
ENV ARCH_ABI "arm-linux-androideabi-4.9"
ENV HOSTOSN "linux-x86_64"   
ENV SYSROOT $ANDROID_NDK/platforms/android-9/arch-arm  
ENV PREBUILT $ANDROID_NDK/toolchains/$ARCH_ABI
ENV LDFLAGS "--sysroot=$SYSROOT"
ENV LD  "$ANDROID_NDK/toolchains/$ARCH_ABI/prebuilt/$HOSTOSN/arm-linux-androideabi/bin/ld $LDFLAGS"
ENV LIBS "-llog $ANDROID_NDK/sources/cxx-stl/gnu-libstdc++/4.9/libs/armeabi-v7a/libgnustl_static.a"
ENV CPPFLAGS ""
ENV INCLUDES "-I$ANDROID_NDK/sources/cxx-stl/gnu-libstdc++/4.9/include/ -I$ANDROID_NDK/platforms/android-9/arch-arm/usr/include/ -I$ANDROID_NDK/sources/cxx-stl/gnu-libstdc++/4.9/libs/armeabi-v7a/include/"
ENV CXXFLAGS "-march=armv7-a -mfloat-abi=softfp -DGOOGLE_PROTOBUF_NO_RTTI --sysroot=$SYSROOT"
ENV CCFLAGS "$CXXFLAGS"
ENV CXX "$PREBUILT/prebuilt/$HOSTOSN/bin/arm-linux-androideabi-g++ $CXXFLAGS"
ENV CC "$CXX"
ENV RANLIB "$ANDROID_NDK/toolchains/$ARCH_ABI/prebuilt/$HOSTOSN/bin/arm-linux-androideabi-ranlib"  

# Cross-compile Android armeabi-v7a's protobuf.
# if you want to compile shared libs, please repalce '--disable-shared' with 'disable-static --enbale-shared'.
RUN cd protobuf-3.4.0 \
    && ./autogen.sh \  
    && ./configure --host=arm-linux-androideabi --with-sysroot=$SYSROOT --enable-cross-compile --with-protoc=protoc --disable-shared CXX="$CXX" CC="$CC" LD="$LD" \
    && make

# Copy static library to third-party
RUN if [ ! -d "Anakin/third-party/arm-android/protobuf/lib/" ]; then mkdir -p Anakin/third-party/arm-android/protobuf/lib/; fi
RUN cp protobuf-3.4.0/src/.libs/*.a Anakin/third-party/arm-android/protobuf/lib/

RUN cd Anakin/tools/ \
    && sed -i "s/export ANDROID_NDK=\/home\/public\/android-ndk-r14b/#/g" andrid_build.sh \
    && sed -i "s/-DUSE_OPENCV=YES/-DUSE_OPENCV=NO/g" andrid_build.sh \
    && ./andrid_build.sh
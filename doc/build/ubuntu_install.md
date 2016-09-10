Debian Package installation guide
=================================

## Debian Package installation
Currently , PaddlePaddle only provides ubuntu14.04 debian packages.
There are two versions package, including CPU and GPU. The download address is:

https://github.com/baidu/Paddle/releases/tag/V0.8.0b0


After downloading PaddlePaddle deb packages, you can run:

```bash
dpkg -i paddle-0.8.0b-cpu.deb
apt-get install -f
```
And if you use GPU version deb package, you need to install CUDA toolkit and cuDNN, and set related environment variables(such as LD_LIBRARY_PATH) first. It is normal when `dpkg -i` get errors. `apt-get install -f` will continue install paddle, and install dependences. 

**Note**

PaddlePaddle package only supports x86 CPU with AVX instructions. If not, you have to download and build from source code.

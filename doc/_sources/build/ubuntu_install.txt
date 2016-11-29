Debian Package installation guide
=================================

PaddlePaddle supports :code:`deb` pacakge. The installation of this :code:`deb` package is tested in ubuntu 14.04, but it should be support other debian based linux, too.

There are four versions of debian package, :code:`cpu`, :code:`gpu`, :code:`cpu-noavx`, :code:`gpu-noavx`. And :code:`noavx` version is used to support CPU which does not contain :code:`AVX` instructions. The download url of :code:`deb` package is \: https://github.com/baidu/Paddle/releases/


After downloading PaddlePaddle deb packages, you can use :code:`gdebi` install.

..	code-block:: bash

	gdebi paddle-*.deb

If :code:`gdebi` is not installed, you can use :code:`sudo apt-get install gdebi` to install it.

Or you can use following commands to install PaddlePaddle.

..	code-block:: bash

	dpkg -i paddle-*.deb
	apt-get install -f

And if you use GPU version deb package, you need to install CUDA toolkit and cuDNN, and set related environment variables(such as LD_LIBRARY_PATH) first. It is normal when `dpkg -i` get errors. `apt-get install -f` will continue install paddle, and install dependences. 


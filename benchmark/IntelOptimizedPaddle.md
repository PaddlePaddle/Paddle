# Benchmark

Machine:

- Server
 	- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz, 2 Sockets, 20 Cores per socket
- Laptop
 	- DELL XPS15-9560-R1745: i7-7700HQ 8G 256GSSD
 	- i5 MacBook Pro (Retina, 13-inch, Early 2015)
- Desktop
 	- i7-6700k

System: CentOS release 6.3 (Final), Docker 1.12.1.

PaddlePaddle: paddlepaddle/paddle:latest (TODO: will rerun after 0.11.0)

- MKL-DNN tag v0.10
- MKLML 2018.0.20170720
- OpenBLAS v0.2.20
	 
On each machine, we will test and compare the performance of training on single node using MKL-DNN / MKLML / OpenBLAS respectively.

## Benchmark Model

### Server
Test on batch size 64, 128, 256 on Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

Input image size - 3 * 224 * 224, Time: images/second

- VGG-19

| BatchSize    | 64    | 128  | 256     |
|--------------|-------| -----| --------|
| OpenBLAS     | 7.82  | 8.62  | 10.34  | 
| MKLML        | 11.02 | 12.86 | 15.33  |
| MKL-DNN      | 27.69 | 28.8 | 29.27  |


chart on batch size 128
TBD

 - ResNet
 - GoogLeNet

### Laptop
TBD
### Desktop
TBD

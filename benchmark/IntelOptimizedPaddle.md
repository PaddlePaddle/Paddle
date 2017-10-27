# Benchmark

Machine:

- Server
 	- Intel(R) Xeon(R) Gold 6148M CPU @ 2.40GHz, 2 Sockets, 20 Cores per socket
- Laptop
 	- DELL XPS15-9560-R1745: i7-7700HQ 8G 256GSSD
 	- i5 MacBook Pro (Retina, 13-inch, Early 2015)
- Desktop
 	- i7-6700k

System: CentOS 7.3.1611

PaddlePaddle: commit cfa86a3f70cb5f2517a802f32f2c88d48ab4e0e0

- MKL-DNN tag v0.10
- MKLML 2018.0.20170720
- OpenBLAS v0.2.20
	 
On each machine, we will test and compare the performance of training on single node using MKL-DNN / MKLML / OpenBLAS respectively.

## Benchmark Model

### Server
Test on batch size 64, 128, 256 on Intel(R) Xeon(R) Gold 6148M CPU @ 2.40GHz

Input image size - 3 * 224 * 224, Time: images/second

- VGG-19

| BatchSize    | 64    | 128  | 256     |
|--------------|-------| -----| --------|
| OpenBLAS     | 7.86  | 9.02  | 10.62  | 
| MKLML        | 11.80 | 13.43 | 16.21  |
| MKL-DNN      | 29.07 | 30.40 | 31.06  |


chart on batch size 128
TBD

 - ResNet
 - GoogLeNet

### Laptop
TBD
### Desktop
TBD

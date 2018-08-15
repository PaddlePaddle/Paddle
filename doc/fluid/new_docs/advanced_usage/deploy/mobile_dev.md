# iOS开发文档

## 编译

### 一. 使用 build.sh 编译

```sh 
sh build.sh ios

# 如果只想编译某个特定模型的 op, 则需执行以下命令
sh build.sh ios googlenet

# 在这个文件夹下, 你可以拿到生成的 .a 库
cd ../build/release/ios/build

```

### 二. 使用 xcode 编译

我们提供了 ios 开发更为熟悉的 xcode 编译环境:
在 ios/ 目录下打开 PaddleMobile.xcworkspace 即可编译 PaddleMobile 或者 运行 Demo

### 三. 集成

#### 如使用 c++ 接口
将 

```
libpaddle-mobile.a 
io.h  
program.h 
types.h 
lod_tensor.h 
tensor.h
```
拖入工程, io.h 为接口文件, 可在 [github](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/src/io/io.h)上查看接口注释

#### 如使用 oc 接口
将在xcode 编译生成的
```
libPaddleMobile.a 
PaddleMobile.h
```
拖入工程, 接口如下:

```
/*
	创建单例对象
*/
+ (instancetype)sharedInstance;

/*
	load 模型, 开辟内存
*/
- (BOOL)load:(NSString *)modelPath andWeightsPath:(NSString *)weighsPath;

/*
	进行预测, means 和 scale 为训练模型时的预处理参数, 如训练时没有做这些预处理则直接使用 predict
*/
- (NSArray *)predict:(CGImageRef)image means:(NSArray<NSNumber *> *)means scale:(float)scale;

/*
	进行预测
*/
- (NSArray *)predict:(CGImageRef)image;

/*
	清理内存
*/
- (void)clear;

```

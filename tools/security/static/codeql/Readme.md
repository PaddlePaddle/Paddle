# Paddle CodeQL examples

使用CodeQL对Paddle进行静态代码审计。包括python部分和c++部分，用于审计Paddle中的Python代码与C++代码。

该项目包括若干C++和Python的CodeQL样例，样例具体审计功能见相应qhelp文件。开发者可按照此方法尝试编写更多query审计代码。

## Usage

1. 安装CodeQL CLI，步骤见[官方](https://codeql.github.com/docs/codeql-cli/getting-started-with-the-codeql-cli/)。
2. 创建Paddle database。C++的database需要项目可编译，创建好环境后make。

- C++ database:

```shell
codeql database create --language=cpp paddle-database --command="make -j$(nproc)"
```

- Python database:

```shell
codeql database create --language=python --source-root Paddle/python paddle-python-database
```

3. 运行示例qlpack。

- C++:

```shell
codeql database run-queries paddle-database static/codeql/cpp
```

- Python:

```shell
codeql database run-queries paddle-python-database static/codeql/python
```

4. 解析bqrs文件

```shell
codeql bqrs decode --output <CSV_FILE> --format=csv <RESULT>.bqrs
```

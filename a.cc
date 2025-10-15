// 创建单个整数
int* ptr = new int(42);

// 创建数组
int* arr = new int[10];

// 创建对象
class MyClass {
public:
    MyClass() { std::cout << "构造函数\n"; }
    ~MyClass() { std::cout << "析构函数\n"; }
};

MyClass* obj = new MyClass();

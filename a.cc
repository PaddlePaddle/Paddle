int* ptr = new int(42);
int* arr = new int[10];

class MyClass {
public:
    MyClass() { std::cout << "constructor\n"; } // 测试
};

MyClass* obj = new MyClass();

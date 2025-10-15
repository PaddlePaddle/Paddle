int* ptr = new int(42);

int* arr = new int[10];

class MyClass {
public:
    MyClass() { std::cout << "constructor\n"; }
    ~MyClass() { std::cout << "destructor\n"; }
};

MyClass* obj = new MyClass();

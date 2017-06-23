# Design doc of `Variable`

## Overview
`Variable` is a general container that hosts a typed pointer. The main function of `Variable` is to hold a single value of any type.  We summary some necessary features which should be satisfied by `Variable`:
	
* Accept external data pointer of any type
* Get typed data
* Destroy data in destructor

## Prototype

Two aspects should be considered when designing `Variable`, the first is interactive interfaces with external data, the second is  type maintaining.  Here, we use `boost::any` to hold data and its type.

```cpp
Class Variable {
public:
  // Default and only constructor.
  // Variable should be constructed without any parameter,
  // the data pointer must be passed by calling Reset.
  Variable();
  ~Variable();
  
  // Pass data pointer to Variable.
  // When passing nullptr, just deconstruct hosted data.
  // Otherwise deconstruct original data and host the passed data 
  // and init the destructor pointer. 
  template <calss T>
  void Reset(T* ptr = nullptr);
  
  // Get read-only typed object reference.
  // The type checking will be done implicitly.
  template <class T>
  const T& Get() const;
      
  // Get mutable pointer to the hosted data.
  // If the value is nullptr, create a new object, then return it. And
  // the type T should have a default constructor.
  // The `is_new_obj` can use be used to determine whether the returned pointer
  // is a new object.
  template <class T>
  T* GetMutable(bool* is_new_obj=nullptr);

privated:
  // A typed data pointer, set the type when calling Reset.
  boost::any value_;
  
  // A destroy call.
  template <class T>
  static void Destroy(void* pointer);
  
  // A pointer to save the destructor.
  typedef void (*DestroyCall)(void *);
  DestroyCall destroy_ = nullptr;
};
```

## Usage

Here, we give several usages. 

1. New a `Variable` and set a tensor.
  
```cpp
Tensor t = new Tensor();
Variable var = new Variable();
var.Reset(t);
```

2. New a `Variable` and get a tensor by the `GetMutable` interface.

```cpp
Variable var = new Variable();
auto t = var.GetMutable<Tensor>();
```

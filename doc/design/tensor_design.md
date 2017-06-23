# Tensor Design


## Allocation

`Allocation` is a [RAII](http://en.cppreference.com/w/cpp/language/raii) class template, which is used to handle a piece of memory. 

```cpp
// Template parameter 'Device' can be 'CpuDevice' or 'GpuDevice'
template <typename Device>
class Allocation {
public:
    Allocation();
    Allocation(size_t size, Device device);

    // Creates a non-owned allocation
    Allocation(void* ptr, size_t size, Device device);

    ~Allocation();
    //No copying!
    Allocation(const Allocation&) = delete;
    //No assigning!
    Allocation& operator=(const Allocation&) = delete;

    void* ptr() const;
    void* end() const;
    Device device() const;
    size_t size() const;

private:
    bool owned_;
    void* ptr_;
    size_t size_;
    Device device_;
};
```

`ptr_` points to the head of the memory piece and `size_` shows its length. `owned_` marks whether the memory piece is allocated by `allocation` itself, if so the memory will be freed when `allocation` is destructed.

`Device` is something like Majel's `Place`. However, `Place` in Majel is an alias of `boost::variant`, while `Device` here is a certain class (can be specialized to `CpuDevice` or `GpuDevice`). `CpuDevice` and `GpuDevice` are exactly Majel's `CpuPlace` and `GpuPlace`, we rename them to fit the overall naming style.

```cpp
struct CpuDevice{
    inline bool operator==(const CpuDevice&) const {
        return true;
    }
    inline bool operator!=(const CpuDevice&) const {
        return false;
    }
};

struct GpuDevice {
    GpuDevice(int d) : device_id(d) { }
    
    inline bool operator==(const GpuDevice &o) const {
        return device_id == o.device_id;
    }
    inline bool operator!=(const GpuDevice &o) const {
        return !(*this == o);
    }
    GpuDevice() : GpuDevice(0) { }
    int device_id;
};
```

## Tensor

`Tensor` is the combination of Majel's `Buffer` and `Array`.

```cpp
template<typename Device, typename T, int rank>
class Tensor {
public:
    // tensor with zero size and no memory
    Tensor();
    // allocates new densely packed tensor
    Tensor(const Dim<rank> size, Device device);

    // make a new tensor by another existing tensor
    // new tensor and source tensor have the same numel but deferent rank
    template<int src_rank>
    Tensor(const Dim<rank>& size, Tensor<Device, T, src_rank>& src);
    
    // '=' are not allowed, because users may be confused about
    //     whether it's deep copy or shallow copy.
    Tensor& operator=(const Tensor& src) = delete;

    // return raw pointer to the data.
    T* raw_ptr() const;

    // return tensor size
    Dim<rank> size() const;

    // return the number of tensor elements
    int numel() const;

    // return tensor stride
    Dim<rank> stride() const;

    // return raw pointer to the 'idx'th element
    T* index(const Dim<rank>& idx) const;

    // resize tensor, data may be erased
    void resize(const Dim<rank>& size);

    // reshape tensor, data will be retained
    void reshape(const Dim<rank>& size);

private:
    std::shared_ptr<Allocation<Device> > allocation_;
    Dim<rank> size_;
    Dim<rank> stride_;
    T* ptr_;
};

```

The member variable `allocation_` points to the `Allocation` object where data are stored. However, one `Allocation` object can be shared by several tensors, so we need another raw pointer `ptr_` to indicate where is the head of **this** tensor's data. 

`size_` and `stride_` are `Dim` object. Inspired from Majel, `Dim` is a struct template for indicating tensor size and element index:

```cpp
template<int rank>
struct Dim {
	// constructor
	template<typename... Args>
	Dim(int _head, Args... _tail) : head(_head), tail(_tail...) { }
	
	int head;
	Dim<rank-1> tail;
}

template<>
struct Dim<1> {
	int head;
}

// helper function to make a Dim
template<typename... Args>
Dim<sizeof...(Args)> make_dim(Args... idxes) {
    return Dim<sizeof...(Args)>(idxes...);
}
```

In addition to `Tensor`'s member function, a few related global functions are going to be offered, such as `Copy()` and `ShareData()`:

```cpp
// Copy() is used for tensor deep copy
template <typename Device, typename T, int rank>
Tensor<Device, T, rank> Copy(const Tensor<Device, T, rank>& src);

// ShareDate() is used for tensor shallow copy
Tensor<Device, T, rank> ShareData(const Tensor<Device, T, rank>& src);
```

## Tensor Usage

We can use tensors as follow:

```cpp
// make a totally new tensor on CPU with raw constructor
Tensor<CpuDevice, double, 2> t_a(make_dim(2, 3), CpuDevice());
// make a totally new tensor on GPU 1
Tensor<GpuDevice, float, 3> t_b(make_dim(2, 3, 4), GpuDevice(1));

// resize t_a
// resize can not change tensor's rank
t_a.resize(make_dim(1, 4));

// make a new allocation shared tensor with the same numel and differnet rank
// t_b's numel is 2*3*4=24, t_c's desired numel is 3*8=24, they are same so the construction is allowed.
Tensor<GpuDevice, float, 2> t_c(make_dim(3, 8), t_b);

// get tensor's data pointer
void* data_ptr = t_a.raw_ptr();
```

## TODO

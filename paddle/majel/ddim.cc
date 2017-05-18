#include <majel/ddim.h>

namespace majel {

///@cond HIDDEN

template <int i>
Dim<i> make_dim(const int* d) {
  return Dim<i>(*d, make_dim<i - 1>(d + 1));
}

template <>
Dim<1> make_dim<1>(const int* d) {
  return Dim<1>(*d);
}

void make_ddim(DDim& ddim, const int* dims, int n) {
  switch (n) {
    case 1:
      ddim = make_dim<1>(dims);
      break;
    case 2:
      ddim = make_dim<2>(dims);
      break;
    case 3:
      ddim = make_dim<3>(dims);
      break;
    case 4:
      ddim = make_dim<4>(dims);
      break;
    case 5:
      ddim = make_dim<5>(dims);
      break;
    case 6:
      ddim = make_dim<6>(dims);
      break;
    case 7:
      ddim = make_dim<7>(dims);
      break;
    case 8:
      ddim = make_dim<8>(dims);
      break;
    case 9:
      ddim = make_dim<9>(dims);
      break;
    default:
      throw std::invalid_argument(
          "Dynamic dimensions must have between [1, 9] dimensions.");
  }
}

///@endcond

DDim make_ddim(std::initializer_list<int> dims) {
  DDim result(make_dim(0));
  make_ddim(result, dims.begin(), dims.size());
  return result;
}

DDim make_ddim(const std::vector<int>& dims) {
  DDim result(make_dim(0));
  make_ddim(result, &dims[0], dims.size());
  return result;
}

///@cond HIDDEN
// XXX For some reason, putting this in an anonymous namespace causes errors
class DynamicMutableIndexer : public boost::static_visitor<int&> {
public:
  DynamicMutableIndexer(int idx) : idx_(idx) {}

  template <int D>
  int& operator()(Dim<D>& dim) const {
    return dim[idx_];
  }

private:
  int idx_;
};

class DynamicConstIndexer : public boost::static_visitor<int> {
public:
  DynamicConstIndexer(int idx) : idx_(idx) {}

  template <int D>
  int operator()(const Dim<D>& dim) const {
    return dim[idx_];
  }

private:
  int idx_;
};

///@endcond

int& DDim::operator[](int idx) {
  return boost::apply_visitor(DynamicMutableIndexer(idx), var);
}

int DDim::operator[](int idx) const {
  return boost::apply_visitor(DynamicConstIndexer(idx), var);
}

bool DDim::operator==(DDim d) const {
  if (var.which() != d.getVar().which()) {
    return false;
  } else {
    std::vector<int> v1 = vectorize(*this);
    std::vector<int> v2 = vectorize(d);

    for (unsigned int i = 0; i < v1.size(); i++) {
      if (v1[i] != v2[i]) {
        return false;
      }
    }

    return true;
  }
}

bool DDim::operator!=(DDim d) const { return !(*this == d); }

DDim DDim::operator+(DDim d) const {
  std::vector<int> v1 = vectorize(*this);
  std::vector<int> v2 = vectorize(d);

  std::vector<int> v3;

  assert(v1.size() == v2.size());

  for (unsigned int i = 0; i < v1.size(); i++) {
    v3.push_back(v1[i] + v2[i]);
  }

  return make_ddim(v3);
}

DDim DDim::operator*(DDim d) const {
  std::vector<int> v1 = vectorize(*this);
  std::vector<int> v2 = vectorize(d);

  std::vector<int> v3;

  assert(v1.size() == v2.size());

  for (unsigned int i = 0; i < v1.size(); i++) {
    v3.push_back(v1[i] * v2[i]);
  }

  return make_ddim(v3);
}

int get(const DDim& ddim, int idx) { return ddim[idx]; }

void set(DDim& ddim, int idx, int value) { ddim[idx] = value; }

///@cond HIDDEN
struct VectorizeVisitor : public boost::static_visitor<> {
  std::vector<int>& vector;

  VectorizeVisitor(std::vector<int>& v) : vector(v) {}

  template <typename T>
  void operator()(const T& t) {
    vector.push_back(t.head);
    this->operator()(t.tail);
  }

  void operator()(const Dim<1>& t) { vector.push_back(t.head); }
};
///@endcond

std::vector<int> vectorize(const DDim& ddim) {
  std::vector<int> result;
  VectorizeVisitor visitor(result);
  boost::apply_visitor(visitor, ddim);
  return result;
}

ssize_t product(const DDim& ddim) {
  ssize_t result = 1;
  std::vector<int> v = vectorize(ddim);
  for (auto i : v) {
    result *= i;
  }
  return result;
}

///\cond HIDDEN

struct ArityVisitor : boost::static_visitor<int> {
  template <int D>
  int operator()(Dim<D>) const {
    return D;
  }
};

///\endcond

int arity(const DDim& d) { return boost::apply_visitor(ArityVisitor(), d); }

///\cond HIDDEN

struct DDimPrinter : boost::static_visitor<void> {
  std::ostream& os;
  DDimPrinter(std::ostream& os_) : os(os_) {}

  template <typename T>
  void operator()(const T& t) {
    os << t;
  }
};

///\endcond

std::ostream& operator<<(std::ostream& os, const majel::DDim& ddim) {
  DDimPrinter printer(os);
  boost::apply_visitor(printer, ddim);
  return os;
}

}  // namespace majel

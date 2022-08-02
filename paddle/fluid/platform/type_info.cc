/*written by OccupyMars2025*/

#include <paddle/fluid/platform/type_info.h>
#include <numeric>

int Tinfo::Bits(const at::ScalarType& type){
	int bits = elementSize(self->type) * 8;
  	return THPUtils_packInt64(bits);
}
float Tinfo::Eps(const at::ScalarType& type){
    return std::numeric_limits<at::scalar_value_type<scalar_t>::type>::epsilon());
}
float Tinfo::Min(const at::ScalarType& type){
    return std::numeric_limits<at::scalar_value_type<scalar_t>::type>::lowest());
}
float Tinfo::Max(const at::ScalarType& type){
    return std::numeric_limits<at::scalar_value_type<scalar_t>::type>::max());
}
float Tinfo::Tiny(const at::ScalarType& type){
    return std::numeric_limits<at::scalar_value_type<scalar_t>::type>::min());
}
float Tinfo::Resolution(const at::ScalarType& type){
    return std::numeric_limits<at::scalar_value_type<scalar_t>::type>::resolution());
   
}



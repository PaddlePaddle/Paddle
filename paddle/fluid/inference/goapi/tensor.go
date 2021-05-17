// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package paddle

// #include "pd_tensor.h"
// #include "pd_utils.h"
// #include "pd_types.h"
// #include "pd_common.h"
// #include "stdlib.h"
import "C"
import (
	"fmt"
	"reflect"
	"unsafe"
)

type DataType C.PD_DataType

const (
	Unk     DataType = C.PD_DATA_UNK
	Float32 DataType = C.PD_DATA_FLOAT32
	Int32   DataType = C.PD_DATA_INT32
	Int64   DataType = C.PD_DATA_INT64
	Uint8   DataType = C.PD_DATA_UINT8
	Int8    DataType = C.PD_DATA_INT8
)

type PlaceType C.PD_PlaceType

const (
	UnkPlace PlaceType = C.PD_PLACE_UNK
	CpuPlace PlaceType = C.PD_PLACE_CPU
	GpuPlace PlaceType = C.PD_PLACE_GPU
	XpuPlace PlaceType = C.PD_PLACE_XPU
)

type Tensor struct {
	c *C.PD_Tensor
}

///
/// \brief Reset the shape of the tensor.
/// Generally it's only used for the input tensor.
///
/// \param[in] shape The shape to set.
///
func (t *Tensor) Reshape(shape []int32) {
	C.PD_TensorReshape(t.c, C.size_t(len(shape)), (*C.int32_t)(unsafe.Pointer(&shape[0])))
}

///
/// \brief Get the tensor shape
///
/// \return The tensor shape.
///
func (t *Tensor) Shape() []int32 {
	cData := C.PD_TensorGetShape(t.c)
	length := int(cData.size)
	return cvtToGoSliceInt32(length, cData.data)
}

///
/// \brief Set the tensor lod information
/// \param[in] pd_tensor tensor.
/// \param[in] lod lod information.
///
func (t *Tensor) SetLod(lod [][]uint) {
	cLod := (*C.struct_PD_TwoDimArraySize)(C.malloc(C.size_t(C.sizeof_struct_PD_TwoDimArraySize)))
	length := len(lod)
	cLod.size = C.size_t(uint(length))
	var lodList = make([]*C.struct_PD_OneDimArraySize, length+1)

	for i, v := range lod {
		oneDimArray := (*C.struct_PD_OneDimArraySize)(C.malloc(C.size_t(C.sizeof_struct_PD_OneDimArraySize)))
		defer C.free(unsafe.Pointer(oneDimArray))
		tmpLength := len(v)
		oneDimArray.size = C.size_t(uint(tmpLength))

		tmpC := (*C.size_t)(C.malloc(C.size_t(C.sizeof_size_t * tmpLength)))
		defer C.free(unsafe.Pointer(tmpC))
		tmpSlice := (*[1 << 27]C.size_t)(unsafe.Pointer(tmpC))[:tmpLength:tmpLength]
		for j, w := range v {
			tmpSlice[j] = C.size_t(w)
		}
		oneDimArray.data = tmpC

		lodList[i] = oneDimArray
	}
	cLod.data = (**C.struct_PD_OneDimArraySize)(unsafe.Pointer(&lodList[0]))
	C.PD_TensorSetLod(t.c, cLod)
	C.free(unsafe.Pointer(cLod))
	// C.PD_TwoDimArraySizeDestroy(cLod)
}

///
/// \brief Get the tensor lod information
///
/// \return the lod information.
///
func (t *Tensor) Lod() [][]uint {
	cLod := C.PD_TensorGetLod(t.c)
	length := int(cLod.size)
	res := make([][]uint, length)
	cLodSlice := (*[1 << 27]*C.struct_PD_OneDimArraySize)(unsafe.Pointer(cLod.data))[:length:length]

	for i := 0; i < length; i++ {
		size := uint(cLodSlice[i].size)
		lod := make([]uint, size)

		tmpSlice := (*[1 << 27]C.size_t)(unsafe.Pointer(cLodSlice[i].data))[:size:size]
		for j, v := range tmpSlice {
			lod[j] = uint(v)
		}

		res[i] = lod
	}

	C.PD_TwoDimArraySizeDestroy(cLod)
	return res
}

///
/// \brief Get the tensor data type
/// \param[in] pd_tensor tensor.
/// \return the tensor data type.
///
func (t *Tensor) Type() DataType {
	cDtype := C.PD_TensorGetDataType(t.c)
	return DataType(cDtype)
}

///
/// \brief Get the tensor name
///
/// \return the tensor name.
///
func (t *Tensor) Name() string {
	return C.GoString(C.PD_TensorGetName(t.c))
}

///
/// \brief Copy the host memory to tensor data.
/// It's usually used to set the input tensor data.
///
/// \param[in] value
///
func (t *Tensor) CopyFromCpu(value interface{}) {
	val := reflect.ValueOf(value)
	dtype, _ := dataTypeOf(val)

	switch dtype {
	case Float32:
		data := val.Interface().([]float32)
		C.PD_TensorCopyFromCpuFloat(t.c, (*C.float)(unsafe.Pointer(&data[0])))
	case Int32:
		data := val.Interface().([]int32)
		C.PD_TensorCopyFromCpuInt32(t.c, (*C.int32_t)(unsafe.Pointer(&data[0])))
	case Int64:
		data := val.Interface().([]int64)
		C.PD_TensorCopyFromCpuInt64(t.c, (*C.int64_t)(unsafe.Pointer(&data[0])))
	case Uint8:
		data := val.Interface().([]uint8)
		C.PD_TensorCopyFromCpuUint8(t.c, (*C.uint8_t)(unsafe.Pointer(&data[0])))
	case Int8:
		data := val.Interface().([]int8)
		C.PD_TensorCopyFromCpuInt8(t.c, (*C.int8_t)(unsafe.Pointer(&data[0])))
	}
}

///
/// \brief Copy the tensor data to the host memory.
/// It's usually used to get the output tensor data.
///
/// \param[value] data The tensor will copy the data to the address.
///
func (t *Tensor) CopyToCpu(value interface{}) {
	val := reflect.ValueOf(value)
	dtype, _ := dataTypeOf(val)

	switch dtype {
	case Float32:
		data := val.Interface().([]float32)
		C.PD_TensorCopyToCpuFloat(t.c, (*C.float)(unsafe.Pointer(&data[0])))
	case Int32:
		data := val.Interface().([]int32)
		C.PD_TensorCopyToCpuInt32(t.c, (*C.int32_t)(unsafe.Pointer(&data[0])))
	case Int64:
		data := val.Interface().([]int64)
		C.PD_TensorCopyToCpuInt64(t.c, (*C.int64_t)(unsafe.Pointer(&data[0])))
	case Uint8:
		data := val.Interface().([]uint8)
		C.PD_TensorCopyToCpuUint8(t.c, (*C.uint8_t)(unsafe.Pointer(&data[0])))
	case Int8:
		data := val.Interface().([]int8)
		C.PD_TensorCopyToCpuInt8(t.c, (*C.int8_t)(unsafe.Pointer(&data[0])))
	}
}

var types = []struct {
	typ      reflect.Type
	dataType C.PD_DataType
}{
	{reflect.TypeOf(float32(0)), C.PD_DATA_FLOAT32},
	{reflect.TypeOf(int32(0)), C.PD_DATA_INT32},
	{reflect.TypeOf(int64(0)), C.PD_DATA_INT64},
	{reflect.TypeOf(uint8(0)), C.PD_DATA_UINT8},
	{reflect.TypeOf(int8(0)), C.PD_DATA_INT8},
}

func dataTypeOf(val reflect.Value) (dt DataType, err error) {
	typ := val.Type()
	for typ.Kind() == reflect.Array || typ.Kind() == reflect.Slice {
		if val.Len() > 0 {
			val = val.Index(0)
		}
		typ = typ.Elem()
	}
	for _, t := range types {
		if typ.Kind() == t.typ.Kind() {
			return DataType(t.dataType), nil
		}
	}
	return dt, fmt.Errorf("unsupported type %v", typ)
}

// -------------------------

func (t *Tensor) OneDimData() interface{} {
	var size int32 = 0
	var place PlaceType = CpuPlace

	var res interface{}

	dt := t.Type()

	switch dt {
	case Float32:
		cData := unsafe.Pointer(C.PD_TensorDataFloat(t.c, (*C.PD_PlaceType)(unsafe.Pointer(&place)), (*C.int32_t)(unsafe.Pointer(&size))))
		res = (*[1 << 27]float32)(cData)[:int(size):int(size)]
	case Int32:
		cData := unsafe.Pointer(C.PD_TensorDataInt32(t.c, (*C.PD_PlaceType)(unsafe.Pointer(&place)), (*C.int32_t)(unsafe.Pointer(&size))))
		res = (*[1 << 27]int32)(cData)[:int(size):int(size)]
	case Int64:
		cData := unsafe.Pointer(C.PD_TensorDataInt64(t.c, (*C.PD_PlaceType)(unsafe.Pointer(&place)), (*C.int32_t)(unsafe.Pointer(&size))))
		res = (*[1 << 27]int64)(cData)[:int(size):int(size)]
	case Uint8:
		cData := unsafe.Pointer(C.PD_TensorDataUint8(t.c, (*C.PD_PlaceType)(unsafe.Pointer(&place)), (*C.int32_t)(unsafe.Pointer(&size))))
		res = (*[1 << 27]uint8)(cData)[:int(size):int(size)]
	case Int8:
		cData := unsafe.Pointer(C.PD_TensorDataInt8(t.c, (*C.PD_PlaceType)(unsafe.Pointer(&place)), (*C.int32_t)(unsafe.Pointer(&size))))
		res = (*[1 << 27]int8)(cData)[:int(size):int(size)]
	}

	return res
}

///
/// \brief
///
/// \return
///
func (t *Tensor) Data() interface{} {
	dt := t.Type()
	shape := t.Shape()
	var size int32 = 0
	var place PlaceType = CpuPlace

	var (
		slice reflect.Value
		typ   reflect.Type
	)

	var cData unsafe.Pointer
	switch dt {
	case Float32:
		cData = unsafe.Pointer(C.PD_TensorDataFloat(t.c, (*C.PD_PlaceType)(unsafe.Pointer(&place)), (*C.int32_t)(unsafe.Pointer(&size))))
	case Int32:
		cData = unsafe.Pointer(C.PD_TensorDataInt32(t.c, (*C.PD_PlaceType)(unsafe.Pointer(&place)), (*C.int32_t)(unsafe.Pointer(&size))))
	case Int64:
		cData = unsafe.Pointer(C.PD_TensorDataInt64(t.c, (*C.PD_PlaceType)(unsafe.Pointer(&place)), (*C.int32_t)(unsafe.Pointer(&size))))
	case Uint8:
		cData = unsafe.Pointer(C.PD_TensorDataUint8(t.c, (*C.PD_PlaceType)(unsafe.Pointer(&place)), (*C.int32_t)(unsafe.Pointer(&size))))
	case Int8:
		cData = unsafe.Pointer(C.PD_TensorDataInt8(t.c, (*C.PD_PlaceType)(unsafe.Pointer(&place)), (*C.int32_t)(unsafe.Pointer(&size))))
	}

	typ = typeForDataType(dt)
	l := int(size) * int(typ.Size())
	typ = reflect.SliceOf(typ)
	slice = reflect.MakeSlice(typ, int(size), int(size))

	goData := (*[1 << 27]byte)(cData)[:l:l]
	baseBytes := *(*[]byte)(unsafe.Pointer(&sliceHeader{
		Data: unsafe.Pointer(slice.Pointer()),
		Len:  l,
		Cap:  l,
	}))
	copy(baseBytes, goData)

	// Now we have the data in place in the base slice we can add the
	// dimensions. We want to walk backwards through the shape. If the shape is
	// length 1 or 0 then we're already done.
	if len(shape) == 0 {
		return slice.Index(0)
	}
	if len(shape) == 1 {
		return slice
	}

	// We have a special case if the tensor has no data. Our backing slice is
	// empty, but we still want to create slices following the shape. In this
	// case only the final part of the shape will be 0 and we want to recalculate
	// n at this point ignoring that 0.
	// For example if our shape is 3 * 2 * 0 then n will be zero, but we still
	// want 6 zero length slices to group as follows.
	// {{} {}} {{} {}} {{} {}}
	if size == 0 {
		size = int32(numElements(shape[:len(shape)-1]))
	}
	for i := len(shape) - 2; i >= 0; i-- {
		underlyingSize := typ.Elem().Size()
		typ = reflect.SliceOf(typ)
		subsliceLen := int(shape[i+1])
		if subsliceLen != 0 {
			size = size / int32(subsliceLen)
		}
		// Just using reflection it is difficult to avoid unnecessary
		// allocations while setting up the sub-slices as the Slice function on
		// a slice Value allocates. So we end up doing pointer arithmetic!
		// Pointer() on a slice gives us access to the data backing the slice.
		// We insert slice headers directly into this data.
		data := unsafe.Pointer(slice.Pointer())
		nextSlice := reflect.MakeSlice(typ, int(size), int(size))

		for j := 0; j < int(size); j++ {
			// This is equivalent to nSlice[j] = slice[j*subsliceLen: (j+1)*subsliceLen]
			setSliceInSlice(nextSlice, j, sliceHeader{
				Data: unsafe.Pointer(uintptr(data) + (uintptr(j*subsliceLen) * underlyingSize)),
				Len:  subsliceLen,
				Cap:  subsliceLen,
			})
		}

		slice = nextSlice
	}

	return slice.Interface()
}

// Refer to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/tensor.go for input processing.
// setSliceInSlice sets slice[index] = content.
func setSliceInSlice(slice reflect.Value, index int, content sliceHeader) {
	const sliceSize = unsafe.Sizeof(sliceHeader{})
	// We must cast slice.Pointer to uninptr & back again to avoid GC issues.
	// See https://github.com/google/go-cmp/issues/167#issuecomment-546093202
	*(*sliceHeader)(unsafe.Pointer(uintptr(unsafe.Pointer(slice.Pointer())) + (uintptr(index) * sliceSize))) = content
}

func typeForDataType(dt DataType) reflect.Type {
	for _, t := range types {
		if dt == DataType(t.dataType) {
			return t.typ
		}
	}
	panic(fmt.Errorf("DataType %v is not supported.", dt))
}

func shapeAndDataTypeOf(val reflect.Value) (shape []int32, dt DataType, err error) {
	typ := val.Type()
	for typ.Kind() == reflect.Array || typ.Kind() == reflect.Slice {
		shape = append(shape, int32(val.Len()))
		if val.Len() > 0 {
			val = val.Index(0)
		}
		typ = typ.Elem()
	}
	for _, t := range types {
		if typ.Kind() == t.typ.Kind() {
			return shape, DataType(t.dataType), nil
		}
	}
	return shape, dt, fmt.Errorf("unsupported type %v", typ)
}

func numElements(shape []int32) int32 {
	n := int32(1)
	for _, v := range shape {
		n *= v
	}
	return n
}

// It isn't safe to use reflect.SliceHeader as it uses a uintptr for Data and
// this is not inspected by the garbage collector
type sliceHeader struct {
	Data unsafe.Pointer
	Len  int
	Cap  int
}

package com.baidu.paddle.inference;

public class Tensor {

    long nativeTensorPointer;

    //构造函数
    public Tensor(long nativeTensorPointer) {
        this.nativeTensorPointer = nativeTensorPointer;
    }

    @Override
    protected void finalize() throws Throwable {
        destroyNativeTensor();
    }

    public void destroyNativeTensor() {
        if(nativeTensorPointer != 0) cppTensorDestroy(nativeTensorPointer);
        nativeTensorPointer = 0;
    }

    public void reshape(int dim_num, int[] shape) {
        cppTensorReshape(nativeTensorPointer, dim_num, shape);
    }

    public int getSize() {
        int[] shape = getShape();
        if (shape.length == 0) return 0;
        int size = 1;
        for (int i : shape) size *= i;
        return size;
    }

    public int[] getShape() {
        return cppTensorGetShape(nativeTensorPointer);
    }

    public String getName() {
        return cppTensorGetName(nativeTensorPointer);
    }

    public long getCppPaddleTensorPointer() {
        return nativeTensorPointer;
    }

    public void copyFromCpu(Object obj) {
        if (obj instanceof float[]) {
            cppTensorCopyFromCpuFloat(this.nativeTensorPointer, (float[]) obj);
        } else if (obj instanceof long[]) {
            cppTensorCopyFromCpuLong(this.nativeTensorPointer, (long[]) obj);
        } else if (obj instanceof int[]) {
            cppTensorCopyFromCpuInt(this.nativeTensorPointer, (int[]) obj);
        } else if (obj instanceof byte[]) {
            cppTensorCopyFromCpuByte(this.nativeTensorPointer, (byte[]) obj);
        } else if (obj instanceof boolean[]) {
            cppTensorCopyFromCpuBoolean(this.nativeTensorPointer, (boolean[]) obj);
        }
    }

    public void copyToCpu(Object obj) {
        if (obj instanceof float[]) {
            cppTensorCopyToCpuFloat(this.nativeTensorPointer, (float[]) obj);
        } else if (obj instanceof long[]) {
            cppTensorCopyToCpuLong(this.nativeTensorPointer, (long[]) obj);
        } else if (obj instanceof int[]) {
            cppTensorCopyToCpuInt(this.nativeTensorPointer, (int[]) obj);
        } else if (obj instanceof byte[]) {
            cppTensorCopyToCpuByte(this.nativeTensorPointer, (byte[]) obj);
        } else if (obj instanceof boolean[]) {
            cppTensorCopyToCpuBoolean(this.nativeTensorPointer, (boolean[]) obj);
        }
    }

    private native void cppTensorDestroy(long TensorPointer);

    private native void cppTensorReshape(long tensor, int dim_num, int[] shape);

    private native int[] cppTensorGetShape(long tensor);

    private native String cppTensorGetName(long tensor);

    private native void cppTensorCopyFromCpuFloat(long TensorPointer, float[] data);

    private native void cppTensorCopyFromCpuInt(long TensorPointer, int[] data);

    private native void cppTensorCopyFromCpuLong(long TensorPointer, long[] data);

    private native void cppTensorCopyFromCpuByte(long TensorPointer, byte[] data);

    private native void cppTensorCopyFromCpuBoolean(long TensorPointer, boolean[] data);

    private native void cppTensorCopyToCpuFloat(long TensorPointer, float[] data);

    private native void cppTensorCopyToCpuInt(long TensorPointer, int[] data);

    private native void cppTensorCopyToCpuLong(long TensorPointer, long[] data);

    private native void cppTensorCopyToCpuByte(long TensorPointer, byte[] data);

    private native void cppTensorCopyToCpuBoolean(long TensorPointer, boolean[] data);
}

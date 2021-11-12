package com.baidu.paddle.inference;

public class Tensor {

    long nativeTensorPointer;

    //构造函数
    public Tensor(long nativeTensorPointer) {
        this.nativeTensorPointer = nativeTensorPointer;
    }

    public void Reshape(int dim_num, int[] shape) {
        TensorReshape(nativeTensorPointer, dim_num, shape);
    }

    public int GetSize() {
        int[] shape = GetShape();
        if (shape.length == 0) return 0;
        int size = 1;
        for (int i : shape) size *= i;
        return size;
    }

    public int[] GetShape() {
        return TensorGetShape(nativeTensorPointer);
    }

    public String GetName() {
        return TensorGetName(nativeTensorPointer);
    }

    public long getCppPaddleTensorPointer() {
        return nativeTensorPointer;
    }

    public void CopyFromCpu(Object obj) {
        if (obj instanceof float[]) {
            TensorCopyFromCpuFloat(this.nativeTensorPointer, (float[]) obj);
        } else if (obj instanceof long[]) {
            TensorCopyFromCpuLong(this.nativeTensorPointer, (long[]) obj);
        } else if (obj instanceof int[]) {
            TensorCopyFromCpuInt(this.nativeTensorPointer, (int[]) obj);
        } else if (obj instanceof byte[]) {
            TensorCopyFromCpuByte(this.nativeTensorPointer, (byte[]) obj);
        } else if (obj instanceof boolean[]) {
            TensorCopyFromCpuBoolean(this.nativeTensorPointer, (boolean[]) obj);
        }
    }

    public void CopyToCpu(Object obj) {
        if (obj instanceof float[]) {
            TensorCopyToCpuFloat(this.nativeTensorPointer, (float[]) obj);
        } else if (obj instanceof long[]) {
            TensorCopyToCpuLong(this.nativeTensorPointer, (long[]) obj);
        } else if (obj instanceof int[]) {
            TensorCopyToCpuInt(this.nativeTensorPointer, (int[]) obj);
        } else if (obj instanceof byte[]) {
            TensorCopyToCpuByte(this.nativeTensorPointer, (byte[]) obj);
        } else if (obj instanceof boolean[]) {
            TensorCopyToCpuBoolean(this.nativeTensorPointer, (boolean[]) obj);
        }
    }

    private native void TensorReshape(long tensor, int dim_num, int[] shape);

    private native int[] TensorGetShape(long tensor);

    private native String TensorGetName(long tensor);

    private native void TensorCopyFromCpuFloat(long TensorPointer, float[] data);

    private native void TensorCopyFromCpuInt(long TensorPointer, int[] data);

    private native void TensorCopyFromCpuLong(long TensorPointer, long[] data);

    private native void TensorCopyFromCpuByte(long TensorPointer, byte[] data);

    private native void TensorCopyFromCpuBoolean(long TensorPointer, boolean[] data);

    private native void TensorCopyToCpuFloat(long TensorPointer, float[] data);

    private native void TensorCopyToCpuInt(long TensorPointer, int[] data);

    private native void TensorCopyToCpuLong(long TensorPointer, long[] data);

    private native void TensorCopyToCpuByte(long TensorPointer, byte[] data);

    private native void TensorCopyToCpuBoolean(long TensorPointer, boolean[] data);
}

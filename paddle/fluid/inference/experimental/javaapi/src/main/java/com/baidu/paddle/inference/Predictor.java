package com.baidu.paddle.inference;

public class Predictor {

    private long cppPaddlePredictorPointer;

    private long inputNum;

    private long outputNum;

    public Predictor(Config config) {
        cppPaddlePredictorPointer = createPredictor(config.getCppPaddleConfigPointer());
        config.resetCppPaddleConfigPointer();
        inputNum = getInputNum(cppPaddlePredictorPointer);
        outputNum = getOutputNum(cppPaddlePredictorPointer);
    }

    @Override
    protected void finalize() throws Throwable {
        destroyNativePredictor();
    }

    public static Predictor createPaddlePredictor(Config config){
        Predictor predictor = new Predictor(config);
        return predictor.cppPaddlePredictorPointer == 0L ? null : predictor;
    }

    public void destroyNativePredictor() {
        if(cppPaddlePredictorPointer != 0) cppPredictorDestroy(cppPaddlePredictorPointer);
        cppPaddlePredictorPointer = 0;
    }

    public String getInputNameById(long id){
        return getInputNameByIndex(this.cppPaddlePredictorPointer, id);
    }

    public String getOutputNameById(long id){
        return getOutputNameByIndex(this.cppPaddlePredictorPointer, id);
    }

    public Tensor getInputHandle(String name){
        long cppTensorPointer = getInputHandleByName(this.cppPaddlePredictorPointer, name);
        return cppTensorPointer == 0 ? null : new Tensor(cppTensorPointer);
    }

    public Tensor getOutputHandle(String name){
        long cppTensorPointer = getOutputHandleByName(this.cppPaddlePredictorPointer, name);
        return cppTensorPointer == 0 ? null : new Tensor(cppTensorPointer);
    }

    public void clearIntermediateTensor(){
        predictorClearIntermediateTensor(this.cppPaddlePredictorPointer);
    }

    public void tryShrinkMemory(){
        predictorTryShrinkMemory(this.cppPaddlePredictorPointer);
    }


    public boolean run(){
        return runPD(this.cppPaddlePredictorPointer);
    }

    public long getCppPaddlePredictorPointer() {
        return cppPaddlePredictorPointer;
    }

    public long getInputNum() {
        return inputNum;
    }

    public long getOutputNum() {
        return outputNum;
    }

    private native void cppPredictorDestroy(long cppPaddleConfigPointer);

    private native void predictorTryShrinkMemory(long cppPaddleConfigPointer);

    private native void predictorClearIntermediateTensor(long cppPaddleConfigPointer);

    private native long createPredictor(long cppPaddleConfigPointer);

    private native long getInputNum(long cppPaddlePredictorPointer);

    private native long getOutputNum(long cppPaddlePredictorPointer);

    private native String getInputNameByIndex(long cppPaddlePredictorPointer, long index);

    private native String getOutputNameByIndex(long cppPaddlePredictorPointer, long index);

    private native long getInputHandleByName(long cppPaddlePredictorPointer, String name);

    private native long getOutputHandleByName(long cppPaddlePredictorPointer, String name);

    private native boolean runPD(long cppPaddlePredictorPointer);
}

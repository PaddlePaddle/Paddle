package com.baidu.paddle.inference;

public class Config {
    private long cppPaddleConfigPointer;
    private String modelFile;
    private String paramsFile;
    private String modelDir;
    private String progFile;
    private int mathThreadsNum;


    public Config() {
        this.cppPaddleConfigPointer = createCppConfig();
    }

    @Override
    protected void finalize() throws Throwable {
        destroyNativeConfig();
    }

    public void destroyNativeConfig() {
        if(cppPaddleConfigPointer != 0) cppConfigDestroy(cppPaddleConfigPointer);
        cppPaddleConfigPointer = 0;
    }

    public boolean isValid() {
        if(cppPaddleConfigPointer == 0) return false;
        return isCppConfigValid(cppPaddleConfigPointer);
    }


    public void setCppModel(String modelFile, String paramsFile) {

        this.modelFile = modelFile;
        this.paramsFile = paramsFile;
        setCppModel(this.cppPaddleConfigPointer, modelFile, paramsFile);
    }

    public void setCppModelDir(String modelDir) {
        this.modelDir = modelDir;
        setCppModelDir(this.cppPaddleConfigPointer, modelDir);
    }

    public void setCppProgFile(String progFile){
        this.progFile = progFile;
        setCppProgFile(this.cppPaddleConfigPointer, progFile);
    }

    public void setCppParamsFile(String paramsFile){
        this.paramsFile = paramsFile;
        setCppParamsFile(this.cppPaddleConfigPointer, paramsFile);
    }

    public String getCppModelDir() {
        return modelDir(this.cppPaddleConfigPointer);
    }

    public String getCppProgFile(){
        return progFile(this.cppPaddleConfigPointer);
    }

    public String getCppParamsFile() {
        return paramsFile(this.cppPaddleConfigPointer);
    }

    public void setCpuMathLibraryNumThreads(int mathThreadsNum){
        this.mathThreadsNum = mathThreadsNum;
        setCpuMathLibraryNumThreads(this.cppPaddleConfigPointer, mathThreadsNum);
    }

    public int getCpuMathLibraryNumThreads(){
        return cpuMathLibraryNumThreads(this.cppPaddleConfigPointer);
    }

    public void enableMKLDNN(){
        enableMKLDNN(this.cppPaddleConfigPointer);
    }

    public boolean mkldnnEnabled(){
        return mkldnnEnabled(this.cppPaddleConfigPointer);
    }

    public void enableMkldnnBfloat16(){
        enableMkldnnBfloat16(this.cppPaddleConfigPointer);
    }

    public boolean mkldnnBfloat16Enabled(){
        return mkldnnBfloat16Enabled(this.cppPaddleConfigPointer);
    }

    public void enableUseGpu(long memorySize, int deviceId){
        enableUseGpu(this.cppPaddleConfigPointer, memorySize, deviceId);
    }

    public void disableGpu(){
        disableGpu(this.cppPaddleConfigPointer);
    }

    public boolean useGpu(){
        return useGpu(this.cppPaddleConfigPointer);
    }

    public int getGpuDeviceId(){
        return gpuDeviceId(this.cppPaddleConfigPointer);
    }

    public int getMemoryPoolInitSizeMb(){
        return memoryPoolInitSizeMb(this.cppPaddleConfigPointer);
    }

    public float getFractionOfGpuMemoryForPool(){
        return fractionOfGpuMemoryForPool(this.cppPaddleConfigPointer);
    }

    public void switchIrOptim(boolean flag){
        switchIrOptim(this.cppPaddleConfigPointer, flag);
    }

    public boolean irOptim(){
        return irOptim(this.cppPaddleConfigPointer);
    }

    public void switchIrDebug(boolean flag){
        switchIrDebug(this.cppPaddleConfigPointer, flag);
    }

    public void enableMemoryOptim(boolean flag){
        enableMemoryOptim(this.cppPaddleConfigPointer, flag);
    }

    public boolean memoryOptimEnabled(){
        return memoryOptimEnabled(this.cppPaddleConfigPointer);
    }

    public void enableProfile(){
        enableProfile(this.cppPaddleConfigPointer);
    }

    public boolean profileEnabled(){
        return profileEnabled(this.cppPaddleConfigPointer);
    }

    public void disableGlogInfo(){
        disableGlogInfo(this.cppPaddleConfigPointer);
    }

    public String summary(){
        return summary(this.cppPaddleConfigPointer);
    }

    public long getCppPaddleConfigPointer() {
        return cppPaddleConfigPointer;
    }

    public String getModelFile() {
        return modelFile;
    }

    public String getParamsFile() {
        return paramsFile;
    }

    public String getModelDir() {
        return modelDir;
    }

    public String getProgFile() {
        return progFile;
    }

    public int getMathThreadsNum() {
        return mathThreadsNum;
    }

    public void resetCppPaddleConfigPointer() {
        cppPaddleConfigPointer = 0;
    }

    private native void cppConfigDestroy(long cppPaddleConfigPointer);

    // 1. create Config

    private native long createCppConfig();

    private native boolean isCppConfigValid(long cppPaddleConfigPointer);

    // 2. not combined model settings

    private native void setCppModel(long cppPaddleConfigPointer, String modelFile, String paramsFile);

    // 3. combined model settings

    private native void setCppModelDir(long cppPaddleConfigPointer, String modelDir);

    private native void setCppProgFile(long cppPaddleConfigPointer, String modelFile);

    private native void setCppParamsFile(long cppPaddleConfigPointer, String paramsFile);

    private native String modelDir(long cppPaddleConfigPointer);

    private native String progFile(long cppPaddleConfigPointer);

    private native String paramsFile(long cppPaddleConfigPointer);

    // 4. cpu settings

    private native void setCpuMathLibraryNumThreads(long cppPaddleConfigPointer, int mathThreadsNum);

    private native int cpuMathLibraryNumThreads(long cppPaddleConfigPointer);

    // 5. MKLDNN settings

    private native void enableMKLDNN(long cppPaddleConfigPointer);

    private native boolean mkldnnEnabled(long cppPaddleConfigPointer);

    private native void enableMkldnnBfloat16(long cppPaddleConfigPointer);

    private native boolean mkldnnBfloat16Enabled(long cppPaddleConfigPointer);

    // 6. gpu setting

    // 这里有个bug java没有uint64 这里用 long代替
    // memorySize 太大的时候 java里long会是负数
    private native void enableUseGpu(long cppPaddleConfigPointer, long memorySize, int deviceId);

    private native void disableGpu(long cppPaddleConfigPointer);

    private native boolean useGpu(long cppPaddleConfigPointer);

    private native int gpuDeviceId(long cppPaddleConfigPointer);

    private native int memoryPoolInitSizeMb(long cppPaddleConfigPointer);

    private native float fractionOfGpuMemoryForPool(long cppPaddleConfigPointer);

    // 7. TensorRT use To Do



    // 8. optim setting

    private native void switchIrOptim(long cppPaddleConfigPointer, boolean flag);

    private native boolean irOptim(long cppPaddleConfigPointer);

    private native void switchIrDebug(long cppPaddleConfigPointer, boolean flag);

    // 9. enable memory optimization

    private native void enableMemoryOptim(long cppPaddleConfigPointer, boolean flag);

    private native boolean memoryOptimEnabled(long cppPaddleConfigPointer);

    // 10. profile setting

    private native void enableProfile(long cppPaddleConfigPointer);

    private native  boolean profileEnabled(long cppPaddleConfigPointer);

    // 11. log setting

    private native void  disableGlogInfo(long cppPaddleConfigPointer);

    // 12. view config configuration

    private native String summary(long cppPaddleConfigPointer);


}

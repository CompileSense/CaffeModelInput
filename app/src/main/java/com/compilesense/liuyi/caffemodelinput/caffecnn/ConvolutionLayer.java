package com.compilesense.liuyi.caffemodelinput.caffecnn;


import android.util.Log;

import com.compilesense.liuyi.caffemodelinput.messagepack.ParamUnpacker;

/**
 * Created by shenjingyuan002 on 2017/3/3.
 */

public class ConvolutionLayer implements Layer {
    private static final String TAG = "ConvolutionLayer";
    public final long nativeObject;
    private String name;
    private int stride[];
    private int pad[];
    private int group;
    private boolean nonLinear;
    private String paramFilePath;
    private boolean paramHasLoad = false;
    private float[][][][] weight;           // weight parameter of network
    private int[] weightShape;
    private float[] bias;                   // bias parameter of network

    public ConvolutionLayer(String name, int[] stride, int[] pad, int group, boolean nonLinear) {
        this.name = name;
        this.stride = stride;
        this.pad = pad;
        this.group = group;
        this.nonLinear = nonLinear;

        nativeObject = createConvolutionLayer(name, stride, pad, group, nonLinear);
        Log.d(TAG,"nativeObject:"+nativeObject);
    }

    public void releaseConvolutionLayer(){
        deleteConvolutionLayer(this.nativeObject);
        if (weight != null){
            weight = null;
        }
        if (bias != null){
            bias = null;
        }
    }

    public ConvolutionLayer setParamPath(String paramPath) {
        this.paramFilePath = paramPath;
        return this;
    }

    public ConvolutionLayer loadParam(){
        double time = System.currentTimeMillis();
        ParamUnpacker paramUnpacker = new ParamUnpacker();
        Object[] objects = paramUnpacker.unpackFunction(paramFilePath, new Class[]{float[][][][].class, float[].class});
        weight = (float[][][][]) objects[0];
        bias = (float[]) objects[1];
        weightShape = new int[4];
        weightShape[0] = weight.length;
        weightShape[1] = weight[0].length;
        weightShape[2] = weight[0][0].length;
        weightShape[3] = weight[0][0][0].length;
        Log.d(TAG,"--------------------------------------------");
        Log.d(TAG,"完成参数加载,用时:"+(System.currentTimeMillis() - time));
        Log.d(TAG,"参数大小: weight:"+( weightShape[0] )+","
                +( weightShape[1] ) + ","
                +( weightShape[2] ) + ","
                +( weightShape[3] ));
        Log.d(TAG,"weight last data: "+ (weight
                [ weightShape[0] -1]
                [ weightShape[1] -1]
                [ weightShape[2] -1]
                [ weightShape[3] -1]));
        Log.d(TAG,"参数大小: bias:"+(bias.length));
        Log.d(TAG,"bias last data: "+ (bias[bias.length - 1]));
        Log.d(TAG,"--------------------------------------------");


//        setKernel(this.nativeObject, weight, weightShape, bias);
        setParam1DimenWeight(this.nativeObject, float4DArrayTo1D(weight, weightShape), weightShape, bias);
        paramHasLoad = true;
        return this;
    }

    public static void testCompult(){
        testConvolutionLayerCompute();
    }

    @Override
    public Object compute(Object input) {
        return null;
    }

    @Override
    public void releaseLayer() {
        Log.d(TAG,"releaseLayer, name:"+ name + " , prt:"+ nativeObject);
        releaseConvolutionLayer();
    }

    private float[] float4DArrayTo1D(float[][][][] input, int[] shape){
        int size = 1;
        for (int s : shape) {
            size *= s;
        }
        if (size <= 0){
            return null;
        }

        float[] outPut = new float[size];
        int index = 0;

        for (float[][][] i : input){
            for (float[][] j: i){
                for (float[] k: j){
                    for (float l : k) {
                        outPut[index] = l;
                        index++;
                    }
                }
            }
        }
        return outPut;
    }

    private native long createConvolutionLayer(
            String name,
            int stride[],
            int pad[],
            int group,
            boolean nonLinear
    );

    private native void deleteConvolutionLayer(long nativeObject);
    //C++ 中JNI的方式4维数组太难处理了。
    private native void setParam(long objPrt,float[][][][] weight ,int[] weightShape,  float[] bias);
    //先将 4维数组转为1维再调用 JNI 接口。
    private native void setParam1DimenWeight(long objPrt,float[] weight ,int[] weightShape,  float[] bias);

    private static native void testConvolutionLayerCompute();
}

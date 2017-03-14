//
// Created by 神经元002 on 2017/3/9.
//
#include <utils.h>
#include <ComputeTool.h>
#include <ConvolutionLayer.h>
#include <ActivationLayer.h>
#include <stdlib.h>


extern "C" {

JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_caffemodelinput_NativeTest_testComputeIm2Col(
        JNIEnv *env, jobject instance
) {
    float * input = new float[50]{
            1,2,3,4,5,
            1,2,3,4,5,
            1,2,3,4,5,
            1,2,3,4,5,
            1,2,3,4,5,
            1,2,3,4,5,
            1,2,3,4,5,
            1,2,3,4,5,
            1,2,3,4,5,
            1,2,3,4,5
    };
    std::vector<size_t> inputShape(3);
    inputShape[0] = 5;
    inputShape[1] = 5;
    inputShape[2] = 2;
    MultiDimData<float> inputData(input, inputShape);

    float * kernel = new float[9]{
            1,2,3,4,5,6,7,8,9
    };
    std::vector<size_t> kernelShape(2);
    kernelShape[0] = 3;
    kernelShape[1] = 3;
    MultiDimData<float> kernelData(kernel, kernelShape);

    ConvolutionLayer::Params convolutionLayerParams;

    double  st = now_ms();
    MultiDimData<float> * outputPtr =  im2col(&inputData, &kernelData, convolutionLayerParams);
    LOGE("finish im2col, time: %f", now_ms() - st);
    LOGE("output dimens: %ui", (unsigned int) outputPtr->num_dimensions);
    int channel,height,width;
    channel = (int) outputPtr->get_c();
    height = (int) outputPtr->get_h();
    width = (int) outputPtr->get_w();
    for (int c = 0; c < channel; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int index = c * height * width + h * width + w;
                LOGE("output[%i][%i][%i]=%f",c,h,w,outputPtr->data_ptr[index]);
            }
        }
    }

    LOGE("release data");
}

JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_caffemodelinput_NativeTest_testComputeRelu(
        JNIEnv *env, jobject instance
){


#ifdef __MATH_NEON
    LOGD("fuck __arm__1");
    #define FUCKER
#endif


    #ifndef FUCKER
    LOGD("fuck __arm__2");
    #endif

    ActivationLayer activationLayer("activationLayer1",ActivationLayer::ReLU);
    float * data = new float[1000];
    for(int i = 0; i< 1000; i++) {
        data[i] = (-rand()%51+25);
    }
    for(int i = 0; i<20; i++){
        LOGD("input[%i]=%f",i,data[i]);
    }

    std::vector<size_t > shape(1);
    shape[0] = 1000;
    MultiDimData<float> inputData(data, shape);

    double st = now_ms();
    activationLayer.compute(&inputData);
    LOGD("time1---------------------------------------------: %f",now_ms() - st);
    for(int i = 0; i<20; i++){
        LOGD("result[%i]=%f",i,data[i]);
    }

    float * data2 = new float[1000];
    for(int i = 0; i< 1000; i++) {
        data[i] = (-rand()%51+25);
    }
    for(int i = 0; i<20; i++){
        LOGD("input[%i]=%f",i,data[i]);
    }
    st = now_ms();
    for(int i = 0; i< 1000; i++) {
        data[i]=(data[i]>0)?data[i]:0;
    }
    LOGD("time2---------------------------------------------: %f",now_ms() - st);
    for(int i = 0; i<20; i++){
        LOGD("result2[%i]=%f",i,data[i]);
    }

    delete[] data2;
}

JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_caffemodelinput_NativeTest_testComputePrelu(
        JNIEnv *env, jobject instance
){
    ActivationLayer activationLayer("activationLayer1",ActivationLayer::PReLU);
    float * data = new float[1000];
    for(int i = 0; i< 1000; i++) {
        data[i] = (-rand()%51+25);
    }
    for(int i = 0; i<20; i++){
        LOGD("input[%i]=%f",i,data[i]);
    }

    std::vector<size_t > shape(1);
    shape[0] = 1000;
    MultiDimData<float> inputData(data, shape);

    double st = now_ms();
    activationLayer.compute(&inputData);
    LOGD("time1---------------------------------------------: %f",now_ms() - st);
    for(int i = 0; i<20; i++){
        LOGD("result[%i]=%f",i,data[i]);
    }

    float * data2 = new float[1000];
    for(int i = 0; i< 1000; i++) {
        data[i] = (-rand()%51+25);
    }
    for(int i = 0; i<20; i++){
        LOGD("input[%i]=%f",i,data[i]);
    }
    st = now_ms();
    for(int i = 0; i< 1000; i++) {
        data[i]=(data[i]>0)?data[i]:0.25f*data[i];
    }
    LOGD("time2---------------------------------------------: %f",now_ms() - st);
    for(int i = 0; i<20; i++){
        LOGD("result2[%i]=%f",i,data[i]);
    }
    delete[] data2;
}

JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_caffemodelinput_NativeTest_testComputeTanh(
        JNIEnv *env, jobject instance
){


    ActivationLayer activationLayer("activationLayer1",ActivationLayer::TanH);
    float * data = new float[1000];
    float * data2 = new float[1000];
    for(int i = 0; i< 1000; i++) {
        data2[i] = data[i] = (-rand()%51+25);
    }
    for(int i = 0; i<20; i++){
        LOGD("input[%i]=%f",i,data[i]);
    }

    std::vector<size_t > shape(1);
    shape[0] = 1000;
    MultiDimData<float> inputData(data, shape);

    double st = now_ms();
    activationLayer.compute(&inputData);
    LOGD("time1---------------------------------------------: %f",now_ms() - st);
    for(int i = 0; i<20; i++){
        LOGD("result[%i]=%f",i,data[i]);
    }
    st = now_ms();
    for(int i = 0; i< 1000; i++) {
        data[i]=tanhf(data[i]);
    }
    LOGD("time2---------------------------------------------: %f",now_ms() - st);
    for(int i = 0; i<20; i++){
        LOGD("result2[%i]=%f",i,data[i]);
    }
    delete[] data2;
}
}



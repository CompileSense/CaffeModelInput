//
// Created by 神经元002 on 2017/3/7.
//

#include "FullyConnectedLayer.h"
#include "utils.h"
FullyConnectedLayer::FullyConnectedLayer(
        const std::string &name, bool nonLinear) : name(name), nonLinear(nonLinear) {
    this->paramHadLoad = false;
    LOGD("ConvolutionLayer create! name: %s",this->name.data());
}

FullyConnectedLayer::~FullyConnectedLayer() {
    LOGD("ConvolutionLayer delete! name: %s",this->name.data());
}
void FullyConnectedLayer::setParam(jfloatArray weightJArray, float* weight, int weightSize,
                                   jfloatArray biasJArray, float* bias, int biasSize) {
    this->weightJArray = weightJArray;
    this->weight = weight;
    this->weightSize = weightSize;
    this->biasJArray = biasJArray;
    this->bias = bias;
    this->biasSize = biasSize;

    this->paramHadLoad = true;
    LOGD("ConvolutionLayer setKernel! name: %s",this->name.data());
}

void FullyConnectedLayer::releaseParam(JNIEnv *env) {
    if (this->paramHadLoad){
        delete(weight);
        delete(bias);
    }
    this->paramHadLoad = false;
    LOGD("ConvolutionLayer releaseParam! name: %s",this->name.data());
}

#ifdef __cplusplus
extern "C" {
#endif
//构造一个 conv layer 实例
JNIEXPORT jlong JNICALL
Java_com_compilesense_liuyi_caffemodelinput_caffecnn_FullyConnectedLayer_createNativeObject(
        JNIEnv *env, jobject instance,
        jstring name_,
        jboolean nonLinear_
){
    //name
    const char * name = env->GetStringUTFChars(name_,NULL);

    //nonLinear
    bool nonLinear = nonLinear_;
    FullyConnectedLayer * fcLayerPrt = new FullyConnectedLayer(
            name,
            nonLinear
    );
    env->ReleaseStringUTFChars(name_,name);
    return (jlong) fcLayerPrt;
}

JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_caffemodelinput_caffecnn_FullyConnectedLayer_deleteNativeObject(
        JNIEnv *env, jobject instance,
        jlong nativeObjPrt_
){
    FullyConnectedLayer * prt = (FullyConnectedLayer *) nativeObjPrt_;
    prt->releaseParam(env);
    delete prt;
}

JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_caffemodelinput_caffecnn_FullyConnectedLayer_setParam(
        JNIEnv *env, jobject instance,
        jlong nativeObjPrt_,
        jfloatArray weight_,
        jfloatArray bias_
){
    int weightArrayLength = 0;
    float* weight = jFloatArray2prtFast(env, weight_, &weightArrayLength);

    int biasArrayLength = 0;
    float * bias = jFloatArray2prtFast(env, bias_, &biasArrayLength);
    FullyConnectedLayer * prt = (FullyConnectedLayer *) nativeObjPrt_;
    prt->setParam(weight_, weight, weightArrayLength,
                  bias_, bias, biasArrayLength);
    jFloatArrayRelease(env, weight_, weight);
    jFloatArrayRelease(env, bias_, bias);
}

#ifdef __cplusplus
}
#endif

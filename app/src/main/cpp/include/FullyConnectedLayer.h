//
// Created by 神经元002 on 2017/3/7.
//

#ifndef CAFFEMODELINPUT_FULLYCONECTEDLAYER_H
#define CAFFEMODELINPUT_FULLYCONECTEDLAYER_H

#include <string>
#include <jni.h>

class FullyConnectedLayer {
private:
    std::string name;
    bool nonLinear;

    bool paramHadLoad;
    jfloatArray weightJArray;
    float * weight;
    int weightSize;
    jfloatArray biasJArray;
    float * bias;
    float biasSize;

public:
    FullyConnectedLayer(const std::string &name, bool nonLinear);
    ~FullyConnectedLayer();
    void setParam(jfloatArray weightJArray, float* weight, int weightSize,
                  jfloatArray biasJArray, float* bias, int biasSize);
    void releaseParam(JNIEnv *env);
};


#endif //CAFFEMODELINPUT_FULLYCONECTEDLAYER_H

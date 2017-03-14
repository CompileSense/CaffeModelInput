//
// Created by 神经元002 on 2017/3/10.
//

#include "ActivationLayer.h"
#include <ComputeTool.h>

ActivationLayer::ActivationLayer(const std::string name, Type type_)
        : name(name), type(type_)
{
    LOGD("ActivationLayer create. name: %s", name.data());
}

void ActivationLayer::compute(MultiDimData<float> *input) {
    switch (type){
        case ReLU:
            relu(input);
            break;
        case PReLU:
            prelu(input);
            break;
        case TanH:
            tanh(input);
            break;
        case Abs:
            abs(input);
            break;
    }
}
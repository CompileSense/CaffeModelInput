//
// Created by 神经元002 on 2017/3/10.
//

#include "ActivationLayer.h"
#include <ComputeTool.h>

ActivationLayer::ActivationLayer(const std::string name, Type type_)
        : name(name), type(type_)
{

}

void ActivationLayer::compute(MultiDimData<float> *input) {
    if (type == ReLU){
        relu(input);
    }
}
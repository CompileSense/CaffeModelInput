//
// Created by 神经元002 on 2017/3/10.
//

#ifndef CAFFEMODELINPUT_ACTIVATIONLAYER_H
#define CAFFEMODELINPUT_ACTIVATIONLAYER_H

#include <MultiDimData.h>

class ActivationLayer {
public:
    enum Type {ReLU,PReLU,TanH,Abs};
    ActivationLayer(const std::string name, Type type_);
    void compute(MultiDimData<float> *input);

private:
    std::string name;
    Type type;
};


#endif //CAFFEMODELINPUT_ACTIVATIONLAYER_H

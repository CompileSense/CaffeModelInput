//
// Created by 神经元002 on 2017/3/9.
//

#ifndef CAFFEMODELINPUT_MULTIDIMDATA_H
#define CAFFEMODELINPUT_MULTIDIMDATA_H

#include <vector>
#include <cwchar>
#include <memory>
#include <math.h>

/**
 * 在网络中的数据大多是多维的,如果用一维数组表示,那么 Shape 这个数据就是形影不离的。
 * 所以将其封装为一个类型。
 */
using namespace std;
template<typename dataType>
class MultiDimData {
public:
    dataType * data_ptr;//用 new 来申请内存
    vector<size_t> shape;//{w,h,c,n}
    size_t num_dimensions;

    MultiDimData(){}
    MultiDimData(dataType * data_ptr, std::vector<size_t> shape):
            data_ptr(data_ptr), shape(shape){
        num_dimensions = shape.size();
    }
    ~MultiDimData(){
        delete[] data_ptr;
    }

    size_t totalSize(){
        int num = 1;
        for (int i = 0; i < num_dimensions; ++i){
            num *= shape[i];
        }
        return (size_t)num;
    }

    size_t get_n(){
        return  (num_dimensions < 4) ? 0 : shape[3];
    }
    size_t get_c(){
        return  (num_dimensions < 3) ? 0 : shape[2];
    }
    size_t get_h(){
        return  (num_dimensions < 2) ? 0 : shape[1];
    }
    size_t get_w(){
        return  (num_dimensions < 1) ? 0 : shape[0];
    }
};

#endif //CAFFEMODELINPUT_MULTIDIMDATA_H

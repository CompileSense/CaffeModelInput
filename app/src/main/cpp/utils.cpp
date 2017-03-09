//
// Created by 神经元002 on 2017/3/3.
//

#include "include/utils.h"
#include <jni.h>
#include <string>
#include <vector>

double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec*1000. + tv.tv_usec/1000.;
}

/**
 * 在 java 数组的访问方式中, GetFloatArrayRegion 是效率最高的,
 * 但是会占用跟多的内存。GetLongArrayElements 效率较低,但是线程安全。
 * 一遍 int 数组都不会太长,而 param 中 weight 是 float 类型的,
 * 所以 float 实现两种方式。
 */
int * jIntArray2prt( JNIEnv *env, jintArray jArray, int* length){
    *length = env->GetArrayLength(jArray);
    std::vector<int> fuker(1,3);

    int* cPrt = new int[ *length ];
    memset(cPrt,0,sizeof(jint) * (*length) );
    env->GetIntArrayRegion(jArray, 0, *length, cPrt);
    return cPrt;
}
//注意释放内存,调用这个后要调用 jFloatArrayRelease 。
float * jFloatArray2prtFast(JNIEnv *env, jfloatArray jArray, int *length){
    *length = env->GetArrayLength(jArray);
    float* cPrt = new float[*length];
    memset(cPrt,0,sizeof(jint) * (*length) );
    env->GetFloatArrayRegion(jArray, 0, *length, cPrt);
    return cPrt;
}
//将本地数数据返回到 JVM 的数组中。这里并没有 delete C++ 数组,要手动销毁
void setPrt2JFloatArray(JNIEnv *env, jfloatArray jArray, float *prt, int length){
    env->SetFloatArrayRegion(jArray,0,length, prt);
}

//内存不一定会在 C++ 的内存区域里, 如果在 JVM 里, 释放 JVM 里的内存会引起崩溃。
float * jFloatArray2prt(JNIEnv *env, jfloatArray jArray, int *length){
    *length = env->GetArrayLength(jArray);
    jboolean isCopy;
    float * cPrt = env->GetFloatArrayElements(jArray, &isCopy);
    env->ReleaseFloatArrayElements(jArray, cPrt, 0);
    return cPrt;
}

void jFloatArrayRelease(JNIEnv *env,jfloatArray jArray, float * prt){
    //发生未知BUG
//    env->ReleaseFloatArrayElements(jArray, prt, 0);
}


/**
 * 注意释放返回的数据
 */
float * jFloatArrayFrom4DimensionJavaArray(JNIEnv *env,
                                           jobjectArray jArray,
                                           int *weightShape, int weightShapeLength) {
    //检查数组总大小
    int totalSize = 1;
    for (int i = 0; i < weightShapeLength; ++i) {
        totalSize *= weightShape[0];
    }
    if (totalSize <= 0){
        return NULL;
    }

    //申请内存
    float * data;
    try {
        data = new float[totalSize];
    }catch (std::bad_alloc &memExp){
        data = NULL;
        LOGE("convert jFloatArray failed! file:%s, line:%i", __FILE__, __LINE__ );
        return NULL;
    }

    //遍历4维数组,将数据赋给 data
    //TODO: 使用 NEON 等 SIMD 技术优化。
    int index = 0;
    jobjectArray * sub1 = (jobjectArray *) parasMultiDimenArray(env, jArray);
    for (int i = 0; i < weightShape[0]; ++i) {//a[i][][][]
        jobjectArray * sub2 = (jobjectArray *) parasMultiDimenArray(env, sub1[i]);
        for (int j = 0; j < weightShape[1]; ++j){//a[i][j][][]
            jfloatArray * sub3 = (jfloatArray *)parasMultiDimenArray(env,true, sub2[j]);

            for (int k = 0; k < weightShape[2]; ++k) {//a[i][j][k][]
                float * tmp = pareArray(env, sub3[k]);

                for (int l = 0; l < weightShape[3]; ++l) {//a[i][j][k][l]
                    data[index] = tmp[l];
                    index++;
                }

                env->ReleaseFloatArrayElements(sub3[k], tmp, 0);
                env->DeleteLocalRef(sub3[k]);
            }

            env->DeleteLocalRef(sub2[j]);
            delete sub3;
        }
        env->DeleteLocalRef(sub1[i]);
        delete sub2;
    }
    delete sub1;

    return data;
}

/**
 * 从多维数数组中获取子数组
 *
 * 注意释放内存,每调用一次这个函数必须释放一次指针。
 *
 * @ param: lastMulti, 因为倒数第二维度需要获取到具体类型的数组,所以添加这个参数。
 *      默认为 false, 在为 true 时返回 jFloatArray, 在为 false 时返回 jObjectArray.
 * @ param: * length_, 可选, 返回子数组的长度。
 */
jarray * parasMultiDimenArray(JNIEnv *env, bool lastMulti, jobjectArray jArray, int * length_){

    int length = 0;
    length = env->GetArrayLength(jArray);
    if (length_ != NULL){
        *length_ = length;
    }
    if(length == 0){
        return NULL;
    }

//    if (lastMulti){
//        jfloatArray * subArray = new jfloatArray[length];
//        for (int i = 0; i < length; ++i) {
//            subArray[i] = (jfloatArray) env->GetObjectArrayElement(jArray, i);
//        }
//        return subArray;
//
//    } else {
//        jobjectArray * subArray = new jobjectArray[length];
//        for (int i = 0; i < length; ++i) {
//            subArray[i] = (jobjectArray) env->GetObjectArrayElement(jArray, i);
//        }
//        return subArray;
//    }

    std::vector<jarray> subArray;
    if (lastMulti){
        for (int i = 0; i < length; ++i) {
            jfloatArray item = (jfloatArray) env->GetObjectArrayElement(jArray, i);
            subArray.push_back(item);
        }

    } else {
        for (int i = 0; i < length; ++i) {
            jobjectArray item = (jobjectArray) env->GetObjectArrayElement(jArray, i);
            subArray.push_back(item);
        }

    }

    return &subArray[0];

}
jarray * parasMultiDimenArray(JNIEnv *env, bool lastMultiArray, jobjectArray jArray){
    parasMultiDimenArray(env, lastMultiArray, jArray, NULL);
};
jarray * parasMultiDimenArray(JNIEnv *env, jobjectArray jArray, int * length){
    parasMultiDimenArray(env, false, jArray, length);
};
jarray * parasMultiDimenArray(JNIEnv *env, jobjectArray jArray){
    parasMultiDimenArray(env, jArray, NULL);
};

float * pareArray(JNIEnv *env, jfloatArray jArray, int *length_){
    int length = env->GetArrayLength(jArray);
    if (length_ != NULL){
        *length_ = length;
    }
    if (length == 0){
        return NULL;
    }

    jfloat * elements = env->GetFloatArrayElements(jArray, NULL);
    return elements;
}
jfloat * pareArray(JNIEnv *env, jfloatArray jArray){
    pareArray(env, jArray, NULL);
};
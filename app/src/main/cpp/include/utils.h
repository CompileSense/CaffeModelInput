//
// Created by 神经元002 on 2017/3/3.
//

#ifndef CAFFEMODELINPUT_UTILS_H
#define CAFFEMODELINPUT_UTILS_H

#include <jni.h>
#include <vector>
#include <cwchar>
#include "android/log.h"

#define LOG_TAG "Native"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))

#define max(a, b)  (((a) > (b)) ? (a) : (b))

double now_ms(void);

/**
 * 注意释放返回的数据
 */
float * jFloatArrayFrom4DimensionJavaArray(JNIEnv *env,
                                           jobjectArray jArray,
                                           int *weightShape, int weightShapeLength);

int* jIntArray2prt (JNIEnv *env, jintArray jArray, int* length);

float * jFloatArray2prtFast (JNIEnv *env, jfloatArray jArray, int *length);
void setPrt2JFloatArray(JNIEnv *env, jfloatArray jArray, float *prt, int length);

float * jFloatArray2prt (JNIEnv *env, jfloatArray jArray, int *length);
void jFloatArrayRelease (JNIEnv *env,jfloatArray jArray, float * prt);

jarray * parasMultiDimenArray(JNIEnv *env, bool lastMultiArray, jobjectArray jArray, int * length);
jarray * parasMultiDimenArray(JNIEnv *env, bool lastMultiArray, jobjectArray jArray);
jarray * parasMultiDimenArray(JNIEnv *env, jobjectArray jArray, int * length);
jarray * parasMultiDimenArray(JNIEnv *env, jobjectArray jArray);

jfloat * pareArray(JNIEnv *env, jfloatArray jArray, int * length);
jfloat * pareArray(JNIEnv *env, jfloatArray jArray);

#endif //CAFFEMODELINPUT_UTILS_H

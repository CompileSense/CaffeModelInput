#include <jni.h>
#include <string>
#include "include/ConvolutionLayer.h"
#include "include/utils.h"
#include <memory>


extern "C"
jstring
Java_com_compilesense_liuyi_caffemodelinput_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}
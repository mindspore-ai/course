# 端侧C++推理流程

## 实验介绍

本实验主要以MindSpore Lite图像分类demo为例，介绍端侧推理流程及如何使用C++ API实现推理过程。

进行本实验前，建议先完成[图像分类demo部署章节](lite_demo_deploy)，并参考该章节下`code`文件夹中的源代码。

## 实验目的

- 理解端侧推理架构，并掌握C++ API使用方法。

## 预备知识

- 具备一定的Android开发基础知识
- 了解并熟悉MindSpore AI计算框架，MindSpore官网：<https://www.mindspore.cn/>

## 实验环境

- Android手机
- Android Studio >= 3.2 (推荐4.0以上版本)
- NDK >= 21.3
- CMake >= 3.10.2  
- Android SDK >= 26
- JDK >= 1.8

## 实验步骤

### 程序结构

端侧图像分类demo分为JAVA层和JNI层，其中，JAVA层主要通过Android Camera 2 API实现摄像头获取图像帧，以及相应的图像处理等功能；JNI层完成模型推理的过程。

程序结构如下所示：

```text
app
├── src/main
│   ├── assets # 资源文件
|   |   └── mobilenetv2.ms # 存放模型文件
│   |
│   ├── cpp # 模型加载和预测主要逻辑封装类
|   |   ├── ..
|   |   ├── mindspore_lite_x.x.x-minddata-arm64-cpu #MindSpore Lite版本
|   |   ├── MindSporeNetnative.cpp # MindSpore调用相关的JNI方法
│   |   └── MindSporeNetnative.h # 头文件
|   |   └── MsNetWork.cpp # MindSpre接口封装
│   |
│   ├── java # java层应用代码
│   │   └── com.mindspore.himindsporedemo
│   │       ├── gallery.classify # 图像处理及MindSpore JNI调用相关实现
│   │       │   └── ...
│   │       └── widget # 开启摄像头及绘制相关实现
│   │           └── ...
│   │
│   ├── res # 存放Android相关的资源文件
│   └── AndroidManifest.xml # Android配置文件
│
├── CMakeLists.txt # cmake编译入口文件
│
├── build.gradle # 其他Android配置文件
├── download.gradle # 工程依赖文件下载
└── ...
```

### 依赖项说明

Android JNI层调用MindSpore C++ API时，需要相关库文件支持。可通过MindSpore Lite[源码编译](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/build.html)生成`mindspore-lite-{version}-minddata-{os}-{device}.tar.gz`库文件包并解压缩（包含`libmindspore-lite.so`库文件和相关头文件），在本例中需使用生成带图像预处理模块的编译命令。

> version：输出件版本号，与所编译的分支代码对应的版本一致。
>
> device：当前分为cpu（内置CPU算子）和gpu（内置CPU和GPU算子）。
>
> os：输出件应部署的操作系统。

本示例中，build过程由download.gradle文件自动下载MindSpore Lite 版本文件，并放置在`app/src/main/cpp/`目录下。

在app的`build.gradle`文件中配置CMake编译支持，以及`arm64-v8a`的编译支持，如下所示：

```text
android{
    defaultConfig{
        externalNativeBuild{
            cmake{
                arguments "-DANDROID_STL=c++_shared"
            }
        }

        ndk{
            abiFilters 'arm64-v8a'
        }
    }
}
```

在`app/CMakeLists.txt`文件中建立`.so`库文件链接，如下所示。

```text
# ============== Set MindSpore Dependencies. =============
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/third_party/flatbuffers/include)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION})
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/include/ir/dtype)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/include/schema)

add_library(mindspore-lite SHARED IMPORTED )
add_library(minddata-lite SHARED IMPORTED )

set_target_properties(mindspore-lite PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/lib/libmindspore-lite.so)
set_target_properties(minddata-lite PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/lib/libminddata-lite.so)
# --------------- MindSpore Lite set End. --------------------

# Link target library.
target_link_libraries(
    ...
     # --- mindspore ---
        minddata-lite
        mindspore-lite
    ...
)
```

### 端侧推理代码解析

在JNI层调用MindSpore Lite C++ API实现端测推理。

推理代码流程如下，完整代码请参见`src/cpp/MindSporeNetnative.cpp`。

1. 加载MindSpore Lite模型文件，构建上下文、会话以及用于推理的计算图。  

    - 加载模型文件：创建并配置用于模型推理的上下文

        ```cpp
        // Buffer is the model data passed in by the Java layer
        jlong bufferLen = env->GetDirectBufferCapacity(buffer);
        char *modelBuffer = CreateLocalModelBuffer(env, buffer);  
        ```

    - 创建会话

        ```cpp
        void **labelEnv = new void *;
        MSNetWork *labelNet = new MSNetWork;
        *labelEnv = labelNet;

        // Create context.
        lite::Context *context = new lite::Context;
        context->thread_num_ = numThread;  //Specify the number of threads to run inference

        // Create the mindspore session.
        labelNet->CreateSessionMS(modelBuffer, bufferLen, context);
        delete(context);

        ```

    - 加载模型文件并构建用于推理的计算图

        ```cpp
        void MSNetWork::CreateSessionMS(char* modelBuffer, size_t bufferLen, std::string name, mindspore::lite::Context* ctx)
        {
            CreateSession(modelBuffer, bufferLen, ctx);  
            session = mindspore::session::LiteSession::CreateSession(ctx);
            auto model = mindspore::lite::Model::Import(modelBuffer, bufferLen);
            int ret = session->CompileGraph(model);
        }
        ```

2. 将输入图片转换为传入MindSpore模型的Tensor格式。

    - 将待检测图片数据转换为输入MindSpore模型的Tensor。

        ```cpp
        if (!BitmapToLiteMat(env, srcBitmap, &lite_mat_bgr)) {
         MS_PRINT("BitmapToLiteMat error");
         return NULL;
        }
        if (!PreProcessImageData(lite_mat_bgr, &lite_norm_mat_cut)) {
         MS_PRINT("PreProcessImageData error");
         return NULL;
        }

        ImgDims inputDims;
        inputDims.channel = lite_norm_mat_cut.channel_;
        inputDims.width = lite_norm_mat_cut.width_;
        inputDims.height = lite_norm_mat_cut.height_;

        // Get the mindsore inference environment which created in loadModel().
        void **labelEnv = reinterpret_cast<void **>(netEnv);
        if (labelEnv == nullptr) {
         MS_PRINT("MindSpore error, labelEnv is a nullptr.");
         return NULL;
        }
        MSNetWork *labelNet = static_cast<MSNetWork *>(*labelEnv);

        auto mSession = labelNet->session();
        if (mSession == nullptr) {
         MS_PRINT("MindSpore error, Session is a nullptr.");
         return NULL;
        }
        MS_PRINT("MindSpore get session.");

        auto msInputs = mSession->GetInputs();
        if (msInputs.size() == 0) {
         MS_PRINT("MindSpore error, msInputs.size() equals 0.");
         return NULL;
        }
        auto inTensor = msInputs.front();

        float *dataHWC = reinterpret_cast<float *>(lite_norm_mat_cut.data_ptr_);
        // Copy dataHWC to the model input tensor.
        memcpy(inTensor->MutableData(), dataHWC,
             inputDims.channel * inputDims.width * inputDims.height * sizeof(float));
        ```

3. 对输入Tensor按照模型进行推理，获取输出Tensor，并进行后处理。

   - 图执行，端测推理。

        ```cpp
        // After the model and image tensor data is loaded, run inference.
        auto status = mSession->RunGraph();
        ```

   - 获取输出数据。

        ```cpp
        auto names = mSession->GetOutputTensorNames();
        std::unordered_map<std::string,mindspore::tensor::MSTensor *> msOutputs;
        for (const auto &name : names) {
            auto temp_dat =mSession->GetOutputByTensorName(name);
            msOutputs.insert(std::pair<std::string, mindspore::tensor::MSTensor *> {name, temp_dat});
          }
         std::string resultStr = ProcessRunnetResult(::RET_CATEGORY_SUM,
                                              ::labels_name_map, msOutputs);
        ```

   - 根据模型，进行输出数据的后续处理。

        ```cpp
        std::string ProcessRunnetResult(const int RET_CATEGORY_SUM, const char *const labels_name_map[],
                 std::unordered_map<std::string, mindspore::tensor::MSTensor *> msOutputs) {
         // Get the branch of the model output.
         // Use iterators to get map elements.
         std::unordered_map<std::string, mindspore::tensor::MSTensor *>::iterator iter;
         iter = msOutputs.begin();

         // The mobilenetv2.ms model output just one branch.
         auto outputTensor = iter->second;

         int tensorNum = outputTensor->ElementsNum();
         MS_PRINT("Number of tensor elements:%d", tensorNum);

         // Get a pointer to the first score.
         float *temp_scores = static_cast<float *>(outputTensor->MutableData());
         float scores[RET_CATEGORY_SUM];
         for (int i = 0; i < RET_CATEGORY_SUM; ++i) {
          scores[i] = temp_scores[i];
         }

         float unifiedThre = 0.5;
         float probMax = 1.0;
         for (size_t i = 0; i < RET_CATEGORY_SUM; ++i) {
          float threshold = g_thres_map[i];
          float tmpProb = scores[i];
          if (tmpProb < threshold) {
           tmpProb = tmpProb / threshold * unifiedThre;
          } else {
           tmpProb = (tmpProb - threshold) / (probMax - threshold) * unifiedThre + unifiedThre;
         }
          scores[i] = tmpProb;
        }

         for (int i = 0; i < RET_CATEGORY_SUM; ++i) {
         if (scores[i] > 0.5) {
             MS_PRINT("MindSpore scores[%d] : [%f]", i, scores[i]);
          }
         }

         // Score for each category.
         // Converted to text information that needs to be displayed in the APP.
         std::string categoryScore = "";
         for (int i = 0; i < RET_CATEGORY_SUM; ++i) {
          categoryScore += labels_name_map[i];
          categoryScore += ":";
          std::string score_str = std::to_string(scores[i]);
          categoryScore += score_str;
          categoryScore += ";";
         }
           return categoryScore;
        }

        ```

### 实现目标检测场景

在理解了端侧推理流程之后，可以尝试完善目标检测场景。补充目录中`code/object_detection/app/src/main/cpp/MindSporeNetnative.cpp`文件，完成推理流程，便可以使用相同方法在手机中部署目标检测demo。

## 实验结论

本实验基于MindSpore Lite预置模型完成了端侧推理过程，可在Android手机中体验目标检测功能。

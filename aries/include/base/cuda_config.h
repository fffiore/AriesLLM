#ifndef BLAS_HELPER_H
#define BLAS_HELPER_H
#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace kernel {
  struct CudaConfig {
      cudaStream_t stream = nullptr;
      cublasHandle_t cublas_handle = nullptr;

      CudaConfig() {
          // 构造时创建句柄
          if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
              // 处理错误，这里忽略
          }
      }

      ~CudaConfig() {
          // 析构时销毁
          if (cublas_handle) {
              cublasDestroy(cublas_handle);
          }
          // stream 的销毁通常在外部管理
      }
  };
}
#endif  // BLAS_HELPER_H

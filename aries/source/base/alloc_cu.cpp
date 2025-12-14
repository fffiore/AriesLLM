#include <cuda_runtime_api.h>
#include "base/alloc.h"
#include <glog/logging.h>
#include <cstdio>

namespace base {

CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {}

void* CUDADeviceAllocator::allocate(size_t byte_size) const {
  int id = -1;
  cudaError_t state = cudaGetDevice(&id);
  CHECK(state == cudaSuccess);

  // 大于 1MB 的走 big_buffers 逻辑
  if (byte_size > 1024 * 1024) {
    auto& big_buffers = big_buffers_map_[id];
    int sel_id = -1;
    // 寻找 best fit (最小的足够大的块)，只要够大 (>= byte_size) 且空闲 (!busy) 就可以复用
    for (int i = 0; i < big_buffers.size(); i++) {
      if (big_buffers[i].byte_size >= byte_size && !big_buffers[i].busy) {
        if (sel_id == -1 || big_buffers[sel_id].byte_size > big_buffers[i].byte_size) {
          sel_id = i;
        }
      }
    }

    // 如果找到了合适的块，直接复用
    if (sel_id != -1) {
      big_buffers[sel_id].busy = true;
      return big_buffers[sel_id].data;
    }

    // 没找到，申请新的
    void* ptr = nullptr;
    state = cudaMalloc(&ptr, byte_size);
    if (cudaSuccess != state) {
      char buf[256];
      snprintf(buf, 256,
               "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
               "left on device.",
               byte_size >> 20);
      LOG(ERROR) << buf;
      return nullptr;
    }
    big_buffers.emplace_back(ptr, byte_size, true);
    return ptr;
  }

  // 小内存分配逻辑，小内存碎片化影响较小
  auto& cuda_buffers = cuda_buffers_map_[id];
  for (int i = 0; i < cuda_buffers.size(); i++) {
    if (cuda_buffers[i].byte_size >= byte_size && !cuda_buffers[i].busy) {
      cuda_buffers[i].busy = true;
      no_busy_cnt_[id] -= cuda_buffers[i].byte_size;
      return cuda_buffers[i].data;
    }
  }
  void* ptr = nullptr;
  state = cudaMalloc(&ptr, byte_size);
  if (cudaSuccess != state) {
    char buf[256];
    snprintf(buf, 256,
             "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
             "left on device.",
             byte_size >> 20);
    LOG(ERROR) << buf;
    return nullptr;
  }
  cuda_buffers.emplace_back(ptr, byte_size, true);
  return ptr;
}

void CUDADeviceAllocator::release(void* ptr) const {
  if (!ptr) return;

  // 在 small buffers 中查找并释放
  for (auto& it : cuda_buffers_map_) {
    auto& cuda_buffers = it.second;
    for (int i = 0; i < cuda_buffers.size(); i++) {
      if (cuda_buffers[i].data == ptr) {
        // 找到了，标记为空闲
        no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
        cuda_buffers[i].busy = false;
        
        // 检查是否需要清理碎片 (闲置内存 > 1GB)
        if (no_busy_cnt_[it.first] > 1024ULL * 1024 * 1024) {
             // 创建新 vector，只保留 busy 的
             std::vector<CudaMemoryBuffer> new_buffers;
             cudaSetDevice(it.first);
             for(const auto& buf : cuda_buffers) {
                 if(!buf.busy) {
                     cudaFree(buf.data);
                 } else {
                     new_buffers.push_back(buf);
                 }
             }
             cuda_buffers = new_buffers;
             no_busy_cnt_[it.first] = 0;
             // 当前 ptr 标记为 false
        }
        return; 
      }
    }
  }
  
  // 在 big buffers 中查找
  for (auto& it : big_buffers_map_) {
      auto& big_buffers = it.second;
      for (int i = 0; i < big_buffers.size(); i++) {
        if (big_buffers[i].data == ptr) {
          big_buffers[i].busy = false;
          return;
        }
      }
  }

  // 没找到，直接 Free
  cudaFree(ptr);
}

std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;

}  // namespace base
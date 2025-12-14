#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <cmath>
#include <cub/cub.cuh>
#include <cuda_runtime_api.h>
#include <math_constants.h>

namespace kernel {

constexpr int BLOCK_SIZE_M = 64;
constexpr int BLOCK_SIZE_N = 64;
// 编译的最大 head dim（用于静态数组分配与性能路径）
constexpr int HEAD_SIZE = 64;
// 为 cub::BlockReduce 指定的 block 线程数（launch 时必须匹配）
constexpr int BLOCK_THREADS = 128;

__global__ void flash_attn_prefill_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    float* __restrict__ O,
    int32_t pos_val,
    int32_t num_heads,
    int32_t kv_mul,
    int32_t head_dim,
    int32_t seq_len,
    int32_t kv_dim,
    int32_t max_seq_len_cache,
    float sm_scale
) {
    if (head_dim > HEAD_SIZE) return;
    if (blockDim.x != BLOCK_THREADS) return;

    int head_idx = blockIdx.y;
    int kv_head_idx = head_idx / kv_mul;
    int q_block_idx = blockIdx.z;

    int q_start = q_block_idx * BLOCK_SIZE_M;
    int q_len = min(BLOCK_SIZE_M, seq_len - q_start);
    if (q_len <= 0) return;

    int tx = threadIdx.x;

    extern __shared__ float smem[];
    float* sQ = smem; 
    float* sK = sQ + (BLOCK_SIZE_M * HEAD_SIZE);
    float* sV = sK + (BLOCK_SIZE_N * HEAD_SIZE);

    float acc_o[HEAD_SIZE];        
    float acc_m_local[BLOCK_SIZE_M > 0 ? BLOCK_SIZE_M : 1]; 
    float acc_l_local[BLOCK_SIZE_M > 0 ? BLOCK_SIZE_M : 1];

    for (int d = 0; d < head_dim; ++d) acc_o[d] = 0.0f;
    for (int i = 0; i < q_len; ++i) {
        acc_m_local[i] = -CUDART_INF_F;
        acc_l_local[i] = 0.0f;
    }

    // Load Q (保持不变)
    for (int i = tx; i < q_len * head_dim; i += blockDim.x) {
        int row = i / head_dim;
        int col = i % head_dim;
        int q_seq_idx = q_start + row;
        int q_index = q_seq_idx * (num_heads * head_dim) + head_idx * head_dim + col;
        sQ[row * head_dim + col] = Q[q_index];
    }
    __syncthreads();

    int start_pos = pos_val;
    int total_k_len = start_pos + seq_len;
    int num_k_blocks = (total_k_len + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;

    for (int k_step = 0; k_step < num_k_blocks; ++k_step) {
        int k_start = k_step * BLOCK_SIZE_N;
        int k_valid_len = min(BLOCK_SIZE_N, total_k_len - k_start);
        int kv_offset_base = k_start * kv_dim + kv_head_idx * head_dim;

        // Load K/V (保持不变)
        for (int i = tx; i < k_valid_len * head_dim; i += blockDim.x) {
            int row = i / head_dim;
            int col = i % head_dim;
            int offset = row * kv_dim + col;
            sK[row * head_dim + col] = K_cache[kv_offset_base + offset];
            sV[row * head_dim + col] = V_cache[kv_offset_base + offset];
        }
        __syncthreads();

        if (tx < q_len) {
            int q_idx_local = tx;
            
            for (int k = 0; k < k_valid_len; ++k) {
                int k_idx_global = k_start + k;
                int q_idx_global = q_start + q_idx_local;
                int current_pos = start_pos + q_idx_global;
                if (k_idx_global > current_pos) continue;

                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += sQ[q_idx_local * head_dim + d] * sK[k * head_dim + d];
                }
                score *= sm_scale;

                float m_prev = acc_m_local[q_idx_local];
                float m_curr = fmaxf(m_prev, score);
                float p_prev = expf(m_prev - m_curr);
                float p_curr = expf(score - m_curr);

                float l_prev = acc_l_local[q_idx_local];
                acc_l_local[q_idx_local] = l_prev * p_prev + p_curr;
                acc_m_local[q_idx_local] = m_curr;

                // acc_o 始终积累，利用 p_prev 缩放之前的和
                for (int d = 0; d < head_dim; ++d) {
                    acc_o[d] = acc_o[d] * p_prev + sV[k * head_dim + d] * p_curr;
                }
            }
        }
        __syncthreads();
    } // k_step loop ends

    if (tx < q_len) {
        int q_idx_local = tx;
        float inv_l = 1.0f / (acc_l_local[q_idx_local] + 1e-12f);
        int q_seq_idx = q_start + q_idx_local;
        int out_offset = q_seq_idx * (num_heads * head_dim) + head_idx * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            O[out_offset + d] = acc_o[d] * inv_l;
        }
    }
}

__global__ void standard_decode_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    float* __restrict__ O,
    const int32_t* __restrict__ pos_ptr,
    int32_t num_heads,
    int32_t kv_mul,
    int32_t head_dim,
    int32_t kv_dim,
    float sm_scale
) {
    int head_idx = blockIdx.x;
    int tid = threadIdx.x; 
    int kv_head_idx = head_idx / kv_mul;
    int pos = *pos_ptr;
    int step = blockDim.x; 

    // load q to registers
    float q_val = 0.0f; 
    // 如果 tid >= head_dim，该线程不加载 q，但在 reduce 时仍需参与 (值为0)
    if (tid < head_dim) {
        q_val = Q[head_idx * head_dim + tid];
    }
    // 每个线程处理一个元素，thread 0..63 处理数据，thread 64..127 空闲。
    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;
    float out_val = 0.0f; // accumulator for v

    // 循环遍历历史 token
    for (int t = 0; t <= pos; ++t) {
        int kv_offset = t * kv_dim + kv_head_idx * head_dim;
        
        // dot product: q * k
        float score = 0.0f;
        // 仅当线程在 head_dim 范围内时计算
        if (tid < head_dim) {
            score = q_val * K_cache[kv_offset + tid];
        }
        
        // block reduce 而不仅仅是 Warp Reduce
        typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        
        score = BlockReduce(temp_storage).Sum(score);
        // sum 后只有 thread 0 有结果，需要广播
        __shared__ float shared_score;
        if (tid == 0) shared_score = score;
        __syncthreads();
        score = shared_score;

        score *= sm_scale;

        // online softmax
        float old_max = max_score;
        max_score = fmaxf(max_score, score);
        float exp_score = expf(score - max_score); 
        float scale = expf(old_max - max_score);
        
        sum_exp = sum_exp * scale + exp_score;
        out_val *= scale;

        // p * v
        if (tid < head_dim) {
             out_val += exp_score * V_cache[kv_offset + tid];
        }
    }

    // normalize
    float inv_sum = 1.0f / (sum_exp + 1e-12f);
    if (tid < head_dim) {
        O[head_idx * head_dim + tid] = out_val * inv_sum;
    }
}

} // namespace kernel

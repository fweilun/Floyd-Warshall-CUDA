#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;

const int INF = 1073741823;
const int BLOCK_SIZE = 32; 
// 修正 1: 配合 Phase 3 的邏輯，這裡必須是 64
const int EXTEND_BLOCK_SIZE = 64; 

#define cudaCheck(err) (cudaCheckImpl(err, __FILE__, __LINE__))
void cudaCheckImpl(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

// 修正 2: Phase 1 必須處理 64x64 的對角線區塊
__global__ void phase1(int* dist, int V, int Round) {
    // 宣告 64x64 的 shared memory
    __shared__ int smem[EXTEND_BLOCK_SIZE][EXTEND_BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int start_idx = Round * EXTEND_BLOCK_SIZE;
    smem[ty][tx] = dist[(start_idx + ty) * V + (start_idx + tx)];
    smem[ty][tx + BLOCK_SIZE] = dist[(start_idx + ty) * V + (start_idx + tx + BLOCK_SIZE)];
    smem[ty + BLOCK_SIZE][tx] = dist[(start_idx + ty + BLOCK_SIZE) * V + (start_idx + tx)];
    smem[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = dist[(start_idx + ty + BLOCK_SIZE) * V + (start_idx + tx + BLOCK_SIZE)];

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < EXTEND_BLOCK_SIZE; ++k) {
        // 更新 4 個點
        smem[ty][tx] = min(smem[ty][tx], smem[ty][k] + smem[k][tx]);
        smem[ty][tx + BLOCK_SIZE] = min(smem[ty][tx + BLOCK_SIZE], smem[ty][k] + smem[k][tx + BLOCK_SIZE]);
        smem[ty + BLOCK_SIZE][tx] = min(smem[ty + BLOCK_SIZE][tx], smem[ty + BLOCK_SIZE][k] + smem[k][tx]);
        smem[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = min(smem[ty + BLOCK_SIZE][tx + BLOCK_SIZE], smem[ty + BLOCK_SIZE][k] + smem[k][tx + BLOCK_SIZE]);
        
        __syncthreads();
    }

    // 寫回 Global Memory
    dist[(start_idx + ty) * V + (start_idx + tx)] = smem[ty][tx];
    dist[(start_idx + ty) * V + (start_idx + tx + BLOCK_SIZE)] = smem[ty][tx + BLOCK_SIZE];
    dist[(start_idx + ty + BLOCK_SIZE) * V + (start_idx + tx)] = smem[ty + BLOCK_SIZE][tx];
    dist[(start_idx + ty + BLOCK_SIZE) * V + (start_idx + tx + BLOCK_SIZE)] = smem[ty + BLOCK_SIZE][tx + BLOCK_SIZE];
}

// 修正 3: Phase 2 必須處理 64x64 的 Pivot Block 和 Target Block
__global__ void phase2(int* dist, int V, int Round) {
    if (blockIdx.x == Round) return; 

    __shared__ int pivot[EXTEND_BLOCK_SIZE][EXTEND_BLOCK_SIZE];
    __shared__ int target[EXTEND_BLOCK_SIZE][EXTEND_BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int pivot_base = Round * EXTEND_BLOCK_SIZE;
    int target_base_x, target_base_y;

    // 判斷是處理 Row 還是 Col
    if (blockIdx.y == 0) {
        target_base_y = Round * EXTEND_BLOCK_SIZE;
        target_base_x = blockIdx.x * EXTEND_BLOCK_SIZE;
    } else {
        target_base_y = blockIdx.x * EXTEND_BLOCK_SIZE;
        target_base_x = Round * EXTEND_BLOCK_SIZE;
    }

    // --- Load Pivot ---
    pivot[ty][tx] = dist[(pivot_base + ty) * V + (pivot_base + tx)];
    pivot[ty][tx + BLOCK_SIZE] = dist[(pivot_base + ty) * V + (pivot_base + tx + BLOCK_SIZE)];
    pivot[ty + BLOCK_SIZE][tx] = dist[(pivot_base + ty + BLOCK_SIZE) * V + (pivot_base + tx)];
    pivot[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = dist[(pivot_base + ty + BLOCK_SIZE) * V + (pivot_base + tx + BLOCK_SIZE)];

    // --- Load Target ---
    target[ty][tx] = dist[(target_base_y + ty) * V + (target_base_x + tx)];
    target[ty][tx + BLOCK_SIZE] = dist[(target_base_y + ty) * V + (target_base_x + tx + BLOCK_SIZE)];
    target[ty + BLOCK_SIZE][tx] = dist[(target_base_y + ty + BLOCK_SIZE) * V + (target_base_x + tx)];
    target[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = dist[(target_base_y + ty + BLOCK_SIZE) * V + (target_base_x + tx + BLOCK_SIZE)];
    
    __syncthreads();

    // 計算 64 次迭代
    #pragma unroll
    for (int k = 0; k < EXTEND_BLOCK_SIZE; ++k) {
        int p_row, p_col;

        if (blockIdx.y == 0) {
            int p_val = pivot[ty][k];
            int p_val_b = pivot[ty + BLOCK_SIZE][k];

            target[ty][tx] = min(target[ty][tx], p_val + target[k][tx]);
            target[ty][tx + BLOCK_SIZE] = min(target[ty][tx + BLOCK_SIZE], p_val + target[k][tx + BLOCK_SIZE]);
            target[ty + BLOCK_SIZE][tx] = min(target[ty + BLOCK_SIZE][tx], p_val_b + target[k][tx]);
            target[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = min(target[ty + BLOCK_SIZE][tx + BLOCK_SIZE], p_val_b + target[k][tx + BLOCK_SIZE]);

        } else {
            int p_val = pivot[k][tx];
            int p_val_r = pivot[k][tx + BLOCK_SIZE];

            target[ty][tx] = min(target[ty][tx], target[ty][k] + p_val);
            target[ty][tx + BLOCK_SIZE] = min(target[ty][tx + BLOCK_SIZE], target[ty][k] + p_val_r);
            target[ty + BLOCK_SIZE][tx] = min(target[ty + BLOCK_SIZE][tx], target[ty + BLOCK_SIZE][k] + p_val);
            target[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = min(target[ty + BLOCK_SIZE][tx + BLOCK_SIZE], target[ty + BLOCK_SIZE][k] + p_val_r);
        }
    }

    // 寫回 Global Memory
    dist[(target_base_y + ty) * V + (target_base_x + tx)] = target[ty][tx];
    dist[(target_base_y + ty) * V + (target_base_x + tx + BLOCK_SIZE)] = target[ty][tx + BLOCK_SIZE];
    dist[(target_base_y + ty + BLOCK_SIZE) * V + (target_base_x + tx)] = target[ty + BLOCK_SIZE][tx];
    dist[(target_base_y + ty + BLOCK_SIZE) * V + (target_base_x + tx + BLOCK_SIZE)] = target[ty + BLOCK_SIZE][tx + BLOCK_SIZE];
}

__global__ void phase3(int* dist, int V, int Round) {
    if (blockIdx.x == Round || blockIdx.y == Round) return;

    // 這裡使用 EXTEND_BLOCK_SIZE = 64
    const int sm_width = EXTEND_BLOCK_SIZE;
    
    __shared__ int row_block[EXTEND_BLOCK_SIZE * EXTEND_BLOCK_SIZE];
    __shared__ int col_block[EXTEND_BLOCK_SIZE * EXTEND_BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i_start = blockIdx.y * EXTEND_BLOCK_SIZE;
    int j_start = blockIdx.x * EXTEND_BLOCK_SIZE;
    
    int col_strip_y = i_start;
    int col_strip_x = Round * EXTEND_BLOCK_SIZE;
    int row_strip_y = Round * EXTEND_BLOCK_SIZE;
    int row_strip_x = j_start;

    // 1. 左上
    col_block[ty * sm_width + tx] = dist[(col_strip_y + ty) * V + (col_strip_x + tx)];
    row_block[ty * sm_width + tx] = dist[(row_strip_y + ty) * V + (row_strip_x + tx)];
    // 2. 右上
    col_block[ty * sm_width + (tx + BLOCK_SIZE)] = dist[(col_strip_y + ty) * V + (col_strip_x + tx + BLOCK_SIZE)];
    row_block[ty * sm_width + (tx + BLOCK_SIZE)] = dist[(row_strip_y + ty) * V + (row_strip_x + tx + BLOCK_SIZE)];
    // 3. 左下
    col_block[(ty + BLOCK_SIZE) * sm_width + tx] = dist[(col_strip_y + ty + BLOCK_SIZE) * V + (col_strip_x + tx)];
    row_block[(ty + BLOCK_SIZE) * sm_width + tx] = dist[(row_strip_y + ty + BLOCK_SIZE) * V + (row_strip_x + tx)];
    // 4. 右下
    col_block[(ty + BLOCK_SIZE) * sm_width + (tx + BLOCK_SIZE)] = dist[(col_strip_y + ty + BLOCK_SIZE) * V + (col_strip_x + tx + BLOCK_SIZE)];
    row_block[(ty + BLOCK_SIZE) * sm_width + (tx + BLOCK_SIZE)] = dist[(row_strip_y + ty + BLOCK_SIZE) * V + (row_strip_x + tx + BLOCK_SIZE)];

    __syncthreads();
    
    int v1 = dist[(i_start + ty) * V + (j_start + tx)];
    int v2 = dist[(i_start + ty) * V + (j_start + tx + BLOCK_SIZE)];
    int v3 = dist[(i_start + ty + BLOCK_SIZE) * V + (j_start + tx)];
    int v4 = dist[(i_start + ty + BLOCK_SIZE) * V + (j_start + tx + BLOCK_SIZE)];
    
    #pragma unroll
    for (int k = 0; k < EXTEND_BLOCK_SIZE; ++k) {
        int col_val_top = col_block[ty * sm_width + k];
        int col_val_bot = col_block[(ty + BLOCK_SIZE) * sm_width + k];
        
        int row_val_left = row_block[k * sm_width + tx];
        int row_val_right = row_block[k * sm_width + (tx + BLOCK_SIZE)];

        v1 = min(v1, col_val_top + row_val_left);
        v2 = min(v2, col_val_top + row_val_right);
        v3 = min(v3, col_val_bot + row_val_left);
        v4 = min(v4, col_val_bot + row_val_right);
    }
    
    dist[(i_start + ty) * V + (j_start + tx)] = v1;
    dist[(i_start + ty) * V + (j_start + tx + BLOCK_SIZE)] = v2;
    dist[(i_start + ty + BLOCK_SIZE) * V + (j_start + tx)] = v3;
    dist[(i_start + ty + BLOCK_SIZE) * V + (j_start + tx + BLOCK_SIZE)] = v4;
}

int main(int argc, char *argv[]) {
    FILE* file = fopen(argv[1], "r");
    if (!file) return 1;
    int V, E;
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);

    // Padding 到 64 的倍數
    int padded_V = ((V + EXTEND_BLOCK_SIZE - 1) / EXTEND_BLOCK_SIZE) * EXTEND_BLOCK_SIZE;
    size_t size = sizeof(int) * padded_V * padded_V;
    int *h_dist;
    cudaCheck(cudaMallocHost(&h_dist, size));

    for (int i = 0; i < padded_V; ++i) {
        for (int j = 0; j < padded_V; ++j) {
            h_dist[i * padded_V + j] = (i == j) ? 0 : INF;
        }
    }

    vector<int> edge_buffer(E * 3);
    fread(edge_buffer.data(), sizeof(int), E * 3, file) != (size_t)(E * 3);
    fclose(file);

    for (int i = 0; i < E; ++i) {
        int u = edge_buffer[i*3 + 0];
        int v = edge_buffer[i*3 + 1];
        int w = edge_buffer[i*3 + 2];
        h_dist[u * padded_V + v] = w;
    }

    int *d_dist;
    cudaCheck(cudaMalloc(&d_dist, size));
    cudaCheck(cudaMemcpy(d_dist, h_dist, size, cudaMemcpyHostToDevice));

    int rounds = padded_V / EXTEND_BLOCK_SIZE; 
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    for (int r = 0; r < rounds; ++r) {
        phase1<<<1, dimBlock>>>(d_dist, padded_V, r);
        
        dim3 dimGrid2(rounds, 2); 
        phase2<<<dimGrid2, dimBlock>>>(d_dist, padded_V, r);
        
        dim3 dimGrid3(rounds, rounds);
        phase3<<<dimGrid3, dimBlock>>>(d_dist, padded_V, r);
    }
    
    cudaCheck(cudaMemcpy(h_dist, d_dist, size, cudaMemcpyDeviceToHost));
    ofstream out(argv[2], ios::binary);
    for (int i = 0; i < V; ++i) {
        out.write((char*)&h_dist[i * padded_V], sizeof(int) * V);
    }
    out.close();
    
    cudaFreeHost(h_dist);
    cudaFree(d_dist);
    return 0;
}
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;

const int INF = 1073741823;
const int BLOCK_SIZE = 32; // GPU 適合 32x32 的 block

#define cudaCheck(err) (cudaCheckImpl(err, __FILE__, __LINE__))
void cudaCheckImpl(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

__global__ void phase1(int* dist, int V, int Round) {
    __shared__ int smem[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int global_x = Round * BLOCK_SIZE + tx;
    int global_y = Round * BLOCK_SIZE + ty;
    smem[ty][tx] = dist[global_y * V + global_x];
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        int val = smem[ty][k] + smem[k][tx];
        if (val < smem[ty][tx]) {
            smem[ty][tx] = val;
        }
        __syncthreads();
    }

    // 寫回 Global Memory
    dist[global_y * V + global_x] = smem[ty][tx];
}

__global__ void phase2(int* dist, int V, int Round) {
    if (blockIdx.x == Round) return; // 對角線 Block 已經在 Phase 1 處理過

    __shared__ int pivot_block[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int target_block[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int pivot_base_idx = Round * BLOCK_SIZE;
    
    // 定義目標 Block 的位置
    int target_base_x, target_base_y;

    if (blockIdx.y == 0) {
        // 處理 Row Round: (Round, i)
        target_base_y = Round * BLOCK_SIZE;
        target_base_x = blockIdx.x * BLOCK_SIZE;
    } else {
        // 處理 Col Round: (i, Round)
        target_base_y = blockIdx.x * BLOCK_SIZE;
        target_base_x = Round * BLOCK_SIZE;
    }

    // 載入 Pivot Block (Phase 1 已經算好的对角线块)
    pivot_block[ty][tx] = dist[(pivot_base_idx + ty) * V + (pivot_base_idx + tx)];
    // 載入 Target Block (當前要更新的块)
    target_block[ty][tx] = dist[(target_base_y + ty) * V + (target_base_x + tx)];
    
    __syncthreads();
    if (blockIdx.y == 0) {
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            int val = pivot_block[ty][k] + target_block[k][tx];
            if (val < target_block[ty][tx]) {
                target_block[ty][tx] = val;
            }
        }
    } else {
        // Col Strip
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            int val = target_block[ty][k] + pivot_block[k][tx];
            if (val < target_block[ty][tx]) {
                target_block[ty][tx] = val;
            }
        }
    }

    __syncthreads();
    dist[(target_base_y + ty) * V + (target_base_x + tx)] = target_block[ty][tx];
}

__global__ void phase3(int* dist, int V, int Round) {
    // blockIdx.x: Col Id, blockIdx.y: Row Id
    if (blockIdx.x == Round || blockIdx.y == Round) return;

    __shared__ int row_block[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int col_block[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 我們要更新的 Block (i, j)
    int i_start = blockIdx.y * BLOCK_SIZE;
    int j_start = blockIdx.x * BLOCK_SIZE;
    
    // 相依 Block 1: (i, Round) -> 也就是 Col Block (在 Phase 2 算好的)
    int col_strip_y = i_start;
    int col_strip_x = Round * BLOCK_SIZE;

    // 相依 Block 2: (Round, j) -> 也就是 Row Block (在 Phase 2 算好的)
    int row_strip_y = Round * BLOCK_SIZE;
    int row_strip_x = j_start;

    // 載入相依數據到 Shared Memory
    // Col Block: (i, Round)
    col_block[ty][tx] = dist[(col_strip_y + ty) * V + (col_strip_x + tx)];
    // Row Block: (Round, j)
    row_block[ty][tx] = dist[(row_strip_y + ty) * V + (row_strip_x + tx)];

    __syncthreads();
    
    int current_val = dist[(i_start + ty) * V + (j_start + tx)];
    
    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        int val = col_block[ty][k] + row_block[k][tx];
        if (val < current_val) {
            current_val = val;
        }
    }
    
    dist[(i_start + ty) * V + (j_start + tx)] = current_val;
}


int main(int argc, char *argv[]) {

    FILE* file = fopen(argv[1], "r");
    if (!file) {
        fprintf(stderr, "fopen failed: %s\n", argv[1]);
        return 1;
    }

    int V, E;
    if (fread(&V, sizeof(int), 1, file) != 1) return 1;
    if (fread(&E, sizeof(int), 1, file) != 1) return 1;
    
    printf("V: %d, E: %d\n", V, E);

    // Padding V to be multiple of BLOCK_SIZE
    int padded_V = ((V + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    size_t size = sizeof(int) * padded_V * padded_V;
    
    // Allocate Host Memory (Pinned memory is faster for transfer)
    int *h_dist;
    cudaCheck(cudaMallocHost(&h_dist, size));

    for (int i = 0; i < padded_V; ++i) {
        for (int j = 0; j < padded_V; ++j) {
            if (i == j) h_dist[i * padded_V + j] = 0;
            else h_dist[i * padded_V + j] = INF;
        }
    }

    // Read Edges
    // 雖然一次讀全部比較快，但為了處理 Padding，我們需要讀進緩衝區再填入
    vector<int> edge_buffer(E * 3);
    fread(&edge_buffer[0], sizeof(int), E * 3, file) != (size_t)(E * 3);
    fclose(file);

    for (int i = 0; i < E; ++i) {
        int u = edge_buffer[i*3 + 0];
        int v = edge_buffer[i*3 + 1];
        int w = edge_buffer[i*3 + 2];
        // 寫入 Padded Matrix
        h_dist[u * padded_V + v] = w;
    }

    // Allocate Device Memory
    int *d_dist;
    cudaCheck(cudaMalloc(&d_dist, size));

    // Copy Host to Device
    cudaCheck(cudaMemcpy(d_dist, h_dist, size, cudaMemcpyHostToDevice));

    // Launch Kernels
    int rounds = padded_V / BLOCK_SIZE;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    for (int r = 0; r < rounds; ++r) {
        // Phase 1
        phase1<<<1, dimBlock>>>(d_dist, padded_V, r);
        
        // Phase 2
        dim3 dimGrid2(rounds, 2); 
        phase2<<<dimGrid2, dimBlock>>>(d_dist, padded_V, r);
        
        // Phase 3
        dim3 dimGrid3(rounds, rounds);
        phase3<<<dimGrid3, dimBlock>>>(d_dist, padded_V, r);
    }
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());

    // Copy back to Host
    cudaCheck(cudaMemcpy(h_dist, d_dist, size, cudaMemcpyDeviceToHost));
    ofstream out(argv[2], ios::binary);
    if (V == padded_V) {
        out.write((char*)h_dist, sizeof(int) * V * V);
    } else {
        for (int i = 0; i < V; ++i) {
            out.write((char*)&h_dist[i * padded_V], sizeof(int) * V);
        }
    }
    
    out.close();
    cudaFreeHost(h_dist);
    cudaFree(d_dist);

    return 0;
}
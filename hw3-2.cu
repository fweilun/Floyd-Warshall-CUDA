#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// 設定常數
const int INF = 1073741823;
const int BLOCK_SIZE = 64; // 與 CPU 版本的 B 保持一致，適合 GPU Shared Memory

// CUDA Error Check Helper
#define cudaCheck(err) (cudaCheckError(err, __FILE__, __LINE__))
void cudaCheckError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

// 輔助函數：計算向上取整
inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}


__global__ void floyd_warshall_block_kernel(int* dist, int V_padded, int Round, int mode) {
    int bx, by;

    if (mode == 0) {
        bx = Round;
        by = Round;
    } else if (mode == 1) {
        int idx = blockIdx.x; 
        if (blockIdx.y == 0) { 
            bx = (idx < Round) ? idx : idx + 1;
            by = Round;
        } else {
            bx = Round;
            by = (idx < Round) ? idx : idx + 1;
        }
    } else {
        // Phase 3: 處理所有剩餘 Block (by, bx)
        bx = blockIdx.x;
        by = blockIdx.y;
        if (bx == Round || by == Round) return; 
    }

    __shared__ int sm_a[64][64];
    __shared__ int sm_b[64][64];

    // 執行緒 ID
    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15

    // 計算當前 Block 在 Global Memory 的起始偏移量
    size_t block_start_idx = (size_t)by * 64 * V_padded + bx * 64;
    size_t pivot_col_start = (size_t)by * 64 * V_padded + Round * 64;
    size_t pivot_row_start = (size_t)Round * 64 * V_padded + bx * 64;

    // ---------------------------------------------------
    // 1. 載入資料到 Shared Memory
    // ---------------------------------------------------
    // 每個執行緒負責載入一部分數據。
    // Block 大小 64x64 = 4096 ints. Threads = 256. 每個執行緒載入 16 ints (4 個 int4)
    
    // 為了利用 int4，我們將指針轉型
    int4* gl_ptr_a = (int4*)(dist + pivot_col_start);
    int4* gl_ptr_b = (int4*)(dist + pivot_row_start);
    int tid = ty * 16 + tx;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = ty + i * 16;
        // Load Pivot Col Block -> sm_a
        int4 vec_a = gl_ptr_a[row * (V_padded / 4) + tx];
        sm_a[row][tx * 4 + 0] = vec_a.x;
        sm_a[row][tx * 4 + 1] = vec_a.y;
        sm_a[row][tx * 4 + 2] = vec_a.z;
        sm_a[row][tx * 4 + 3] = vec_a.w;
        // Load Pivot Row Block -> sm_b
        int4 vec_b = gl_ptr_b[row * (V_padded / 4) + tx];
        sm_b[row][tx * 4 + 0] = vec_b.x;
        sm_b[row][tx * 4 + 1] = vec_b.y;
        sm_b[row][tx * 4 + 2] = vec_b.z;
        sm_b[row][tx * 4 + 3] = vec_b.w;
    }

    __syncthreads();
    
    int4* gl_ptr_c = (int4*)(dist + block_start_idx);
    int my_dist[4][4]; // Registers: [row_offset][col_offset]

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = ty * 4 + i; // 每個 thread 負責連續的 4 行，以配合 int4 寫入? 
        
        row = ty + i * 16; 
        
        int4 vec_c = gl_ptr_c[row * (V_padded / 4) + tx];
        my_dist[i][0] = vec_c.x;
        my_dist[i][1] = vec_c.y;
        my_dist[i][2] = vec_c.z;
        my_dist[i][3] = vec_c.w;
    }

    if (mode == 0) {
        for (int k = 0; k < 64; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int row = ty + i * 16;
                int dik = sm_a[row][k]; // 廣播
                
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int col = tx * 4 + j;
                    int dkj = sm_a[k][col];
                    int sum = dik + dkj;
                    if (sum < sm_a[row][col]) {
                        sm_a[row][col] = sum;
                    }
                }
            }
            __syncthreads();
        }
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int row = ty + i * 16;
            int4 vec_out;
            vec_out.x = sm_a[row][tx * 4 + 0];
            vec_out.y = sm_a[row][tx * 4 + 1];
            vec_out.z = sm_a[row][tx * 4 + 2];
            vec_out.w = sm_a[row][tx * 4 + 3];
            gl_ptr_c[row * (V_padded / 4) + tx] = vec_out;
        }

    } else {
        for (int k = 0; k < 64; ++k) {
            int d_row_k[4]; // sm_b[k][tx*4 + 0..3]
            d_row_k[0] = sm_b[k][tx * 4 + 0];
            d_row_k[1] = sm_b[k][tx * 4 + 1];
            d_row_k[2] = sm_b[k][tx * 4 + 2];
            d_row_k[3] = sm_b[k][tx * 4 + 3];

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int row = ty + i * 16;
                int d_col_k = sm_a[row][k]; // sm_a[row][k]
                
                // 更新 4 個列
                /* Vectorized add & min is tricky in pure int, so standard logic */
                // Logic: my_dist[i][j] = min(my_dist[i][j], d_col_k + d_row_k[j])
                
                // 為了性能，手動展開
                int sum;
                
                sum = d_col_k + d_row_k[0];
                if (sum < my_dist[i][0]) my_dist[i][0] = sum;

                sum = d_col_k + d_row_k[1];
                if (sum < my_dist[i][1]) my_dist[i][1] = sum;

                sum = d_col_k + d_row_k[2];
                if (sum < my_dist[i][2]) my_dist[i][2] = sum;

                sum = d_col_k + d_row_k[3];
                if (sum < my_dist[i][3]) my_dist[i][3] = sum;
            }
        }
        
        // 寫回 Global Memory
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int row = ty + i * 16;
            int4 vec_out;
            vec_out.x = my_dist[i][0];
            vec_out.y = my_dist[i][1];
            vec_out.z = my_dist[i][2];
            vec_out.w = my_dist[i][3];
            gl_ptr_c[row * (V_padded / 4) + tx] = vec_out;
        }
    }
}

// ---------------------------------------------------------------------------
// Main Logic
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    FILE* file = fopen(argv[1], "rb");
    int V, E;
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);
    
    // Padding V to multiple of 64 for simplified kernel boundary checks
    int V_padded = V;
    if (V % BLOCK_SIZE != 0) {
        V_padded = (V / BLOCK_SIZE + 1) * BLOCK_SIZE;
    }
    int* h_dist;
    cudaMallocHost(&h_dist, sizeof(int) * V_padded * V_padded);
    #pragma omp parallel for
    for (int i = 0; i < V_padded; ++i) {
        for (int j = 0; j < V_padded; ++j) {
            if (i == j) h_dist[i * V_padded + j] = 0;
            else h_dist[i * V_padded + j] = INF;
        }
    }

    // Read Edges
    vector<int> edge_buffer(E * 3);
    fread(edge_buffer.data(), sizeof(int), E * 3, file);
    fclose(file);

    for (int i = 0; i < E; ++i) {
        int u = edge_buffer[i*3 + 0];
        int v = edge_buffer[i*3 + 1];
        int w = edge_buffer[i*3 + 2];
        h_dist[u * V_padded + v] = w;
    }

    // Device Memory Allocation
    int* d_dist;
    cudaMalloc(&d_dist, sizeof(int) * V_padded * V_padded);
    cudaMemcpy(d_dist, h_dist, sizeof(int) * V_padded * V_padded, cudaMemcpyHostToDevice);

    int rounds = V_padded / BLOCK_SIZE;
    dim3 threads(16, 16); 

    for (int r = 0; r < rounds; ++r) {
        // Phase 1: Diagonal Block
        floyd_warshall_block_kernel<<<1, threads>>>(d_dist, V_padded, r, 0);
        
        // Phase 2: Pivot Row & Col Blocks
        if (rounds > 1) {
            dim3 grid2(rounds - 1, 2);
            floyd_warshall_block_kernel<<<grid2, threads>>>(d_dist, V_padded, r, 1);
        }

        // Phase 3: Remaining Blocks
        dim3 grid3(rounds, rounds);
        floyd_warshall_block_kernel<<<grid3, threads>>>(d_dist, V_padded, r, 2);
    }

    // Copy back
    cudaMemcpy(h_dist, d_dist, sizeof(int) * V_padded * V_padded, cudaMemcpyDeviceToHost);
    FILE* outfile = fopen(argv[2], "wb");
    if (V == V_padded) {
        fwrite(h_dist, sizeof(int), V * V, outfile);
    } else {
        for (int i = 0; i < V; ++i) {
            fwrite(&h_dist[i * V_padded], sizeof(int), V, outfile);
        }
    }
    fclose(outfile);
    cudaFreeHost(h_dist);
    cudaFree(d_dist);

    return 0;
}
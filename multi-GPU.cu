#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <chrono>

#define EXTEND_BLOCK_SIZE 64
#define BLOCK_SIZE 32       
using namespace std;


std::chrono::steady_clock::time_point total_start, total_end;

const int INF = ((1 << 30) - 1);
void input(char* infile);
void output(char *outFileName);

void block_FW();
int ceil_div(int a, int b);

int N_orig, V, m;
int* h_dist = NULL;

int ceil_div(int a, int b) { 
    return (a + b - 1) / b; 
}

__global__ void phase1(int* dist, int V, int Round) {
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

__global__ void phase2(int* dist, int Round, int V) {
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

// 一組的threads對應到一個 dist[x][y]
// 會抓到 dist[x][k], dist[k][y] 更新 dist[x][y]。 k 會是那個round 的pivot block index 這樣對嗎
__global__ void phase3(int* dist, int Round, int V, int yoffset) {
    if (blockIdx.x == Round || (blockIdx.y + yoffset) == Round) return;

    const int sm_width = EXTEND_BLOCK_SIZE;
    
    __shared__ int row_block[EXTEND_BLOCK_SIZE * EXTEND_BLOCK_SIZE];
    __shared__ int col_block[EXTEND_BLOCK_SIZE * EXTEND_BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i_start = (blockIdx.y + yoffset) * EXTEND_BLOCK_SIZE;
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
        // 優化提示：這裡沒有 Bank Conflict，col_block 是 Broadcast，row_block 是無衝突存取。
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

void block_FW() {
    const int num_threads = 2;
    int* d_dist_array[num_threads];
    int rounds = ceil_div(V, EXTEND_BLOCK_SIZE);

    cudaError_t err = cudaHostRegister(h_dist, V*V*sizeof(int), cudaHostRegisterDefault);

    #pragma omp parallel num_threads(num_threads)
    {
        const int cpu_threadID = omp_get_thread_num();
        cudaSetDevice(cpu_threadID);
        cudaMalloc(&d_dist_array[cpu_threadID], V*V*sizeof(int));

        const int blocks = V / EXTEND_BLOCK_SIZE;
        int offset = EXTEND_BLOCK_SIZE*V;

        int rounds_per_thread = rounds / 2;
        const int yoffset_blocks = cpu_threadID == 0 ? 0 : rounds_per_thread;

        if(cpu_threadID == 1 && (rounds % 2) == 1) 
            rounds_per_thread += 1;

        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid_phase3(rounds, rounds_per_thread);

        const size_t row_size = offset*sizeof(int);
        const size_t thread_area_size = row_size * rounds_per_thread;
        const size_t yOffset_size_bytes = yoffset_blocks * offset * sizeof(int);
        
        cudaMemcpy(d_dist_array[cpu_threadID] + yoffset_blocks * V * EXTEND_BLOCK_SIZE, 
                   h_dist + yoffset_blocks * V * EXTEND_BLOCK_SIZE, 
                   thread_area_size, cudaMemcpyHostToDevice);

        for(int r = 0; r < rounds; r++) {
            const size_t pivot_block_offset = r * EXTEND_BLOCK_SIZE * V;
            const int isInSelfRange = (r >= yoffset_blocks) && (r < (yoffset_blocks + rounds_per_thread));
            // pivot block 的那列
            if (isInSelfRange) {
                cudaMemcpy(h_dist + pivot_block_offset, 
                           d_dist_array[cpu_threadID] + r * EXTEND_BLOCK_SIZE * V, 
                           row_size, cudaMemcpyDeviceToHost);
            }

            #pragma omp barrier
            cudaMemcpy(d_dist_array[cpu_threadID] + r * EXTEND_BLOCK_SIZE * V, 
                       h_dist + pivot_block_offset, 
                       row_size, cudaMemcpyHostToDevice);
            
            phase1 <<<1, dimBlock>>>(d_dist_array[cpu_threadID], r, V);

            dim3 dimGrid_phase2(rounds, 2); 
            phase2 <<<dimGrid_phase2, dimBlock>>>(d_dist_array[cpu_threadID], r, V);
            
            phase3 <<<dimGrid_phase3, dimBlock>>>(d_dist_array[cpu_threadID], r, V, yoffset_blocks);
        }

        cudaMemcpy(h_dist + yoffset_blocks * V * EXTEND_BLOCK_SIZE, 
                   d_dist_array[cpu_threadID] + yoffset_blocks * V * EXTEND_BLOCK_SIZE, 
                   thread_area_size, cudaMemcpyDeviceToHost);
        
        #pragma omp barrier
        
        cudaFree(d_dist_array[cpu_threadID]);
    }
    
    cudaHostUnregister(h_dist);
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    if (!file) {
        perror("Error opening input file");
        exit(EXIT_FAILURE);
    }
    fread(&N_orig, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    V = N_orig;
    if (V % EXTEND_BLOCK_SIZE != 0) {
        V = N_orig + (EXTEND_BLOCK_SIZE - (N_orig % EXTEND_BLOCK_SIZE));
    }

    // Allocate Host Memory (using h_dist)
    h_dist = (int*) malloc(sizeof(int)*V*V);
    if (!h_dist) {
        perror("Error allocating host memory");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < V; ++ i) {
        for (int j = 0; j < V; ++j) {
            h_dist[i * V + j] = (i == j) ? 0 : INF;
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) { 
        fread(pair, sizeof(int), 3, file); 
        h_dist[pair[0] * V + pair[1]] = pair[2]; 
    } 
    fclose(file);
}

void output(char *outFileName) {
    FILE *outfile = fopen(outFileName, "wb");
    for (int i = 0; i < N_orig; ++i) {
        fwrite(&h_dist[i * V], sizeof(int), N_orig, outfile);
    }
    fclose(outfile);
}

int main(int argc, char *argv[]){
    total_start = std::chrono::steady_clock::now();
    input(argv[1]);
    block_FW();
    output(argv[2]);
    cudaFreeHost(h_dist); // 使用 cudaFreeHost 清理 h_dist
    total_end = std::chrono::steady_clock::now();
    std::cout << "[TOTAL_TIME] " << std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count() << "\n";
    return 0;
}
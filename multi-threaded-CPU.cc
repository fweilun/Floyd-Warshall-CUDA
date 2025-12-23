#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <immintrin.h> // SSE
#include <omp.h>       // OpenMP
#include <cstdlib>     // aligned_alloc, free

using namespace std;

const int INF = 1073741823; // 約 1e9，確保 INF + INF 不會溢出 int
int V, E;
int *dist;

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

void update_block(int B, int bi, int bj, int bk) {
    int i_start = bi * B;
    int j_start = bj * B;
    int k_start = bk * B;
    int i_end = min(i_start + B, V);
    int j_end = min(j_start + B, V);
    int k_end = min(k_start + B, V);

    for (int k = k_start; k < k_end; k++) {
        for (int i = i_start; i < i_end; i++) {
            int* row_i = &dist[i * V];
            int dik = dist[i * V + k];
            if (dik == INF) continue;
            int* row_k = &dist[k * V];
            __m128i dik_vec = _mm_set1_epi32(dik);
            int j = j_start;
            for (; j <= j_end - 4; j += 4) {
                __m128i d_ij = _mm_loadu_si128((__m128i*)&row_i[j]);
                __m128i d_kj = _mm_loadu_si128((__m128i*)&row_k[j]);
                __m128i new_dist = _mm_add_epi32(dik_vec, d_kj);
                __m128i min_dist = _mm_min_epi32(d_ij, new_dist);
                _mm_storeu_si128((__m128i*)&row_i[j], min_dist);
            }
            for (; j < j_end; j++) {                
                int val_kj = row_k[j];
                if (val_kj != INF) {
                    int new_dist = dik + val_kj;
                    if (new_dist < row_i[j]) {
                        row_i[j] = new_dist;
                    }
                }
            }
        }
    }
}

void block_FW(int B) {
    int rounds = ceil_div(V, B);
    for (int r = 0; r < rounds; r++) {
        update_block(B, r, r, r);
        #pragma omp parallel
        {
            #pragma omp for nowait
            for (int j = 0; j < rounds; j++) {
                if (j == r) continue;
                update_block(B, r, j, r);
            }
            #pragma omp for nowait
            for (int i = 0; i < rounds; i++) {
                if (i == r) continue;
                update_block(B, i, r, r);
            }
        }
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < rounds; i++) {
            for (int j = 0; j < rounds; j++) {
                if (i != r && j != r) {
                    update_block(B, i, j, r);
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }
    // ios::sync_with_stdio(false);
    FILE* file = fopen(argv[1], "r");
    if (!file) {
        fprintf(stderr, "fopen failed: %s\n", argv[1]);
        return 1;
    }
    printf("%s\n", argv[1]);
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);
    printf("%d %d\n", V, E);

    size_t size = sizeof(int) * V * V;
    dist = (int*)aligned_alloc(32, size);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            dist[i * V + j] = (i == j ? 0 : INF);
        }
    }
    vector<int> edge_buffer(E * 3);
    fread(&edge_buffer[0], sizeof(int), E * 3, file);
    for (int i = 0; i < E; ++i) {
        int u = edge_buffer[i*3 + 0];
        int v = edge_buffer[i*3 + 1];
        int w = edge_buffer[i*3 + 2];
        dist[u * V + v] = w;
    }
    

    int B = 64; 
    block_FW(B);

    ofstream out(argv[2], ios::binary);
    out.write((char*)dist, size);
    out.close();

    free(dist);
    return 0;
}
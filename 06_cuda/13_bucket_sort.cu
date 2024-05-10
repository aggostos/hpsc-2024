#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void sort(int *key, int *bucket, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  atomicAdd(&bucket[key[i]],1);
  int offset = 0;
  __syncthreads();
  for (int j=0; j<range; j++) {
    if (i < bucket[j] + offset && i >= offset) {
      key[i] = j;
    }
    offset += bucket[j];
  }
}	

int main() {
  int n = 50;
  int range = 5;
  int *key, *bucket;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  sort<<<1,n>>>(key, bucket, range);  
  cudaDeviceSynchronize();
  
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  
  printf("\n");
  cudaFree(key);
  cudaFree(bucket);
}

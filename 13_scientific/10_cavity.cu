#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

const int nx = 41;
const int ny = 41;
const int nt = 500;
const int nit = 50;
const float dx = 2.0 / (nx - 1);
const float dy = 2.0 / (ny - 1);
const float dt = 0.01;
const float rho = 1.0;
const float nu = 0.02;
const int N = ny * nx;
const int M = 1024;

__global__ void initialization(float *u, float *v, float *p, float *b) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= N) return;
  u[id] = 0.0;
  v[id] = 0.0;
  p[id] = 0.0;
  b[id] = 0.0;
}

__global__ void compute_b(float *b, float *u, float *v) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int j = id / nx;
  int i = id % nx;
  if (i > 0 && j > 0 && i < nx - 1 && j < ny - 1) {
    b[id] = rho * (1.0 / dt *\
      ((u[id+1] - u[id-1]) / (2.0*dx) + (v[(j+1)*nx + i] - v[(j-1)*nx + i]) / (2.0*dy)) -\
      pow((u[id+1] - u[id-1]) / (2.0*dx), 2) - 2.0 *\
      ((u[(j+1)*nx + i] - u[(j-1)*nx + i]) / (2.0*dy) * (v[id+1] - v[id-1]) / (2*dx)) -\
      pow((v[(j+1)*nx + i] - v[(j-1)*nx + i]) / (2.0*dy), 2));
  }
}

__global__ void update_pn(float *p, float *pn) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= N) return;
  pn[id] = p[id];
}

__global__ void compute_p(float *p, float *pn, float *b) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int j = id / nx;
  int i = id % nx;
  if (i > 0 && j > 0 && i < nx - 1 && j < ny - 1) {
    p[id] = (pow(dy,2) * (pn[id+ 1] + pn[id-1]) +\
            pow(dx,2) * (pn[(j+1)*nx + i] + pn[(j-1)*nx + i]) -\
            b[id] * pow(dx,2) * pow(dy,2)) /\
            (2.0 * (pow(dx,2) + pow(dy,2)));
  }
}

__global__ void set_p_boundary(float *p) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= ny) return;

  p[id*nx + (nx-1)] = p[id*nx + (nx-2)];
  p[id*nx] = p[id*nx + 1];
  __syncthreads();
  p[id] = p[nx + id];
  p[(ny-1)*nx + id] = 0.0;
}

__global__ void update_un_vn(float *u, float *un, float *v, float *vn) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= N) return;
  un[id] = u[id];
  vn[id] = v[id];
}

__global__ void compute_u_v(float *u, float *un, float *v, float *vn, float *p) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int j = id / nx;
  int i = id % nx;
  if (i > 0 && j > 0 && i < nx - 1 && j < ny - 1) {
    u[id] = un[id] - un[id] * (dt/dx) * (un[id] - un[id-1])
                          - vn[id] * (dt/dy) * (un[id] - un[(j-1)*nx + i])
                          - dt / (2.0 * rho * dx) * (p[id+1] - p[id-1])
                          + nu * (dt / pow(dx,2)) * (un[id+1] - 2.0 * un[id] + un[id-1])
                          + nu * (dt / pow(dy,2)) * (un[(j+1)*nx + i] - 2.0 * un[id] + un[(j-1)*nx + i]);

    v[id] = vn[id] - un[id] * (dt/dx) * (vn[id] - vn[id-1])
                      - vn[id] * (dt/dy) * (vn[id] - vn[(j-1)*nx + i])
                      - dt / (2.0 * rho * dx) * (p[(j+1)*nx + i] - p[(j-1)*nx + i])
                      + nu * (dt / pow(dx,2)) * (vn[id+1] - 2.0 * vn[id] + vn[id-1])
                      + nu * (dt / pow(dy,2)) * (vn[(j+1)*nx + i] - 2.0 * vn[id] + vn[(j-1)*nx + i]);
  }
}

__global__ void set_u_v_boundary(float *u, float *v) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= ny) return;

  u[id*nx] = 0.0;
  u[id*nx + (nx-1)] = 0.0;
  v[id*nx] = 0.0;
  v[id*nx + (nx-1)] = 0.0;
  __syncthreads();
  u[id] = 0.0;
  u[(ny-1)*nx + id] = 1.0;
  v[id] = 0.0;
  v[(ny-1)*nx + id] = 0.0;
}

int main() {
  float *u, *v, *p, *b, *un, *vn, *pn;
  cudaMallocManaged(&u, N * sizeof(float));
  cudaMallocManaged(&v, N * sizeof(float));
  cudaMallocManaged(&p, N * sizeof(float));
  cudaMallocManaged(&b, N * sizeof(float));
  cudaMallocManaged(&un, N * sizeof(float));
  cudaMallocManaged(&vn, N * sizeof(float));
  cudaMallocManaged(&pn, N * sizeof(float));

  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");

  // Initialize matrices with 0.0
  initialization<<<(N+M-1)/M,M>>>(u, v, p, b);
  cudaDeviceSynchronize();

  for (int n = 0; n < nt; n++) {
    // Compute b[j][i]
    compute_b<<<(N+M-1)/M,M>>>(b, u, v);
    cudaDeviceSynchronize();

    for (int it = 0; it < nit; it++) {
      // Update pn = p
      update_pn<<<(N+M-1)/M,M>>>(p, pn);
      cudaDeviceSynchronize();

      // Compute p[j][i]
      compute_p<<<(N+M-1)/M,M>>>(p, pn, b);
      cudaDeviceSynchronize();

      // Set boundary conditions for p
      set_p_boundary<<<1,ny>>>(p);
      cudaDeviceSynchronize();
    }

    // Update un = u, vn = v
    update_un_vn<<<(N+M-1)/M,M>>>(u, un, v, vn);
    cudaDeviceSynchronize();

    // Compute u[j][i] and v[j][i]
    compute_u_v<<<(N+M-1)/M,M>>>(u, un, v, vn, p);
    cudaDeviceSynchronize();

    // Set boundary conditions for u and v
    set_u_v_boundary<<<1,ny>>>(u, v);
    cudaDeviceSynchronize();

    // Save data to files
    if (n % 10 == 0) {
      for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
          ufile << u[j*nx + i] << " ";
        }
      }
      ufile << "\n";
      for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
          vfile << v[j*nx + i] << " ";
        }
      }
      vfile << "\n";
      for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
          pfile << p[j*nx + i] << " ";
        }
      }
      pfile << "\n";
    }
  }
  ufile.close();
  vfile.close();
  pfile.close();

  cudaFree(u);
  cudaFree(v);
  cudaFree(p);
  cudaFree(b);
  cudaFree(un);
  cudaFree(vn);
  cudaFree(pn);

  return 0;
}

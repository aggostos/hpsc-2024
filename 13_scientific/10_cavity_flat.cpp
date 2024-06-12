#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

int main() {
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

  float *u = (float*)malloc(N * sizeof(float));
  float *v = (float*)malloc(N * sizeof(float));
  float *p = (float*)malloc(N * sizeof(float));
  float *b = (float*)malloc(N * sizeof(float));
  float *un = (float*)malloc(N * sizeof(float));
  float *vn = (float*)malloc(N * sizeof(float));
  float *pn = (float*)malloc(N * sizeof(float));

  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");

  // Initialize matrices with 0.0
  for (int id = 0; id < N; id++) {
    u[id] = 0.0;
    v[id] = 0.0;
    p[id] = 0.0;
    b[id] = 0.0;
  }

  for (int n = 0; n < nt; n++) {
    // Compute b[j][i]
    for (int id = 0; id < N; id++) {
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

    // Update pn = p
    for (int it = 0; it < nit; it++) {
      for (int id = 0; id < N; id++) {
        pn[id] = p[id];
      }

      // Compute p[j][i]
      for (int id = 0; id < N; id++) {
        int j = id / nx;
        int i = id % nx;
        if (i > 0 && j > 0 && i < nx - 1 && j < ny - 1) {
          p[id] = (pow(dy,2) * (pn[id+ 1] + pn[id-1]) +\
                  pow(dx,2) * (pn[(j+1)*nx + i] + pn[(j-1)*nx + i]) -\
                  b[id] * pow(dx,2) * pow(dy,2)) /\
                  (2.0 * (pow(dx,2) + pow(dy,2)));
        }
      }

      // Boundary conditions for p
      for (int j = 0; j < ny; j++) {
        p[j*nx + (nx-1)] = p[j*nx + (nx-2)];
        p[j*nx] = p[j*nx + 1];
      }

      for (int i = 0; i < nx; i++) {
        p[i] = p[nx + i];
        p[(ny-1)*nx + i] = 0.0;
      }
    }

    // Update un = u, vn = v
    for (int id = 0; id < N; id++) {
      un[id] = u[id];
      vn[id] = v[id];
    }

    // Compute u[j][i] and v[j][i]
    for (int id = 0; id < N; id++) {
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

    for (int j = 0; j < ny; j++) {
      u[j*nx] = 0.0;
      u[j*nx + (nx-1)] = 0.0;
      v[j*nx] = 0.0;
      v[j*nx + (nx-1)] = 0.0;
    }

    for (int i = 0; i < nx; i++) {
      u[i] = 0.0;
      u[(ny-1)*nx + i] = 1.0;
      v[i] = 0.0;
      v[(ny-1)*nx + i] = 0.0;
    }

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

  free(u);
  free(v);
  free(p);
  free(b);
  free(un);
  free(vn);
  free(pn);

  return 0;
}

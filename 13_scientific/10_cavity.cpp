#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;
typedef vector< vector<double> > matrix;

int main() {
  const int nx = 41;
  const int ny = 41;
  const int nt = 500;
  const int nit = 50;
  const double dx = 2.0 / (nx - 1);
  const double dy = 2.0 / (ny - 1);
  const double dt = 0.01;
  const double rho = 1.0;
  const double nu = 0.02;

  matrix u(ny,vector<double>(nx));
  matrix v(ny,vector<double>(nx));
  matrix p(ny,vector<double>(nx));
  matrix b(ny,vector<double>(nx));
  matrix un(ny,vector<double>(nx));
  matrix vn(ny,vector<double>(nx));
  matrix pn(ny,vector<double>(nx));

  // Initialize matrices with 0.0
  for (int j=0; j<ny; j++) {
    for (int i=0; i<nx; i++) {
      u[j][i] = 0.0;
      v[j][i] = 0.0;
      p[j][i] = 0.0;
      b[j][i] = 0.0;
    }
  }
  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");
  for (int n=0; n<nt; n++) {
    // Compute b[j][i]
    for (int j=1; j<ny-1; j++) {
      for (int i=1; i<nx-1; i++) {
        b[j][i] = rho * (1.0 / dt *\
          ((u[j][i+1] - u[j][i-1]) / (2.0*dx) + (v[j+1][i] - v[j-1][i]) / (2.0*dy)) -\
          pow((u[j][i+1] - u[j][i-1]) / (2.0*dx), 2) - 2.0 *\
          ((u[j+1][i] - u[j-1][i]) / (2.0*dy) * (v[j][i+1] - v[j][i-1]) / (2*dx)) -\
          pow((v[j+1][i] - v[j-1][i]) / (2.0*dy), 2));
      }
    }
    
    // Update pn = p
    for (int it=0; it<nit; it++) {
      for (int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++) {
          pn[j][i] = p[j][i];
        }
      }
      
      // Compute p[j][i]
      for (int j=1; j<ny-1; j++) {
        for (int i=1; i<nx-1; i++) {
          p[j][i] = (pow(dy,2) * (pn[j][i+1] + pn[j][i-1]) +\
                     pow(dx,2) * (pn[j+1][i] + pn[j-1][i]) -\
                     b[j][i] * pow(dx,2) * pow(dy,2)) /\
                     (2.0 * (pow(dx,2) + pow(dy,2)));
        }
      }
      
      // Boundary conditions for p
      for (int j=0; j<ny; j++) {
        p[j][nx-1] = p[j][nx-2];
        p[j][0] = p[j][1];
      }
      
      for (int i=0; i<nx; i++) {
        p[0][i] = p[1][i];
        p[ny-1][i] = 0.0;
      }
    }
    
    // Update un = u, vn = v
    for (int j=0; j<ny; j++) {
      for (int i=0; i<nx; i++) {
        un[j][i] = u[j][i];
        vn[j][i] = v[j][i];
      }
    }
    
    // Compute u[j][i] and v[j][i]
    for (int j=1; j<ny-1; j++) {
      for (int i=1; i<nx-1; i++) {
        u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i-1])\
                           - vn[j][i] * (dt/dy) * (un[j][i] - un[j-1][i])\
                           - dt / (2.0 * rho * dx) * (p[j][i+1] - p[j][i-1])\
                           + nu * (dt / pow(dx,2)) * (un[j][i+1] - 2.0 * un[j][i] + un[j][i-1])\
                           + nu * (dt / pow(dy,2)) * (un[j+1][i] - 2.0 * un[j][i] + un[j-1][i]);

        v[j][i] = vn[j][i] - un[j][i] * (dt/dx) * (vn[j][i] - vn[j][i-1])\
		                       - vn[j][i] * (dt/dy) * (vn[j][i] - vn[j-1][i])\
                           - dt / (2.0 * rho * dx) * (p[j+1][i] - p[j-1][i])\
                           + nu * (dt / pow(dx,2)) * (vn[j][i+1] - 2.0 * vn[j][i] + vn[j][i-1])\
                           + nu * (dt / pow(dy,2)) * (vn[j+1][i] - 2.0 * vn[j][i] + vn[j-1][i]);
      }
    }

    for (int j=0; j<ny; j++) {
      u[j][0] = 0.0;
      u[j][nx-1] = 0.0;
      v[j][0] = 0.0;
      v[j][nx-1] = 0.0;
    }

    for (int i=0; i<nx; i++) {
      u[0][i] = 0.0;
      u[ny-1][i] = 1.0;
      v[0][i] = 0.0;
      v[ny-1][i] = 0.0;
    }
    
    if (n % 10 == 0) {
      for (int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++) {
          ufile << u[j][i] << " ";
        }
      }
      ufile << "\n";
      for (int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++) {
          vfile << v[j][i] << " ";
        }
      }
      vfile << "\n";
      for (int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++) {
          pfile << p[j][i] << " ";
        }
      }
      pfile << "\n";
    }
  }
  ufile.close();
  vfile.close();
  pfile.close();

  return 0;
}

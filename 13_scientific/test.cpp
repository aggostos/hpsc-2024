#include <chrono>
#include <thread>
#include <cstdio>
#include <vector>
#include <cmath>
//#include "matplotlibcpp.h"

//namespace plt = matplotlibcpp;

//#include "gnuplot-iostream.h"

using namespace std;
typedef vector<vector<double>> matrix;

int main() {
  //Gnuplot gp;
  const int nx = 41;
  const int ny = 41;
  const int nt = 500;
  const int nit = 50;
  const double dx = 2.0 / (nx - 1);
  const double dy = 2.0 / (ny - 1);
  const double dt = 0.01;
  const double rho = 1.0;
  const double nu = 0.02;

  vector<double> x(nx), y(ny);
  for (int i = 0; i < nx; i++) x[i] = i * dx;
  for (int j = 0; j < ny; j++) y[j] = j * dy;


  matrix u(ny,vector<double>(nx, 0.0));
  matrix v(ny,vector<double>(nx, 0.0));
  matrix p(ny,vector<double>(nx, 0.0));
  matrix b(ny,vector<double>(nx, 0.0)); 

  for (int n = 0; n < nt; n++) {
    for (int j = 1; j < ny-1; j++) {
      for (int i = 1; i < nx-1; i++) {
	b[j][i] = rho * (1.0 / dt *\
	    ((u[j][i+1] - u[j][i-1]) / (2*dx) + (v[j+1][i] - v[j-1][i]) / (2*dy)) -\
	    pow(((u[j][i+1] - u[j][i-1]) / (2*dx)),2) - 2*((u[j+1][i] - u[j-1][i]) / (2*dy) *\
	      (v[j][i+1] - v[j][i-1]) / (2*dx)) - pow(((v[j+1][i] - v[j-1][i]) / (2*dy)),2));
      }
    }
    for (int it = 0; it < nit; it++) {
      matrix pn(p);
      for (int j = 1; j < ny-1; j++) {
	for (int i = 1; i < nx-1; i++) {
	  p[j][i] = (pow(dy,2) * (pn[j][i+1] + pn[j][i-1]) +\
	      pow(dx,2) * (pn[j+1][i] + pn[j-1][i]) -\
	      b[j][i] * pow(dx,2) + pow(dy,2))\
		    / (2 * (pow(dx,2) + pow(dy,2)));
	}
      }
      for (int j = 0; j < ny; j++) {
	p[j][nx-1] = p[j][nx-2];
	p[j][0] = p[j][1];
      }
      for (int i = 0; i < nx; i++) {
	p[0][i] = p[1][i];
	p[ny-1][i] = 0.0;
      }
    }

    matrix un(u);
    matrix vn(v);
    for (int j = 1; j < ny-1; j++) {
      for (int i = 1; i < nx-1; i++) {
	u[j][i] = un[j][i] - un[j][i] * (dt/dx) * (un[j][i] - un[j][i-1])\
		  - un[j][i] * (dt/dy) * (un[j][i] - un[j-1][i])\
		  - dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1])\
		  + nu * (dt / pow(dx,2)) * (un[j][i+1] - 2 * un[j][i] + un[j][i-1])\
		  + nu * (dt / pow(dy,2)) * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]);

	v[j][i] = vn[j][i] - vn[j][i] * (dt/dx) * (vn[j][i] - vn[j][i-1])\
		  - vn[j][i] * (dt/dy) * (vn[j][i] - vn[j-1][i])\
		  - dt / (2 * rho * dx) * (p[j+1][i] - p[j-1][i])\
		  + nu * (dt / pow(dx,2)) * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1])\
		  + nu * (dt / pow(dy,2)) * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]);
      }
    }
    for (int j = 0; j < ny; j++) {
      u[j][0] = 0.0;
      u[j][nx-1] = 0.0;
      v[j][0] = 0.0;
      v[j][nx-1] = 0.0;
    }
    for (int i = 0; i < nx; i++) {
      u[0][i] = 0.0;
      u[ny-1][i] = 1.0;
      v[0][i] = 0.0;
      v[ny-1][i] = 0.0;
    }
    //this_thread::sleep_for(chrono::milliseconds(100));
  }

  return 0;
}

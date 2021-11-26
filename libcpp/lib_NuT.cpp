////////////////////////////////////////////////////////////////////////

// g++ -Wall -O2 -fopenmp -shared -Wl,-soname,lib_NuT -o lib_NuT.so -fPIC lib_NuT.cpp

#include <cmath>

using namespace std;

extern "C"
void  NuT (int N_t, int N_nu, int N_fD, double *tt, double *nu, double* fD, double* DS, double *hSS_real, double *hSS_im)
{
	//tell c++ how to read numpy arrays
	#define  DYNSPEC(i_t,i_nu)  DS[(i_t)*N_nu + (i_nu)]
	#define  REAL(i_fD,i_nu)  hSS_real[(i_fD)*N_nu + (i_nu)]
	#define  IMAG(i_fD,i_nu)  hSS_im[(i_fD)*N_nu + (i_nu)]
	
	#pragma omp parallel for
	for (int i_nu = 0; i_nu < N_nu; i_nu++){
		double phase;
		for (int i_t = 0; i_t < N_t; i_t++){
			for (int i_fD = 0; i_fD < N_fD; i_fD++){
				phase = -2.*M_PI*fD[i_fD]*tt[i_t]*nu[i_nu];
				REAL(i_fD,i_nu) += DYNSPEC(i_t,i_nu)*cos(phase);
				IMAG(i_fD,i_nu) += DYNSPEC(i_t,i_nu)*sin(phase);
			}
		}
	}
		
	#undef  DYNSPEC
	#undef  REAL
	#undef  IMAG
}
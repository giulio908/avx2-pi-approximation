// Compile with: g++ -std=c++11 -O3 -mavx2 pi_avx2.cpp -o pi_avx2

#include <iostream>
#include <ctime>
#include <chrono>
#include <iomanip>

#include <immintrin.h>

using namespace std;

// Setting a non-multiple of 4 tests the tail-masking handling
// long long STEPS = (long long)1<<1; 
long long STEPS = (long long)1<<24;

double pi(long long start, long long count, double dx) {
    double sum = 0;
    for(long long i=start; i<(start+count); i++){
        double x = (i+0.5)*dx;
        sum = sum + 4.0/(1+x*x);
    }
    return sum * dx;
}

double pi_avx2(long long start, long long count, double dx) {
    const long long end = (long long)start + count;
    long long i = start;
    
    // Instantiate vector constants of the required types
    const __m256d DX_VETT = _mm256_set1_pd (dx);
    const __m256d CENTRO_VETT = _mm256_set1_pd (0.5);
    const __m256d QUATTRO_VETT = _mm256_set1_pd (4.0);
    const __m256d UNO_VETT = _mm256_set1_pd (1.0);
    
    const __m256d END_VETT = _mm256_set1_pd ((double)end);
    const __m256d SPIAZZ_VETT = _mm256_set_pd (0.0, 1.0, 2.0, 3.0);
    
    // Accumulator vector for the partial sums
    __m256d SUM_VETT = _mm256_set1_pd (0.0);
    
    for(; i<end; i+=4){
        // Process 4 loop iterations simultaneously: current 'i' and the next 3
        __m256d I_VETT = _mm256_add_pd((_mm256_set1_pd (i)), SPIAZZ_VETT);
        
        // Mask vector, opcode 30: OP := _CMP_GT_OQ from Intel Intrinsics
        __m256d MASK_VETT = _mm256_cmp_pd(END_VETT, I_VETT, _CMP_GT_OQ);
        
        // Shift to the midpoint of each sub-interval rectangle
        __m256d I_CENTRO_VETT =  _mm256_add_pd(I_VETT, CENTRO_VETT);
        
        // Compute x values
        __m256d X_VETT = _mm256_mul_pd(I_CENTRO_VETT, DX_VETT);
        
        // Evaluate the integrand: 4.0 / (1.0 + x^2)
        __m256d DIV_VETT = _mm256_div_pd(QUATTRO_VETT , _mm256_add_pd(UNO_VETT, _mm256_mul_pd(X_VETT, X_VETT)));
        
        // Compute unconditionally, then bitwise-AND with the mask to zero out out-of-bounds lanes
        __m256d DIV_MASKED_VETT = _mm256_and_pd(DIV_VETT, MASK_VETT);
        
        // Accumulate the masked results
        SUM_VETT = _mm256_add_pd(SUM_VETT, DIV_MASKED_VETT);
    }
    
    double SUM_APP[4];
    _mm256_storeu_pd(SUM_APP , SUM_VETT);
    
    double sum = SUM_APP[0] + SUM_APP[1] + SUM_APP[2] + SUM_APP[3];
    
    return sum * dx;
}

int main() {

    double dx = 1.0 / STEPS;
    double p;

    std::clock_t cpuT = std::clock();
    std::chrono::steady_clock::time_point wallT = std::chrono::steady_clock::now();
    p = pi ( 0, STEPS, dx);
    std::chrono::duration<double> wallTime = std::chrono::steady_clock::now() - wallT;
    double pT = (std::clock() - cpuT) / (double)CLOCKS_PER_SEC;
    std::cout << "\n pi (base): " << std::setprecision(15) << p;
    std::cout << "\nFinished in " << pT << " seconds [Process CPU time] " << std::endl;
    std::cout << "Finished in " << wallTime.count() << " seconds [Wall time]" << std::endl;

    cpuT = std::clock();
    wallT = std::chrono::steady_clock::now();
    p = pi_avx2 ( 0, STEPS, dx);
    wallTime = std::chrono::steady_clock::now() - wallT;
    pT = (std::clock() - cpuT) / (double)CLOCKS_PER_SEC;
    std::cout << "\n\n pi (avx2): " << std::setprecision(15) << p;
    std::cout << "\nFinished in " << pT << " seconds [Process CPU time] " << std::endl;
    std::cout << "Finished in " << wallTime.count() << " seconds [Wall time]" << std::endl;

    return 0;
}
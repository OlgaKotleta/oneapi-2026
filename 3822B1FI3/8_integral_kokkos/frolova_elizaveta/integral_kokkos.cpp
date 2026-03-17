#include "integral_kokkos.h"
#include <cmath>

float IntegralKokkos(float start, float end, int count) {
    float h = (end - start) / count;
    float result = 0.0f;
    
    using ExecSpace = Kokkos::SYCL;
    
    Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> policy(
        {{0, 0}},
        {{count, count}},
        {{16, 16}}
    );
    
    Kokkos::parallel_reduce("IntegralKokkos", policy,
        KOKKOS_LAMBDA(int i, int j, float& sum) {
            float x_center = start + (i + 0.5f) * h;
            float y_center = start + (j + 0.5f) * h;
            float f_value = Kokkos::sin(x_center) * Kokkos::cos(y_center);
            sum += f_value * h * h;
        },
        result
    );
    
    Kokkos::fence();
    return result;
}
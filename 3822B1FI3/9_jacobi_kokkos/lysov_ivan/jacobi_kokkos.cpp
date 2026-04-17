#include "jacobi_kokkos.h"

#include <algorithm>
#include <cmath>
#include <vector>

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy) {
    const size_t n = b.size();

    if (n == 0) {
        return {};
    }

    if (a.size() != n * n) {
        return {};
    }

    if (accuracy < 0.0f) {
        accuracy = 0.0f;
    }

    using ExecSpace = Kokkos::SYCL;
    using MemSpace = ExecSpace::memory_space;

    Kokkos::View<float*, MemSpace> a_view("a_view", a.size());
    Kokkos::View<float*, MemSpace> b_view("b_view", b.size());
    Kokkos::View<float*, MemSpace> x_old("x_old", n);
    Kokkos::View<float*, MemSpace> x_new("x_new", n);
    Kokkos::View<float*, MemSpace> diff("diff", n);

    auto a_host = Kokkos::create_mirror_view(a_view);
    auto b_host = Kokkos::create_mirror_view(b_view);

    for (size_t i = 0; i < a.size(); ++i) {
        a_host(i) = a[i];
    }
    for (size_t i = 0; i < b.size(); ++i) {
        b_host(i) = b[i];
    }

    Kokkos::deep_copy(a_view, a_host);
    Kokkos::deep_copy(b_view, b_host);
    Kokkos::deep_copy(x_old, 0.0f);
    Kokkos::deep_copy(x_new, 0.0f);
    Kokkos::deep_copy(diff, 0.0f);

    bool converged = false;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        Kokkos::parallel_for(
            "JacobiKokkosStep",
            Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(n)),
            KOKKOS_LAMBDA(const int i) {
                const size_t row_offset = static_cast<size_t>(i) * n;

                float row_sum = 0.0f;
                for (size_t j = 0; j < n; ++j) {
                    if (j != static_cast<size_t>(i)) {
                        row_sum += a_view(row_offset + j) * x_old(j);
                    }
                }

                const float diag = a_view(row_offset + static_cast<size_t>(i));
                const float new_value = (b_view(i) - row_sum) / diag;

                x_new(i) = new_value;
                diff(i) = Kokkos::fabs(new_value - x_old(i));
            });

        float max_diff = 0.0f;
        Kokkos::parallel_reduce(
            "JacobiKokkosMaxDiff",
            Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(n)),
            KOKKOS_LAMBDA(const int i, float& local_max) {
                if (diff(i) > local_max) {
                    local_max = diff(i);
                }
            },
            Kokkos::Max<float>(max_diff));

        if (max_diff < accuracy) {
            converged = true;
            break;
        }

        std::swap(x_old, x_new);
    }

    auto result_view = converged ? x_new : x_old;
    auto result_host = Kokkos::create_mirror_view(result_view);
    Kokkos::deep_copy(result_host, result_view);

    std::vector<float> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = result_host(i);
    }

    return result;
}
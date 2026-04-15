#include "shared_jacobi_oneapi.h"

std::vector<float> JacobiSharedONEAPI(
  const std::vector<float>& a, const std::vector<float>& b,
  float accuracy, sycl::device device) {
  size_t n = b.size();

  sycl::queue q(device);

  float* shared_a = sycl::malloc_shared<float>(a.size(), q);
  float* shared_b = sycl::malloc_shared<float>(n, q);
  float* shared_prev_x = sycl::malloc_shared<float>(n, q);
  float* shared_cur_x = sycl::malloc_shared<float>(n, q);
  float* shared_diff = sycl::malloc_shared<float>(1, q);

  std::copy(a.begin(), a.end(), shared_a);
  std::copy(b.begin(), b.end(), shared_b);
  std::fill(shared_prev_x, shared_prev_x + n, 0.0f);

  float diff = 0.0f;

  for (size_t iteration = 0; iteration < ITERATIONS; iteration++) {
    diff = 0.0f;
    *shared_diff = 0.0f;

    q.submit([&](sycl::handler& h) {
      auto reduction = sycl::reduction(shared_diff, sycl::plus<float>());

      h.parallel_for(sycl::range<1>(n), reduction, [=](sycl::id<1> idx, auto& sum_diff) {
        size_t i = idx[0];

        float res = 0.0f;
        for (size_t j = 0; j < n; j++) {
          if (i != j) {
            res += shared_a[i * n + j] * shared_prev_x[j];
          }
        }

        float new_x = (shared_b[i] - res) / shared_a[i * n + i];
        shared_cur_x[i] = new_x;
        sum_diff += (new_x - shared_prev_x[i]) * (new_x - shared_prev_x[i]);
        });
      });
    q.wait();

    diff = *shared_diff;

    if (diff < accuracy * accuracy)
      break;

    std::swap(shared_cur_x, shared_prev_x);
  }

  std::vector<float> result(n);
  std::copy(shared_cur_x, shared_cur_x + n, result.begin());

  sycl::free(shared_a, q);
  sycl::free(shared_b, q);
  sycl::free(shared_prev_x, q);
  sycl::free(shared_cur_x, q);
  sycl::free(shared_diff, q);

  return result;
}
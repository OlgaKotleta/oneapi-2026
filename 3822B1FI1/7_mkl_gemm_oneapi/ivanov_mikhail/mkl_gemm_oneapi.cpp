#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
  const std::vector<float>& a, const std::vector<float>& b,
  size_t size, sycl::device device) {
  sycl::queue q(device);
  
  const size_t n = size;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  std::vector<float> c(n * n);

  float* dev_a = sycl::malloc_device<float>(n * n, q);
  float* dev_b = sycl::malloc_device<float>(n * n, q);
  float* dev_c = sycl::malloc_device<float>(n * n, q);

  q.memcpy(dev_a, a.data(), sizeof(float) * n * n);
  q.memcpy(dev_b, b.data(), sizeof(float) * n * n);

  q.wait();

  oneapi::mkl::blas::row_major::gemm(
    q, oneapi::mkl::transpose::nontrans,
    oneapi::mkl::transpose::nontrans,
    n, n, n, alpha,
    dev_a, n, dev_b, n, 
    beta, dev_c, n);

  q.wait();

  q.memcpy(c.data(), dev_c, sizeof(float) * n * n).wait();

  sycl::free(dev_a, q);
  sycl::free(dev_b, q);
  sycl::free(dev_c, q);

  return c;
}
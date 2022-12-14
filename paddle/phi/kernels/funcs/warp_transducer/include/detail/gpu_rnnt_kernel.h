// Copyright 2018-2019, Mingkun Huang
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "rnnt_helper.h"
#include "type_def.h"

template <typename T>
inline __device__ T logp(const T* const denom,
                         const T* const acts,
                         const int maxT,
                         const int maxU,
                         const int alphabet_size,
                         int mb,
                         int t,
                         int u,
                         int v) {
  const int col = (mb * maxT + t) * maxU + u;
  return denom[col] + acts[col * alphabet_size + v];
}

template <typename Tp>
__global__ void compute_alphas_kernel(const Tp* const acts,
                                      const Tp* const denom,
                                      Tp* alphas,
                                      Tp* llForward,
                                      const int* const xlen,
                                      const int* const ylen,
                                      const int* const mlabels,
                                      const int minibatch,
                                      const int maxT,
                                      const int maxU,
                                      const int alphabet_size,
                                      const int blank_) {
  // launch B blocks, each block has U threads
  int b = blockIdx.x;   // batch
  int u = threadIdx.x;  // label id, u
  const int T = xlen[b];
  const int U = ylen[b] + 1;
  const int* labels = mlabels + b * (maxU - 1);  // mb label start point
  const int offset = b * maxT * maxU;
  alphas += offset;
  if (u == 0) alphas[0] = 0;

  __syncthreads();
  for (int n = 1; n < T + U - 1; ++n) {
    int t = n - u;
    if (u == 0) {
      if (t > 0 && t < T) {
        alphas[t * maxU + u] =
            alphas[(t - 1) * maxU + u] +
            logp(denom, acts, maxT, maxU, alphabet_size, b, t - 1, 0, blank_);
      }
    } else if (u < U) {
      if (t == 0)
        alphas[u] = alphas[u - 1] + logp(denom,
                                         acts,
                                         maxT,
                                         maxU,
                                         alphabet_size,
                                         b,
                                         0,
                                         u - 1,
                                         labels[u - 1]);
      else if (t > 0 && t < T) {
        Tp no_emit =
            alphas[(t - 1) * maxU + u] +
            logp(denom, acts, maxT, maxU, alphabet_size, b, t - 1, u, blank_);
        Tp emit = alphas[t * maxU + u - 1] + logp(denom,
                                                  acts,
                                                  maxT,
                                                  maxU,
                                                  alphabet_size,
                                                  b,
                                                  t,
                                                  u - 1,
                                                  labels[u - 1]);
        alphas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
      }
    }
    __syncthreads();
  }

  if (u == 0) {
    Tp loglike =
        alphas[(T - 1) * maxU + U - 1] +
        logp(denom, acts, maxT, maxU, alphabet_size, b, T - 1, U - 1, blank_);
    llForward[b] = loglike;
  }
}

template <typename Tp>
__global__ void compute_alphas_kernel_naive(const Tp* const acts,
                                            const Tp* const denom,
                                            Tp* alphas,
                                            Tp* llForward,
                                            const int* const xlen,
                                            const int* const ylen,
                                            const int* const mlabels,
                                            const int minibatch,
                                            const int maxT,
                                            const int maxU,
                                            const int alphabet_size,
                                            const int blank_) {
  int tid = threadIdx.x;  // mb
  const int T = xlen[tid];
  const int U = ylen[tid] + 1;
  const int* labels = mlabels + tid * (maxU - 1);  // mb label start point
  const int offset = tid * maxT * maxU;
  alphas += offset;
  alphas[0] = 0;

  for (int t = 0; t < T; ++t) {
    for (int u = 0; u < U; ++u) {
      if (u == 0 && t > 0)
        alphas[t * maxU + u] =
            alphas[(t - 1) * maxU + u] +
            logp(denom, acts, maxT, maxU, alphabet_size, tid, t - 1, 0, blank_);
      if (t == 0 && u > 0)
        alphas[u] = alphas[u - 1] + logp(denom,
                                         acts,
                                         maxT,
                                         maxU,
                                         alphabet_size,
                                         tid,
                                         0,
                                         u - 1,
                                         labels[u - 1]);
      if (t > 0 && u > 0) {
        Tp no_emit =
            alphas[(t - 1) * maxU + u] +
            logp(denom, acts, maxT, maxU, alphabet_size, tid, t - 1, u, blank_);
        Tp emit = alphas[t * maxU + u - 1] + logp(denom,
                                                  acts,
                                                  maxT,
                                                  maxU,
                                                  alphabet_size,
                                                  tid,
                                                  t,
                                                  u - 1,
                                                  labels[u - 1]);
        alphas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
      }
    }
  }

  Tp loglike =
      alphas[(T - 1) * maxU + U - 1] +
      logp(denom, acts, maxT, maxU, alphabet_size, tid, T - 1, U - 1, blank_);
  llForward[tid] = loglike;
}

template <typename Tp>
__global__ void compute_betas_kernel(const Tp* const acts,
                                     const Tp* const denom,
                                     Tp* betas,
                                     Tp* llBackward,
                                     const int* const xlen,
                                     const int* const ylen,
                                     const int* const mlabels,
                                     const int minibatch,
                                     const int maxT,
                                     const int maxU,
                                     const int alphabet_size,
                                     const int blank_) {
  int b = blockIdx.x;   // batch
  int u = threadIdx.x;  // label id, u
  const int T = xlen[b];
  const int U = ylen[b] + 1;
  const int* labels = mlabels + b * (maxU - 1);
  const int offset = b * maxT * maxU;
  betas += offset;
  if (u == 0)
    betas[(T - 1) * maxU + U - 1] =
        logp(denom, acts, maxT, maxU, alphabet_size, b, T - 1, U - 1, blank_);

  __syncthreads();
  for (int n = T + U - 2; n >= 0; --n) {
    int t = n - u;
    if (u == U - 1) {
      if (t >= 0 && t < T - 1)
        betas[t * maxU + U - 1] =
            betas[(t + 1) * maxU + U - 1] +
            logp(denom, acts, maxT, maxU, alphabet_size, b, t, U - 1, blank_);
    } else if (u < U) {
      if (t == T - 1)
        betas[(T - 1) * maxU + u] =
            betas[(T - 1) * maxU + u + 1] +
            logp(
                denom, acts, maxT, maxU, alphabet_size, b, T - 1, u, labels[u]);
      else if (t >= 0 && t < T - 1) {
        Tp no_emit =
            betas[(t + 1) * maxU + u] +
            logp(denom, acts, maxT, maxU, alphabet_size, b, t, u, blank_);
        Tp emit =
            betas[t * maxU + u + 1] +
            logp(denom, acts, maxT, maxU, alphabet_size, b, t, u, labels[u]);
        betas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
      }
    }
    __syncthreads();
  }

  if (u == 0) {
    llBackward[b] = betas[0];
  }
}

template <typename Tp>
__global__ void compute_betas_kernel_naive(const Tp* const acts,
                                           const Tp* const denom,
                                           Tp* betas,
                                           Tp* llBackward,
                                           const int* const xlen,
                                           const int* const ylen,
                                           const int* const mlabels,
                                           const int minibatch,
                                           const int maxT,
                                           const int maxU,
                                           const int alphabet_size,
                                           const int blank_) {
  int tid = threadIdx.x;  // mb
  const int T = xlen[tid];
  const int U = ylen[tid] + 1;
  const int* labels = mlabels + tid * (maxU - 1);
  const int offset = tid * maxT * maxU;
  betas += offset;
  betas[(T - 1) * maxU + U - 1] =
      logp(denom, acts, maxT, maxU, alphabet_size, tid, T - 1, U - 1, blank_);

  for (int t = T - 1; t >= 0; --t) {
    for (int u = U - 1; u >= 0; --u) {
      if (u == U - 1 && t < T - 1)
        betas[t * maxU + U - 1] =
            betas[(t + 1) * maxU + U - 1] +
            logp(denom, acts, maxT, maxU, alphabet_size, tid, t, U - 1, blank_);
      if (t == T - 1 && u < U - 1)
        betas[(T - 1) * maxU + u] =
            betas[(T - 1) * maxU + u + 1] + logp(denom,
                                                 acts,
                                                 maxT,
                                                 maxU,
                                                 alphabet_size,
                                                 tid,
                                                 T - 1,
                                                 u,
                                                 labels[u]);
      if (t < T - 1 && u < U - 1) {
        Tp no_emit =
            betas[(t + 1) * maxU + u] +
            logp(denom, acts, maxT, maxU, alphabet_size, tid, t, u, blank_);
        Tp emit =
            betas[t * maxU + u + 1] +
            logp(denom, acts, maxT, maxU, alphabet_size, tid, t, u, labels[u]);
        betas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
      }
    }
  }

  llBackward[tid] = betas[0];
}

template <int NT, typename Tp>
__global__ void compute_grad_kernel(Tp* grads,
                                    const Tp* const acts,
                                    const Tp* const denom,
                                    const Tp* alphas,
                                    const Tp* betas,
                                    const Tp* const logll,
                                    const int* const xlen,
                                    const int* const ylen,
                                    const int* const mlabels,
                                    const int minibatch,
                                    const int maxT,
                                    const int maxU,
                                    const int alphabet_size,
                                    const int blank_) {
  int tid = threadIdx.x;  // alphabet dim
  int idx = tid;
  int col = blockIdx.x;  // mb, t, u

  int u = col % maxU;
  int bt = (col - u) / maxU;
  int t = bt % maxT;
  int mb = (bt - t) / maxT;

  const int T = xlen[mb];
  const int U = ylen[mb] + 1;
  const int* labels = mlabels + mb * (maxU - 1);

  if (t < T && u < U) {
    while (idx < alphabet_size) {
      Tp logpk = denom[col] + acts[col * alphabet_size + idx];
      // Tp logpk = logp(denom, acts, maxT, maxU, alphabet_size, mb, t, u, idx);
      Tp grad = exp(alphas[col] + betas[col] + logpk - logll[mb]);
      // grad to last blank transition
      if (idx == blank_ && t == T - 1 && u == U - 1) {
        grad -= exp(alphas[col] + logpk - logll[mb]);
      }
      if (idx == blank_ && t < T - 1) {
        grad -= exp(alphas[col] + logpk - logll[mb] + betas[col + maxU]);
      }
      if (u < U - 1 && idx == labels[u]) {
        grad -= exp(alphas[col] + logpk - logll[mb] + betas[col + 1]);
      }
      grads[col * alphabet_size + idx] = grad;

      idx += NT;
    }
  }
}

template <int NT, typename Tp>
__global__ void compute_fastemit_grad_kernel(Tp* grads,
                                             const Tp* const acts,
                                             const Tp* const denom,
                                             const Tp* alphas,
                                             const Tp* betas,
                                             const Tp* const logll,
                                             const int* const xlen,
                                             const int* const ylen,
                                             const int* const mlabels,
                                             const int minibatch,
                                             const int maxT,
                                             const int maxU,
                                             const int alphabet_size,
                                             const int blank_,
                                             const Tp fastemit_lambda) {
  int tid = threadIdx.x;  // alphabet dim
  int idx = tid;
  int col = blockIdx.x;  // mb, t, u

  int u = col % maxU;
  int bt = (col - u) / maxU;
  int t = bt % maxT;
  int mb = (bt - t) / maxT;

  const int T = xlen[mb];
  const int U = ylen[mb] + 1;
  const int* labels = mlabels + mb * (maxU - 1);

  if (t < T && u < U) {
    while (idx < alphabet_size) {
      Tp logpk = denom[col] + acts[col * alphabet_size + idx];
      // Tp logpk = logp(denom, acts, maxT, maxU, alphabet_size, mb, t, u, idx);
      Tp grad = exp(alphas[col] + betas[col] + logpk - logll[mb]);

      Tp logy_btu1 =
          rnnt_helper::neg_inf<Tp>();  // log(y(t,u)) + log(beta(t, u+1))
      if (u < U - 1) {
        logy_btu1 =
            denom[col] + acts[col * alphabet_size + labels[u]] + betas[col + 1];
      }
      grad +=
          fastemit_lambda * exp(alphas[col] + logy_btu1 + logpk - logll[mb]);

      // grad to last blank transition
      if (idx == blank_ && t == T - 1 && u == U - 1) {
        grad -= exp(alphas[col] + logpk - logll[mb]);
        grad -=
            fastemit_lambda * exp(alphas[col] + logy_btu1 + logpk - logll[mb]);
      }
      if (idx == blank_ && t < T - 1) {
        grad -= exp(alphas[col] + logpk - logll[mb] + betas[col + maxU]);
      }
      if (u < U - 1 && idx == labels[u]) {
        grad -= exp(alphas[col] + logpk - logll[mb] + betas[col + 1]);
        grad -= fastemit_lambda * exp(alphas[col] + logy_btu1 - logll[mb]);
      }
      grads[col * alphabet_size + idx] = grad;

      idx += NT;
    }
  }
}

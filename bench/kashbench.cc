// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <xnnpack.h>

#include "models/models.h"


int main(int argc, char** argv) {
  printf("We are going to run a test of KashNet now...\n");
  if (xnn_initialize(nullptr /* allocator */) != xnn_status_success) {
    fprintf(stderr, "Failed to initialize XNNPACK");
    return 701;
  }

  const size_t num_threads = 8;
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool(
    pthreadpool_create(num_threads), pthreadpool_destroy);

  models::ExecutionPlan execution_plan = models::KashNet(threadpool.get());
  if (execution_plan.empty()) {
    fprintf(stderr, "failed to create a model");
    return 702;
  }

  for (const std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>& op : execution_plan) {
    xnn_status status = xnn_run_operator(op.get(), threadpool.get());
    if (status != xnn_status_success) {
      fprintf(stderr, "failed to run a model");
      return 703;
    }
  }
  printf("Run complete!\n");
  return 0;
}
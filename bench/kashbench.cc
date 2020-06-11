// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>
#include <chrono>
#include <iostream>

#include <xnnpack.h>

#include "models/models.h"


int main(int argc, char** argv) {
  printf("\nWe are going to run a test of KashNet now...\n");

  // ------------------------ EAGER ---------------------------

  printf("This is a test of the Eager API...\n");

  const size_t num_threads = 8;
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool(
    pthreadpool_create(num_threads), pthreadpool_destroy);


  alignas(16) static float v6[75264];
  alignas(16) static float v9[75264];
  alignas(16) static float v10[75264];

  for(size_t i = 0; i < 75264; ++i) {
    v9[i] = 4;
    v6[i] = 2;
    v10[i] = 0;
  }

  models::ExecutionPlan execution_plan;
  xnn_status status;

  if (xnn_initialize(nullptr) != xnn_status_success) {
    fprintf(stderr, "Failed to initialize XNNPACK");
    return 701;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  xnn_operator_t op9 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op9);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #9" << std::endl;
    std::cerr << status << std::endl;
    return 698;
  }
  execution_plan.emplace_back(op9, xnn_delete_operator);

  {
    const size_t a_shape[] = { 1, 56, 56, 24 };
    const size_t b_shape[] = { 1, 56, 56, 24 };
    status = xnn_setup_add_nd_f32(
      op9,
      4, a_shape, 4, b_shape,
      v9 /* a */, v6 /* b */, v10 /* output */,
      threadpool.get() /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #9" << std::endl;
    return 699;
  }

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

  auto end_time = std::chrono::high_resolution_clock::now();

  if (xnn_deinitialize() != xnn_status_success) {
    fprintf(stderr, "Failed to deinitialize XNNPACK");
    return 704;
  }

  auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);

  printf("Run complete! - Execution took %lld nanoseconds\n", nanoseconds.count());

  printf("First three values of output are :: ");
  for(size_t i = 0; i < 3; ++i) {
    printf("%d ", (int)v10[i]);
  }
  printf("\n\n");

  // ------------------------ GRAPH ---------------------------

  printf("This is a test of the Graph API...\n");

  for(size_t i = 0; i < 75264; ++i) {
    v9[i] = 4;
    v6[i] = 2;
    v10[i] = 0;
  }

  if (xnn_initialize(nullptr) != xnn_status_success) {
    fprintf(stderr, "Failed to initialize XNNPACK");
    return 701;
  }

  start_time = std::chrono::high_resolution_clock::now();
  
  xnn_subgraph_t adder_subgraph;

  if (xnn_create_subgraph(10, 0, &adder_subgraph) != xnn_status_success) {
    fprintf(stderr, "Failed to create XNNPACK subgraph");
    return 801;
  }
  
  const size_t v9_shape[] = { 1, 56, 56, 24 };
  uint32_t v9_id;
  if (xnn_define_tensor_value(adder_subgraph, xnn_datatype_fp32, 4, v9_shape, v9, XNN_INVALID_VALUE_ID, 0, &v9_id) != xnn_status_success) {
    fprintf(stderr, "Failed to define subgraph tensor");
    return 802;
  }

  const size_t v6_shape[] = { 1, 56, 56, 24 };
  uint32_t v6_id;
  if (xnn_define_tensor_value(adder_subgraph, xnn_datatype_fp32, 4, v6_shape, v6, XNN_INVALID_VALUE_ID, 0, &v6_id) != xnn_status_success) {
    fprintf(stderr, "Failed to define subgraph tensor");
    return 802;
  }

  xnn_external_value v10_ex_id = {0, v10};
  const size_t v10_shape[] = { 1, 56, 56, 24 };
  uint32_t v10_id;
  if (xnn_define_tensor_value(adder_subgraph, xnn_datatype_fp32, 4, v10_shape, nullptr, v10_ex_id.id, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &v10_id) != xnn_status_success) {
    fprintf(stderr, "Failed to define subgraph tensor");
    return 802;
  }

  if (xnn_define_add2(adder_subgraph, -1000, 1000, v9_id, v6_id, v10_id, 0) != xnn_status_success) {
    fprintf(stderr, "Failed to define subgraph add");
    return 803;
  }

  xnn_runtime_t adder_runtime;

  if (xnn_create_runtime(adder_subgraph, &adder_runtime) != xnn_status_success) {
    fprintf(stderr, "Failed to define subgraph runtime");
    return 804;
  }

  if (xnn_setup_runtime(adder_runtime, 1, &v10_ex_id) != xnn_status_success) {
    fprintf(stderr, "Failed to setup subgraph runtime");
    return 805;
  }

  if (xnn_invoke_runtime(adder_runtime) != xnn_status_success) {
    fprintf(stderr, "Failed to invoke subgraph runtime");
    return 806;
  }

  if (xnn_delete_runtime(adder_runtime) != xnn_status_success) {
    fprintf(stderr, "Failed to delete subgraph runtime");
    return 807;
  }

  end_time = std::chrono::high_resolution_clock::now();

  if (xnn_deinitialize() != xnn_status_success) {
    fprintf(stderr, "Failed to deinitialize XNNPACK");
    return 704;
  }

  nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);

  printf("Run complete! - Execution took %lld nanoseconds\n", nanoseconds.count());

  printf("First three values of output are :: ");
  for(size_t i = 0; i < 3; ++i) {
    printf("%d ", (int)v10[i]);
  }
  printf("\n\n");

  return 0;
}
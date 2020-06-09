// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "models/models.h"
#include <benchmark/benchmark.h>


#define BENCHMARK_KASH(benchmark_fn) \
  BENCHMARK_CAPTURE(benchmark_fn, kashnet, models::KashNet)->Unit(benchmark::kMicrosecond)->UseRealTime();

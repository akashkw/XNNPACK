// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include "models/models.h"

namespace models {

ExecutionPlan KashNet(pthreadpool_t threadpool) {
  alignas(16) static float v0[150528];
  alignas(16) static float v1[401408];
  alignas(16) static float v2[401408];
  alignas(16) static float v3[200704];
  alignas(16) static float v4[1204224];
  alignas(16) static float v5[301056];
  alignas(16) static float v6[75264];
  alignas(16) static float v7[451584];
  alignas(16) static float v8[451584];
  alignas(16) static float v9[75264];
  alignas(16) static float v10[75264];
  alignas(16) static float v11[451584];
  alignas(16) static float v12[112896];
  alignas(16) static float v13[25088];
  alignas(16) static float v14[150528];
  alignas(16) static float v15[150528];
  alignas(16) static float v16[25088];
  alignas(16) static float v17[25088];
  alignas(16) static float v18[150528];
  alignas(16) static float v19[150528];
  alignas(16) static float v20[25088];
  alignas(16) static float v21[25088];
  alignas(16) static float v22[150528];
  alignas(16) static float v23[37632];
  alignas(16) static float v24[12544];
  alignas(16) static float v25[75264];
  alignas(16) static float v26[75264];
  alignas(16) static float v27[12544];
  alignas(16) static float v28[12544];
  alignas(16) static float v29[75264];
  alignas(16) static float v30[75264];
  alignas(16) static float v31[12544];
  alignas(16) static float v32[12544];
  alignas(16) static float v33[75264];
  alignas(16) static float v34[75264];
  alignas(16) static float v35[12544];
  alignas(16) static float v36[12544];
  alignas(16) static float v37[75264];
  alignas(16) static float v38[75264];
  alignas(16) static float v39[18816];
  alignas(16) static float v40[112896];
  alignas(16) static float v41[112896];
  alignas(16) static float v42[18816];
  alignas(16) static float v43[18816];
  alignas(16) static float v44[112896];
  alignas(16) static float v45[112896];
  alignas(16) static float v46[18816];
  alignas(16) static float v47[18816];
  alignas(16) static float v48[112896];
  alignas(16) static float v49[28224];
  alignas(16) static float v50[7840];
  alignas(16) static float v51[47040];
  alignas(16) static float v52[47040];
  alignas(16) static float v53[7840];
  alignas(16) static float v54[7840];
  alignas(16) static float v55[47040];
  alignas(16) static float v56[47040];
  alignas(16) static float v57[7840];
  alignas(16) static float v58[7840];
  alignas(16) static float v59[47040];
  alignas(16) static float v60[47040];
  alignas(16) static float v61[15680];
  alignas(16) static float v62[62720];
  alignas(16) static float v63[1280];
  alignas(16) static float v64[1001];
  alignas(16) static float w65[864];
  alignas(16) static float w66[32];
  alignas(16) static float w67[288];
  alignas(16) static float w68[32];
  alignas(16) static float w69[512];
  alignas(16) static float w70[16];
  alignas(16) static float w71[1536];
  alignas(16) static float w72[96];
  alignas(16) static float w73[864];
  alignas(16) static float w74[96];
  alignas(16) static float w75[2304];
  alignas(16) static float w76[24];
  alignas(16) static float w77[3456];
  alignas(16) static float w78[144];
  alignas(16) static float w79[1296];
  alignas(16) static float w80[144];
  alignas(16) static float w81[3456];
  alignas(16) static float w82[24];
  alignas(16) static float w83[3456];
  alignas(16) static float w84[144];
  alignas(16) static float w85[1296];
  alignas(16) static float w86[144];
  alignas(16) static float w87[4608];
  alignas(16) static float w88[32];
  alignas(16) static float w89[6144];
  alignas(16) static float w90[192];
  alignas(16) static float w91[1728];
  alignas(16) static float w92[192];
  alignas(16) static float w93[6144];
  alignas(16) static float w94[32];
  alignas(16) static float w95[6144];
  alignas(16) static float w96[192];
  alignas(16) static float w97[1728];
  alignas(16) static float w98[192];
  alignas(16) static float w99[6144];
  alignas(16) static float w100[32];
  alignas(16) static float w101[6144];
  alignas(16) static float w102[192];
  alignas(16) static float w103[1728];
  alignas(16) static float w104[192];
  alignas(16) static float w105[12288];
  alignas(16) static float w106[64];
  alignas(16) static float w107[24576];
  alignas(16) static float w108[384];
  alignas(16) static float w109[3456];
  alignas(16) static float w110[384];
  alignas(16) static float w111[24576];
  alignas(16) static float w112[64];
  alignas(16) static float w113[24576];
  alignas(16) static float w114[384];
  alignas(16) static float w115[3456];
  alignas(16) static float w116[384];
  alignas(16) static float w117[24576];
  alignas(16) static float w118[64];
  alignas(16) static float w119[24576];
  alignas(16) static float w120[384];
  alignas(16) static float w121[3456];
  alignas(16) static float w122[384];
  alignas(16) static float w123[24576];
  alignas(16) static float w124[64];
  alignas(16) static float w125[24576];
  alignas(16) static float w126[384];
  alignas(16) static float w127[3456];
  alignas(16) static float w128[384];
  alignas(16) static float w129[36864];
  alignas(16) static float w130[96];
  alignas(16) static float w131[55296];
  alignas(16) static float w132[576];
  alignas(16) static float w133[5184];
  alignas(16) static float w134[576];
  alignas(16) static float w135[55296];
  alignas(16) static float w136[96];
  alignas(16) static float w137[55296];
  alignas(16) static float w138[576];
  alignas(16) static float w139[5184];
  alignas(16) static float w140[576];
  alignas(16) static float w141[55296];
  alignas(16) static float w142[96];
  alignas(16) static float w143[55296];
  alignas(16) static float w144[576];
  alignas(16) static float w145[5184];
  alignas(16) static float w146[576];
  alignas(16) static float w147[92160];
  alignas(16) static float w148[160];
  alignas(16) static float w149[153600];
  alignas(16) static float w150[960];
  alignas(16) static float w151[8640];
  alignas(16) static float w152[960];
  alignas(16) static float w153[153600];
  alignas(16) static float w154[160];
  alignas(16) static float w155[153600];
  alignas(16) static float w156[960];
  alignas(16) static float w157[8640];
  alignas(16) static float w158[960];
  alignas(16) static float w159[153600];
  alignas(16) static float w160[160];
  alignas(16) static float w161[153600];
  alignas(16) static float w162[960];
  alignas(16) static float w163[8640];
  alignas(16) static float w164[960];
  alignas(16) static float w165[307200];
  alignas(16) static float w166[320];
  alignas(16) static float w167[409600];
  alignas(16) static float w168[1280];
  alignas(16) static float w169[1281280];
  alignas(16) static float w170[1001];

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f), rng);
  std::generate(v0, v0 + 150528, std::ref(f32rng));
  std::generate(v1, v1 + 401408, std::ref(f32rng));
  std::generate(v2, v2 + 401408, std::ref(f32rng));
  std::generate(v3, v3 + 200704, std::ref(f32rng));
  std::generate(v4, v4 + 1204224, std::ref(f32rng));
  std::generate(v5, v5 + 301056, std::ref(f32rng));
  std::generate(v6, v6 + 75264, std::ref(f32rng));
  std::generate(v7, v7 + 451584, std::ref(f32rng));
  std::generate(v8, v8 + 451584, std::ref(f32rng));
  std::generate(v9, v9 + 75264, std::ref(f32rng));
  std::generate(v10, v10 + 75264, std::ref(f32rng));
  std::generate(v11, v11 + 451584, std::ref(f32rng));
  std::generate(v12, v12 + 112896, std::ref(f32rng));
  std::generate(v13, v13 + 25088, std::ref(f32rng));
  std::generate(v14, v14 + 150528, std::ref(f32rng));
  std::generate(v15, v15 + 150528, std::ref(f32rng));
  std::generate(v16, v16 + 25088, std::ref(f32rng));
  std::generate(v17, v17 + 25088, std::ref(f32rng));
  std::generate(v18, v18 + 150528, std::ref(f32rng));
  std::generate(v19, v19 + 150528, std::ref(f32rng));
  std::generate(v20, v20 + 25088, std::ref(f32rng));
  std::generate(v21, v21 + 25088, std::ref(f32rng));
  std::generate(v22, v22 + 150528, std::ref(f32rng));
  std::generate(v23, v23 + 37632, std::ref(f32rng));
  std::generate(v24, v24 + 12544, std::ref(f32rng));
  std::generate(v25, v25 + 75264, std::ref(f32rng));
  std::generate(v26, v26 + 75264, std::ref(f32rng));
  std::generate(v27, v27 + 12544, std::ref(f32rng));
  std::generate(v28, v28 + 12544, std::ref(f32rng));
  std::generate(v29, v29 + 75264, std::ref(f32rng));
  std::generate(v30, v30 + 75264, std::ref(f32rng));
  std::generate(v31, v31 + 12544, std::ref(f32rng));
  std::generate(v32, v32 + 12544, std::ref(f32rng));
  std::generate(v33, v33 + 75264, std::ref(f32rng));
  std::generate(v34, v34 + 75264, std::ref(f32rng));
  std::generate(v35, v35 + 12544, std::ref(f32rng));
  std::generate(v36, v36 + 12544, std::ref(f32rng));
  std::generate(v37, v37 + 75264, std::ref(f32rng));
  std::generate(v38, v38 + 75264, std::ref(f32rng));
  std::generate(v39, v39 + 18816, std::ref(f32rng));
  std::generate(v40, v40 + 112896, std::ref(f32rng));
  std::generate(v41, v41 + 112896, std::ref(f32rng));
  std::generate(v42, v42 + 18816, std::ref(f32rng));
  std::generate(v43, v43 + 18816, std::ref(f32rng));
  std::generate(v44, v44 + 112896, std::ref(f32rng));
  std::generate(v45, v45 + 112896, std::ref(f32rng));
  std::generate(v46, v46 + 18816, std::ref(f32rng));
  std::generate(v47, v47 + 18816, std::ref(f32rng));
  std::generate(v48, v48 + 112896, std::ref(f32rng));
  std::generate(v49, v49 + 28224, std::ref(f32rng));
  std::generate(v50, v50 + 7840, std::ref(f32rng));
  std::generate(v51, v51 + 47040, std::ref(f32rng));
  std::generate(v52, v52 + 47040, std::ref(f32rng));
  std::generate(v53, v53 + 7840, std::ref(f32rng));
  std::generate(v54, v54 + 7840, std::ref(f32rng));
  std::generate(v55, v55 + 47040, std::ref(f32rng));
  std::generate(v56, v56 + 47040, std::ref(f32rng));
  std::generate(v57, v57 + 7840, std::ref(f32rng));
  std::generate(v58, v58 + 7840, std::ref(f32rng));
  std::generate(v59, v59 + 47040, std::ref(f32rng));
  std::generate(v60, v60 + 47040, std::ref(f32rng));
  std::generate(v61, v61 + 15680, std::ref(f32rng));
  std::generate(v62, v62 + 62720, std::ref(f32rng));
  std::generate(v63, v63 + 1280, std::ref(f32rng));
  std::generate(v64, v64 + 1001, std::ref(f32rng));
  std::generate(w65, w65 + 864, std::ref(f32rng));
  std::generate(w66, w66 + 32, std::ref(f32rng));
  std::generate(w67, w67 + 288, std::ref(f32rng));
  std::generate(w68, w68 + 32, std::ref(f32rng));
  std::generate(w69, w69 + 512, std::ref(f32rng));
  std::generate(w70, w70 + 16, std::ref(f32rng));
  std::generate(w71, w71 + 1536, std::ref(f32rng));
  std::generate(w72, w72 + 96, std::ref(f32rng));
  std::generate(w73, w73 + 864, std::ref(f32rng));
  std::generate(w74, w74 + 96, std::ref(f32rng));
  std::generate(w75, w75 + 2304, std::ref(f32rng));
  std::generate(w76, w76 + 24, std::ref(f32rng));
  std::generate(w77, w77 + 3456, std::ref(f32rng));
  std::generate(w78, w78 + 144, std::ref(f32rng));
  std::generate(w79, w79 + 1296, std::ref(f32rng));
  std::generate(w80, w80 + 144, std::ref(f32rng));
  std::generate(w81, w81 + 3456, std::ref(f32rng));
  std::generate(w82, w82 + 24, std::ref(f32rng));
  std::generate(w83, w83 + 3456, std::ref(f32rng));
  std::generate(w84, w84 + 144, std::ref(f32rng));
  std::generate(w85, w85 + 1296, std::ref(f32rng));
  std::generate(w86, w86 + 144, std::ref(f32rng));
  std::generate(w87, w87 + 4608, std::ref(f32rng));
  std::generate(w88, w88 + 32, std::ref(f32rng));
  std::generate(w89, w89 + 6144, std::ref(f32rng));
  std::generate(w90, w90 + 192, std::ref(f32rng));
  std::generate(w91, w91 + 1728, std::ref(f32rng));
  std::generate(w92, w92 + 192, std::ref(f32rng));
  std::generate(w93, w93 + 6144, std::ref(f32rng));
  std::generate(w94, w94 + 32, std::ref(f32rng));
  std::generate(w95, w95 + 6144, std::ref(f32rng));
  std::generate(w96, w96 + 192, std::ref(f32rng));
  std::generate(w97, w97 + 1728, std::ref(f32rng));
  std::generate(w98, w98 + 192, std::ref(f32rng));
  std::generate(w99, w99 + 6144, std::ref(f32rng));
  std::generate(w100, w100 + 32, std::ref(f32rng));
  std::generate(w101, w101 + 6144, std::ref(f32rng));
  std::generate(w102, w102 + 192, std::ref(f32rng));
  std::generate(w103, w103 + 1728, std::ref(f32rng));
  std::generate(w104, w104 + 192, std::ref(f32rng));
  std::generate(w105, w105 + 12288, std::ref(f32rng));
  std::generate(w106, w106 + 64, std::ref(f32rng));
  std::generate(w107, w107 + 24576, std::ref(f32rng));
  std::generate(w108, w108 + 384, std::ref(f32rng));
  std::generate(w109, w109 + 3456, std::ref(f32rng));
  std::generate(w110, w110 + 384, std::ref(f32rng));
  std::generate(w111, w111 + 24576, std::ref(f32rng));
  std::generate(w112, w112 + 64, std::ref(f32rng));
  std::generate(w113, w113 + 24576, std::ref(f32rng));
  std::generate(w114, w114 + 384, std::ref(f32rng));
  std::generate(w115, w115 + 3456, std::ref(f32rng));
  std::generate(w116, w116 + 384, std::ref(f32rng));
  std::generate(w117, w117 + 24576, std::ref(f32rng));
  std::generate(w118, w118 + 64, std::ref(f32rng));
  std::generate(w119, w119 + 24576, std::ref(f32rng));
  std::generate(w120, w120 + 384, std::ref(f32rng));
  std::generate(w121, w121 + 3456, std::ref(f32rng));
  std::generate(w122, w122 + 384, std::ref(f32rng));
  std::generate(w123, w123 + 24576, std::ref(f32rng));
  std::generate(w124, w124 + 64, std::ref(f32rng));
  std::generate(w125, w125 + 24576, std::ref(f32rng));
  std::generate(w126, w126 + 384, std::ref(f32rng));
  std::generate(w127, w127 + 3456, std::ref(f32rng));
  std::generate(w128, w128 + 384, std::ref(f32rng));
  std::generate(w129, w129 + 36864, std::ref(f32rng));
  std::generate(w130, w130 + 96, std::ref(f32rng));
  std::generate(w131, w131 + 55296, std::ref(f32rng));
  std::generate(w132, w132 + 576, std::ref(f32rng));
  std::generate(w133, w133 + 5184, std::ref(f32rng));
  std::generate(w134, w134 + 576, std::ref(f32rng));
  std::generate(w135, w135 + 55296, std::ref(f32rng));
  std::generate(w136, w136 + 96, std::ref(f32rng));
  std::generate(w137, w137 + 55296, std::ref(f32rng));
  std::generate(w138, w138 + 576, std::ref(f32rng));
  std::generate(w139, w139 + 5184, std::ref(f32rng));
  std::generate(w140, w140 + 576, std::ref(f32rng));
  std::generate(w141, w141 + 55296, std::ref(f32rng));
  std::generate(w142, w142 + 96, std::ref(f32rng));
  std::generate(w143, w143 + 55296, std::ref(f32rng));
  std::generate(w144, w144 + 576, std::ref(f32rng));
  std::generate(w145, w145 + 5184, std::ref(f32rng));
  std::generate(w146, w146 + 576, std::ref(f32rng));
  std::generate(w147, w147 + 92160, std::ref(f32rng));
  std::generate(w148, w148 + 160, std::ref(f32rng));
  std::generate(w149, w149 + 153600, std::ref(f32rng));
  std::generate(w150, w150 + 960, std::ref(f32rng));
  std::generate(w151, w151 + 8640, std::ref(f32rng));
  std::generate(w152, w152 + 960, std::ref(f32rng));
  std::generate(w153, w153 + 153600, std::ref(f32rng));
  std::generate(w154, w154 + 160, std::ref(f32rng));
  std::generate(w155, w155 + 153600, std::ref(f32rng));
  std::generate(w156, w156 + 960, std::ref(f32rng));
  std::generate(w157, w157 + 8640, std::ref(f32rng));
  std::generate(w158, w158 + 960, std::ref(f32rng));
  std::generate(w159, w159 + 153600, std::ref(f32rng));
  std::generate(w160, w160 + 160, std::ref(f32rng));
  std::generate(w161, w161 + 153600, std::ref(f32rng));
  std::generate(w162, w162 + 960, std::ref(f32rng));
  std::generate(w163, w163 + 8640, std::ref(f32rng));
  std::generate(w164, w164 + 960, std::ref(f32rng));
  std::generate(w165, w165 + 307200, std::ref(f32rng));
  std::generate(w166, w166 + 320, std::ref(f32rng));
  std::generate(w167, w167 + 409600, std::ref(f32rng));
  std::generate(w168, w168 + 1280, std::ref(f32rng));
  std::generate(w169, w169 + 1281280, std::ref(f32rng));
  std::generate(w170, w170 + 1001, std::ref(f32rng));

  for(size_t i = 0; i < 75264; ++i) {
    v9[i] = 4;
    v6[i] = 2;
  }

  ExecutionPlan operators;
  xnn_status status;

  xnn_operator_t op9 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op9);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #9" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op9, xnn_delete_operator);

  {
    const size_t a_shape[] = { 1, 56, 56, 24 };
    const size_t b_shape[] = { 1, 56, 56, 24 };
    status = xnn_setup_add_nd_f32(
      op9,
      4, a_shape, 4, b_shape,
      v9 /* a */, v6 /* b */, v10 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #9" << std::endl;
    return ExecutionPlan();
  }

  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wpessimizing-move"
  return operators;
  #pragma clang diagnostic pop
}

}  // namespace models

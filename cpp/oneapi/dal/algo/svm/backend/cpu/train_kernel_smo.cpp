/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "oneapi/dal/algo/svm/backend/cpu/train_kernel.hpp"

namespace oneapi::dal::svm::backend {

template <typename Float>
struct train_kernel_cpu<Float, task::classification, method::smo> {
    train_result operator()(const dal::backend::context_cpu& ctx,
                            const descriptor_base& params,
                            const train_input& input) const {
        return train_result();
    }
};

template struct train_kernel_cpu<float, task::classification, method::smo>;
template struct train_kernel_cpu<double, task::classification, method::smo>;

} // namespace oneapi::dal::svm::backend
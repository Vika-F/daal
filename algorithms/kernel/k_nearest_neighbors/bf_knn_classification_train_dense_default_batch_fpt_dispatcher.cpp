/* file: bf_knn_classification_train_dense_default_batch_fpt_dispatcher.cpp */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

#include "algorithms/kernel/k_nearest_neighbors/bf_knn_classification_train_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER_SYCL(bf_knn_classification::training::BatchContainer, batch, DAAL_FPTYPE,
                                           bf_knn_classification::training::defaultDense)
namespace bf_knn_classification
{
namespace training
{
namespace interface1
{
template <typename algorithmFPType, bf_knn_classification::training::Method method>
Batch<algorithmFPType, method>::Batch() : classifier::training::Batch()
{
    _par = new ParameterType();
    initialize();
}

template <typename algorithmFPType, bf_knn_classification::training::Method method>
Batch<algorithmFPType, method>::Batch(const Batch & other) : classifier::training::Batch(other)
{
    _par = new ParameterType(other.parameter());
    initialize();
}

template class Batch<DAAL_FPTYPE, defaultDense>;

} // namespace interface1
} // namespace training
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

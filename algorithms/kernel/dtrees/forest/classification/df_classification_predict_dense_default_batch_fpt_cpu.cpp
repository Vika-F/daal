/* file: df_classification_predict_dense_default_batch_fpt_cpu.cpp */
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

/*
//++
//  Implementation of prediction stage of decision forest algorithm.
//--
*/

#include "algorithms/kernel/dtrees/forest/classification/df_classification_predict_dense_default_batch.h"
#include "algorithms/kernel/dtrees/forest/classification/df_classification_predict_dense_default_batch_impl.i"
#include "algorithms/kernel/dtrees/forest/classification/df_classification_predict_dense_default_batch_container.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace prediction
{
namespace interface3
{
template class BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
}
namespace internal
{
template class PredictKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
}
} // namespace prediction
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

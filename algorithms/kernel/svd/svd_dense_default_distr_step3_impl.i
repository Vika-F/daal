/* file: svd_dense_default_distr_step3_impl.i */
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
//  Implementation of svds
//--
*/

#ifndef __SVD_DENSE_DEFAULT_DISTR_STEP3_IMPL_I__
#define __SVD_DENSE_DEFAULT_DISTR_STEP3_IMPL_I__

#include "externals/service_memory.h"
#include "externals/service_math.h"
#include "service/kernel/service_defines.h"
#include "service/kernel/data_management/service_numeric_table.h"
#include "algorithms/kernel/service_error_handling.h"

#include "algorithms/kernel/svd/svd_dense_default_impl.i"

#include "algorithms/threading/threading.h"

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace internal
{
template <typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
Status SVDDistributedStep3Kernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable * const * a, const size_t nr,
                                                                        NumericTable * r[], const daal::algorithms::Parameter * par)
{
    size_t nBlocks     = na / 2;
    size_t mCalculated = 0;

    ReadRows<algorithmFPType, cpu, NumericTable> Aux1iBlock;
    ReadRows<algorithmFPType, cpu, NumericTable> Aux3iBlock;
    WriteOnlyRows<algorithmFPType, cpu, NumericTable> QiBlock;

    for (size_t k = 0; k < nBlocks; k++)
    {
        NumericTable * ntAux1i = const_cast<NumericTable *>(a[k]);
        NumericTable * ntAux3i = const_cast<NumericTable *>(a[k + nBlocks]);

        const size_t n = ntAux1i->getNumberOfColumns();
        const size_t m = ntAux1i->getNumberOfRows();
        
        const size_t nComponents = ntAux3i->getNumberOfRows();

        const algorithmFPType * Aux1i = Aux1iBlock.set(ntAux1i, 0, m); /* Aux1i = Qin[m][nComponents] */
        DAAL_CHECK_BLOCK_STATUS(Aux1iBlock);

        const algorithmFPType * Aux3i = Aux3iBlock.set(ntAux3i, 0, nComponents); /* Aux3i = Ri [nComponents][n] */
        DAAL_CHECK_BLOCK_STATUS(Aux3iBlock);

        algorithmFPType * Qi = QiBlock.set(r[0], mCalculated, m); /* Qi [m][n] */
        DAAL_CHECK_BLOCK_STATUS(QiBlock);

        DAAL_INT ldAux1i = n;
        DAAL_INT ldAux3i = nComponents;
        DAAL_INT ldQi    = nComponents;

        const auto ec = compute_gemm_on_one_node<algorithmFPType, cpu>('N', 'N', nComponents, m, nComponents, const_cast<algorithmFPType *>(Aux3i), ldAux3i, const_cast<algorithmFPType *>(Aux1i), ldAux1i, Qi, ldQi);
        if (!ec) return ec;

        mCalculated += m;
    }

    return Status();
}

} // namespace internal
} // namespace svd
} // namespace algorithms
} // namespace daal

#endif

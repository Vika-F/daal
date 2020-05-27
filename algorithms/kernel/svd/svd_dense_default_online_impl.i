/* file: svd_dense_default_online_impl.i */
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

#ifndef __SVD_KERNEL_ONLINE_IMPL_I__
#define __SVD_KERNEL_ONLINE_IMPL_I__

#include "externals/service_memory.h"
#include "externals/service_math.h"
#include "service/kernel/service_defines.h"
#include "service/kernel/data_management/service_numeric_table.h"

#include "algorithms/kernel/svd/svd_dense_default_kernel.h"
#include "algorithms/kernel/svd/svd_dense_default_impl.i"

#include "algorithms/threading/threading.h"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace internal
{
template <typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
Status SVDOnlineKernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                                                              const daal::algorithms::Parameter * par)
{
    size_t i, j;
    svd::Parameter defaultParams;
    const svd::Parameter * svdPar = &defaultParams;

    if (par != 0)
    {
        svdPar = static_cast<const svd::Parameter *>(par);
    }

    NumericTable * const ntAi    = const_cast<NumericTable *>(a[0]);
    NumericTable * const ntAux2i = r[1];

    const size_t n = ntAi->getNumberOfColumns();
    const size_t m = ntAi->getNumberOfRows();
    const size_t nComponents = ntAux2i->getNumberOfRows();

    ReadRows<algorithmFPType, cpu, NumericTable> AiBlock(ntAi, 0, m); /* Ai [m][n] */
    DAAL_CHECK_BLOCK_STATUS(AiBlock);
    const algorithmFPType * Ai = AiBlock.get();

    WriteOnlyRows<algorithmFPType, cpu, NumericTable> Aux2iBlock(ntAux2i, 0, nComponents); /* Aux2i = Ri [nComponents][n] */
    DAAL_CHECK_BLOCK_STATUS(Aux2iBlock);
    algorithmFPType * Aux2i = Aux2iBlock.get();

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, m);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, nComponents);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * m, sizeof(algorithmFPType));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * nComponents, sizeof(algorithmFPType));

    TArray<algorithmFPType, cpu> Aux1iTPtr(n * m);
    TArray<algorithmFPType, cpu> Aux2iTPtr(n * nComponents);
    algorithmFPType * Aux1iT = Aux1iTPtr.get();
    algorithmFPType * Aux2iT = Aux2iTPtr.get();

    DAAL_CHECK(Aux1iT && Aux2iT, ErrorMemoryAllocationFailed);

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < m; j++)
        {
            Aux1iT[i * m + j] = Ai[i + j * n];
        }
    }

    const auto ec = compute_QR_on_one_node<algorithmFPType, cpu>(m, n, Aux1iT, m, Aux2iT, nComponents);
    if (!ec) return ec;

    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        WriteOnlyRows<algorithmFPType, cpu, NumericTable> Aux1iBlock(r[0], 0, m); /* Aux1i = Qin[m][nComponents] */
        DAAL_CHECK_BLOCK_STATUS(Aux1iBlock);
        algorithmFPType * Aux1i = Aux1iBlock.get();

        for (i = 0; i < nComponents; i++)
        {
            for (j = 0; j < m; j++)
            {
                Aux1i[i + j * nComponents] = Aux1iT[i * m + j];
            }
        }
    }

    for (i = 0; i < nComponents; i++)
    {
        for (j = 0; j <= i; j++)
        {
            Aux2i[i + j * n] = Aux2iT[i * nComponents + j];
        }
        for (; j < nComponents; j++)
        {
            Aux2i[i + j * n] = 0.0;
        }
    }

    for (i = nComponents; i < n; i++)
    {
        for (j = 0; j < nComponents; j++)
        {
            Aux2i[i + j * n] = Aux2iT[i * nComponents + j];
        }
    }

    return Status();
}

template <typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
Status SVDOnlineKernel<algorithmFPType, method, cpu>::finalizeCompute(const size_t na, const NumericTable * const * a, const size_t nr,
                                                                      NumericTable * r[], const daal::algorithms::Parameter * par)
{
    svd::Parameter defaultParams;
    const svd::Parameter * svdPar = &defaultParams;

    if (par != 0)
    {
        svdPar = static_cast<const svd::Parameter *>(par);
    }

    const NumericTable * ntAux2_0 = a[0];
    NumericTable * ntSigma        = r[0];
    NumericTable * ntV            = r[2];

    const size_t nBlocks = na / 2;
    const size_t n       = ntAux2_0->getNumberOfColumns();

    /* Step 2 */
    const NumericTable * const * step2ntIn = a;

    TArray<NumericTable *, cpu> step2ntOut(nBlocks + 2);
    DAAL_CHECK(step2ntOut.get(), ErrorMemoryAllocationFailed);

    step2ntOut[0] = ntSigma;
    step2ntOut[1] = ntV;
    Status st;
    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        size_t nRowsFull = 0;
        for (size_t k = 0; k < nBlocks; k++)
        {
            nRowsFull += a[k]->getNumberOfRows();
        }
        const size_t nColumns = (nRowsFull < n ? nRowsFull : n);
        for (size_t k = 0; k < nBlocks; k++)
        {
            const NumericTable * const ntRk = a[k];
            const size_t nComponents = ntRk->getNumberOfRows();
            step2ntOut[2 + k] = new HomogenNumericTableCPU<algorithmFPType, cpu>(nColumns, nComponents, st);
            DAAL_CHECK_STATUS_VAR(st);
            if (!step2ntOut[2 + k])
            {
                for (size_t j = 0; j < k; j++)
                {
                    delete step2ntOut[2 + j];
                    step2ntOut[2 + k] = nullptr;
                }
                return Status(ErrorMemoryAllocationFailed);
            }
        }
    }

    SVDDistributedStep2Kernel<algorithmFPType, method, cpu> kernel;
    Status s = kernel.compute(nBlocks, step2ntIn, nBlocks + 2, step2ntOut.get(), par);

    /* Step 3 */

    if (s && svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        WriteOnlyRows<algorithmFPType, cpu, NumericTable> QBlock; /* Aux1i = Qin[m][n] */
        const size_t nComponents = ntSigma->getNumberOfColumns();

        size_t computedRows = 0;

        for (size_t i = 0; i < nBlocks; i++)
        {
            const NumericTable * const ntAux1i = a[nBlocks + i];
            const size_t m                     = ntAux1i->getNumberOfRows();

            algorithmFPType * Qi = QBlock.set(r[1], computedRows, m);
            s                    = QBlock.status();
            if (!s) break;

            HomogenNumericTableCPU<algorithmFPType, cpu> ntQi(Qi, nComponents, m, s);
            DAAL_CHECK_STATUS_VAR(s);

            const NumericTable * step3ntIn[2] = { ntAux1i, step2ntOut[2 + i] };
            NumericTable * step3ntOut[1]      = { &ntQi };

            SVDDistributedStep3Kernel<algorithmFPType, method, cpu> kernelStep3;
            s = kernelStep3.compute(2, step3ntIn, 1, step3ntOut, par);

            if (!s) break;

            computedRows += m;
        }
    }

    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        for (size_t k = 0; k < nBlocks; k++)
        {
            delete step2ntOut[2 + k];
            step2ntOut[2 + k] = nullptr;
        }
    }

    return s;
}

} // namespace internal
} // namespace svd
} // namespace algorithms
} // namespace daal

#endif

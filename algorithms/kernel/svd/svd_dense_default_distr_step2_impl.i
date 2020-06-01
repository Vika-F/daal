/* file: svd_dense_default_distr_step2_impl.i */
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

#ifndef __SVD_DENSE_DEFAULT_DISTR_STEP2_IMPL_I__
#define __SVD_DENSE_DEFAULT_DISTR_STEP2_IMPL_I__

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
Status SVDDistributedStep2Kernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable * const * a, const size_t nr,
                                                                        NumericTable * r[], const daal::algorithms::Parameter * par)
{
    svd::Parameter defaultParams;
    const svd::Parameter * svdPar = &defaultParams;

    if (par != 0)
    {
        svdPar = static_cast<const svd::Parameter *>(par);
    }

    const NumericTable * const ntAux2_0 = a[0];
    NumericTable * const ntSigma        = const_cast<NumericTable *>(r[0]);

    const size_t nBlocks = na;

    const size_t n   = ntAux2_0->getNumberOfColumns();
    size_t nRowsFull = 0;   /* Full number of rows in all partial matrices Ri - partial results of QR decomposition */
    TArray<size_t, cpu> nRowsOffsetsPtr(nBlocks);
    size_t * const nRowsOffsets = nRowsOffsetsPtr.get();
    DAAL_CHECK(nRowsOffsets, ErrorMemoryAllocationFailed);
    for (size_t i = 0; i < nBlocks; ++i)
    {
        nRowsOffsets[i] = nRowsFull;
        nRowsFull += a[i]->getNumberOfRows();
    }
    const size_t nColsInSigma = ntSigma->getNumberOfColumns();
    const size_t nComponents = (nRowsFull < nColsInSigma ? nRowsFull : nColsInSigma);

    WriteOnlyRows<algorithmFPType, cpu, NumericTable> VBlock;
    WriteOnlyRows<algorithmFPType, cpu, NumericTable> sigmaBlock(ntSigma, 0, 1); /* Sigma [1][nColsInSigma]   */
    DAAL_CHECK_BLOCK_STATUS(sigmaBlock);
    algorithmFPType * Sigma = sigmaBlock.get();

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, nRowsFull);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * nRowsFull, sizeof(algorithmFPType));
    
    TArray<algorithmFPType, cpu> UPtr;
    TArray<algorithmFPType, cpu> Aux2TPtr(n * nRowsFull);

    const DAAL_INT ldU = nComponents;

    algorithmFPType * U     = nullptr;
    algorithmFPType * VT    = nullptr;
    algorithmFPType * Aux2T = Aux2TPtr.get();

    DAAL_CHECK(Aux2T, ErrorMemoryAllocationFailed);

    SafeStatus safeStat;

    daal::threader_for(nBlocks, nBlocks, [=, &safeStat](int k) {
        const size_t nRows = a[k]->getNumberOfRows();
        ReadRows<algorithmFPType, cpu, NumericTable> aux2Block(const_cast<NumericTable *>(a[k]), 0, nRows); /* Aux2  [nRows][n] */
        DAAL_CHECK_BLOCK_STATUS_THR(aux2Block);
        const algorithmFPType * Aux2 = aux2Block.get();

        for (size_t i = 0; i < nRows; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                Aux2T[j * nRowsFull + nRowsOffsets[k] + i] = Aux2[i * n + j];
            }
        }
    });
    if (!safeStat) return safeStat.detach();

    {
        const DAAL_INT ldAux2 = nRowsFull;
        const DAAL_INT ldV    = n;

        const DAAL_INT ldRT = nComponents;
        const DAAL_INT ldR  = n;

        TArray<algorithmFPType, cpu> RPtr(nComponents * ldR);
        TArray<algorithmFPType, cpu> RTPtr(n * ldRT);
        algorithmFPType * R = RPtr.get();
        algorithmFPType * RT = RTPtr.get();
        DAAL_CHECK(R && RT, ErrorMemoryAllocationFailed);
        
        if (svdPar->leftSingularMatrix == requiredInPackedForm)
        {
            UPtr.reset(nComponents * ldU);
            U = UPtr.get();
            DAAL_CHECK(U, ErrorMemoryAllocationFailed);
        }
        
        if (svdPar->rightSingularMatrix == requiredInPackedForm)
        {
            VBlock.set(r[1], 0, nComponents); /* V[nComponents][n] */
            DAAL_CHECK_BLOCK_STATUS(VBlock);
            VT = VBlock.get();
        }

        /* By some reason, there was this part in Sample */
        for (size_t i = 0; i < n * ldRT; i++)
        {
            RT[i] = 0.0;
        }

        // Rc = P*R
        const auto ecQr = compute_QR_on_one_node<algorithmFPType, cpu>(nRowsFull, n, Aux2T, ldAux2, RT, ldRT);
        if (!ecQr) return ecQr;
        
        for (size_t i = 0; i < nComponents; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                R[i * ldR + j] = RT[j * ldRT + i];
            }
        }

        // Qn*R -> Qn*(U'*Sigma*V)
        const auto ecSvd = compute_svd_on_one_node<algorithmFPType, cpu>(n, nComponents, R, ldR, Sigma, VT, ldV, U, ldU);
        if (!ecSvd) return ecSvd;
    }

    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        daal::threader_for(nBlocks, nBlocks, [=, &safeStat](int k) {
            const size_t nRows = a[k]->getNumberOfRows();

            DAAL_ASSERT(nRows == r[2 + k]->getNumberOfRows())
            DAAL_ASSERT(nComponents == r[2 + k]->getNumberOfColumns())

            WriteOnlyRows<algorithmFPType, cpu, NumericTable> aux3Block(r[2 + k], 0, nRows); /* Aux3  [nRows][nComponents] */
            DAAL_CHECK_BLOCK_STATUS_THR(aux3Block);
            algorithmFPType * Aux3 = aux3Block.get();

            const DAAL_INT ldAux3 = nComponents;

            const auto ecGemm = compute_gemm_on_one_node_seq<algorithmFPType, cpu>('N', 'T', nComponents, nRows, nComponents, U, ldU, const_cast<algorithmFPType *>(Aux2T + nRowsOffsets[k]), nRowsFull, Aux3, ldAux3);
            if (!ecGemm)
            {
                safeStat.add(ecGemm);
                return;
            }
        });
        if (!safeStat) return safeStat.detach();
    }

    return Status();
}

} // namespace internal
} // namespace svd
} // namespace algorithms
} // namespace daal

#endif

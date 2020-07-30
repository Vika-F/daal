/* file: kernel_function_linear_csr_fast_impl.i */
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
//  Linear kernel functions implementation
//--
*/

#ifndef __KERNEL_FUNCTION_LINEAR_CSR_FAST_IMPL_I__
#define __KERNEL_FUNCTION_LINEAR_CSR_FAST_IMPL_I__

#include "algorithms/kernel_function/kernel_function_types_linear.h"
#include "src/algorithms/kernel_function/kernel_function_csr_impl.i"

#include "src/threading/threading.h"
#include "src/externals/service_spblas.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace linear
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplLinear<fastCSR, algorithmFPType, cpu>::computeInternalVectorVector(const NumericTable * a1, const NumericTable * a2,
                                                                                              NumericTable * r, const ParameterBase * par)
{
    //prepareData
    ReadRowsCSR<algorithmFPType, cpu> mtA1(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a1)), par->rowIndexX, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA1);
    const size_t * rowOffsetsA1 = mtA1.rows();

    ReadRowsCSR<algorithmFPType, cpu> mtA2(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a2)), par->rowIndexY, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA2);
    const size_t * rowOffsetsA2 = mtA2.rows();

    WriteOnlyRows<algorithmFPType, cpu> mtR(r, par->rowIndexResult, 1);
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType * dataR = mtR.get();

    const Parameter * linPar = static_cast<const Parameter *>(par);

    //compute
    dataR[0] = computeDotProduct(rowOffsetsA1[0] - 1, rowOffsetsA1[1] - 1, mtA1.values(), mtA1.cols(), rowOffsetsA2[0] - 1, rowOffsetsA2[1] - 1,
                                 mtA2.values(), mtA2.cols());
    dataR[0] = dataR[0] * linPar->k + linPar->b;

    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplLinear<fastCSR, algorithmFPType, cpu>::computeInternalMatrixVector(const NumericTable * a1, const NumericTable * a2,
                                                                                              NumericTable * r, const ParameterBase * par)
{
    //prepareData
    const size_t nVectors1 = a1->getNumberOfRows();

    ReadRowsCSR<algorithmFPType, cpu> mtA1(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a1)), 0, nVectors1);
    DAAL_CHECK_BLOCK_STATUS(mtA1);
    const size_t * rowOffsetsA1 = mtA1.rows();

    ReadRowsCSR<algorithmFPType, cpu> mtA2(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a2)), par->rowIndexY, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA2);
    const size_t * rowOffsetsA2 = mtA2.rows();

    WriteOnlyRows<algorithmFPType, cpu> mtR(r, 0, nVectors1);
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType * dataR = mtR.get();

    const Parameter * linPar = static_cast<const Parameter *>(par);
    algorithmFPType b        = (algorithmFPType)(linPar->b);
    algorithmFPType k        = (algorithmFPType)(linPar->k);

    //compute
    for (size_t i = 0; i < nVectors1; i++)
    {
        dataR[i] = computeDotProduct(rowOffsetsA1[i] - 1, rowOffsetsA1[i + 1] - 1, mtA1.values(), mtA1.cols(), rowOffsetsA2[0] - 1,
                                     rowOffsetsA2[1] - 1, mtA2.values(), mtA2.cols());
        dataR[i] = dataR[i] * k + b;
    }

    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplLinear<fastCSR, algorithmFPType, cpu>::computeInternalMatrixMatrix(const NumericTable * a1, const NumericTable * a2,
                                                                                              NumericTable * r, const ParameterBase * par)
{
    //prepareData
    const size_t nVectors1 = a1->getNumberOfRows();
    const size_t nVectors2 = a2->getNumberOfRows();

    ReadRowsCSR<algorithmFPType, cpu> mtA1(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a1)), 0, nVectors1);
    DAAL_CHECK_BLOCK_STATUS(mtA1);
    const algorithmFPType * dataA1 = mtA1.values();
    const size_t * colIndicesA1    = mtA1.cols();
    const size_t * rowOffsetsA1    = mtA1.rows();

    WriteOnlyRows<algorithmFPType, cpu> mtR(r, 0, nVectors1);
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType * dataR = mtR.get();

    const Parameter * linPar = static_cast<const Parameter *>(par);
    algorithmFPType b        = (algorithmFPType)(linPar->b);
    algorithmFPType k        = (algorithmFPType)(linPar->k);

    //compute
    if (a1 == a2)
    {
        const size_t blockSize = 512;
        size_t nBlocks = nVectors1 / blockSize;
        if (nBlocks * blockSize < nVectors1)
            nBlocks++;

        const size_t nnz = rowOffsetsA1[nVectors1] - rowOffsetsA1[0];
        if (!(nnz & 0xffffffff00000000))
        {
            TArray<unsigned int, cpu> colIndicesArr(nnz);
            unsigned int * colIndices = colIndicesArr.get();
            
            const size_t idxBlockSize = 2048;
            size_t nIdxBlocks = nnz / idxBlockSize;
            if (nIdxBlocks * idxBlockSize < nnz)
                nIdxBlocks++;

            daal::threader_for(nIdxBlocks, nIdxBlocks, [=](size_t ibl)
            {
                const size_t iStart = ibl * idxBlockSize;
                const size_t iEnd = (iStart + idxBlockSize < nnz ? iStart + idxBlockSize : nnz);
                    
                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = iStart; i < iEnd; ++i)
                {
                    colIndices[i] = unsigned int(colIndicesA1[i]);
                }
            } );

            daal::tls<algorithmFPType *> threadBuffer(
                [=]() -> algorithmFPType * { return new algorithmFPType[blockSize * blockSize]; });
            daal::threader_for(nBlocks, nBlocks, [=, &threadBuffer](size_t ibl)
            {
                daal::threader_for(ibl + 1, ibl + 1, [=, &threadBuffer](size_t jbl)
                {
                    const size_t iStart = ibl * blockSize;
                    const size_t iEnd = (iStart + blockSize < nVectors1 ? iStart + blockSize : nVectors1);
                    algorithmFPType *res = threadBuffer.local();
                    if (ibl == jbl)
                    {
                        const size_t jStart = iStart;
                        const size_t jEnd = iEnd;
                        for (size_t i = iStart, ii = 0; i < iEnd; ++i, ++ii)
                        {
                            for (size_t j = jStart, jj = 0; j <= i; j++, ++jj)
                            {
                                res[ii * blockSize + jj] = computeDotProduct32bit(rowOffsetsA1[i] - 1, rowOffsetsA1[i + 1] - 1, dataA1, colIndices,
                                                                        rowOffsetsA1[j] - 1, rowOffsetsA1[j + 1] - 1, dataA1, colIndices);
                                res[ii * blockSize + jj] = res[ii * blockSize + jj] * k + b;
                            }
                        }
                        for (size_t i = iStart, ii = 0; i < iEnd; ++i, ++ii)
                        {
                            for (size_t j = jStart, jj = 0; j <= i; j++, ++jj)
                            {
                                dataR[i * nVectors1 + j] = dataR[j * nVectors1 + i] = res[ii * blockSize + jj];
                            }
                        }
                    }
                    else // (ibl != jbl)
                    {
                        const size_t jStart = jbl * blockSize;
                        const size_t jEnd = (jStart + blockSize < nVectors1 ? jStart + blockSize : nVectors1);
                        for (size_t i = iStart; i < iEnd; ++i)
                        {
                            for (size_t j = jStart; j < jEnd; j++)
                            {
                                algorithmFPType res = computeDotProduct32bit(rowOffsetsA1[i] - 1, rowOffsetsA1[i + 1] - 1, dataA1, colIndices,
                                                                        rowOffsetsA1[j] - 1, rowOffsetsA1[j + 1] - 1, dataA1, colIndices);
                                res = res * k + b;
                                dataR[i * nVectors1 + j] = dataR[j * nVectors1 + i] = res;
                            }
                        }
                    }
                } );
            } );
            threadBuffer.reduce([=](algorithmFPType * v) -> void { delete[] (v); });
        }
    }
    else
    {
        ReadRowsCSR<algorithmFPType, cpu> mtA2(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a2)), 0, nVectors2);
        DAAL_CHECK_BLOCK_STATUS(mtA2);
        const algorithmFPType *dataA2 = mtA2.values();
        const size_t *colIndicesA2 = mtA2.cols();
        const size_t *rowOffsetsA2 = mtA2.rows();

        daal::threader_for_optional(nVectors1, nVectors1, [=](size_t i)
        {
            for (size_t j = 0; j < nVectors2; j++)
            {
                dataR[i * nVectors2 + j] = computeDotProduct(rowOffsetsA1[i] - 1, rowOffsetsA1[i + 1] - 1, dataA1, colIndicesA1,
                                                             rowOffsetsA2[j] - 1, rowOffsetsA2[j + 1] - 1, dataA2, colIndicesA2);
                dataR[i * nVectors2 + j] = dataR[i * nVectors2 + j] * k + b;
            }
        } );
    }

    return services::Status();
}

} // namespace internal

} // namespace linear

} // namespace kernel_function

} // namespace algorithms

} // namespace daal

#endif

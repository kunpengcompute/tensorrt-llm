# 算子优化

## insertUnfinishedPathKernel优化

**功能设计**

**insertUnfinishedPathKernel：** 在生成任务结束（或达到最大长度）时，将所有尚未完成（Unfinished）的候选路径保存到最终的候选池（CBA, Candidate Beam Array）中。
查看TensorRT-LLM的默认实现：

```cpp
insertUnfinishedPathKernel<<<bh.nBatchSize, 1, 0, stream>>>(bh);
```

其中Grid Size设置为batch size的大小，Block Size的线程块大小设置为1，每一个Block处理一个独立的请求。但是在当前Beam width不为1的场景，每一个请求内部可以独立为不同的beam，线程块设置为1时存在资源浪费。

**优化点：** 每个Block仍然处理一个batch，但是在Block内部进行beam级别的并行，线程块大小设置为BeamWidth，每个线程处理一个beam。

**功能实现**

- Kernel启动时调整Block的线程块大小

  ```cpp
  dim3 blockSize(bh.nBeamWidth);
  dim3 gridSize(bh.nBatchSize);
  insertUnfinishedPathKernelParallelV1<<<gridSize, blockSize, 0, stream>>>(bh);
  ```

- Kernel内部修改访问逻辑，每个线程处理一个beam

  ```cpp
  ...
  size_t const bid = blockIdx.x;       // Batch index
  size_t const tid = threadIdx.x;      // Beam index within block
  // 每个线程处理一个beam
  int const i = tid;
  int const srcBeam = bid * nBM + i;
  int const dstBeam = bid * nBM * 2 + i + bh.numBeamsCBA[bid];
  int const step = bh.sequenceLengths[srcBeam] - 1;
  ...
  ```

## batchApplyPenalty优化

**功能设计**

**batchApplyPenalty：** 对模型输出的原始 Logits（未归一化的概率分数）应用各种惩罚（Penalties）和调整（Bias/Temperature）。
查看TensorRT-LLM的默认实现：

```cpp
dim3 block(512);
dim3 grid(params.batchSize, params.beamWidth, params.maxTokensPerStep);
batchApplyPenalty<T><<<grid, block, 0, params.stream>>>(params.inputLogits, params.outputLogits, params.biases,
    params.penaltyWorkspace, params.penaltyWorkspacePrev, params.temperatures, params.repetitionPenalties,
    params.presencePenalties, params.frequencyPenalties, params.maxSeqLen, params.vocabSize, params.vocabSizePadded,
    params.outputIdsPtr, params.parentIdsPtr, params.inputLengths, params.sequenceLengths, params.minLengths,
    params.endIds, params.batchSlots, params.tokensPerStep, params.finished);
```

其中maxTokensPerStep在当前场景设置为1，每一个Block处理整个词表，当前场景词表大小为151936，每个Block处理压力过大。

**优化点：** 进行词表分区，每个Block处理部分词表内容，增大Block间并行度

**功能实现**

- 将词表分为VOCAB_BLOCKS个分区

  ```cpp
  constexpr int VOCAB_BLOCKS = 128;
  dim3 block(512);
  dim3 grid(params.batchSize, params.beamWidth, VOCAB_BLOCKS);
  batchApplyPenaltyOpt<T><<<grid, block, 0, params.stream>>>(params.inputLogits, params.outputLogits, params.biases,
    params.penaltyWorkspace, params.penaltyWorkspacePrev, params.temperatures, params.repetitionPenalties,
    params.presencePenalties, params.frequencyPenalties, params.maxSeqLen, params.vocabSize, params.vocabSizePadded,
    params.outputIdsPtr, params.parentIdsPtr, params.inputLengths, params.sequenceLengths, params.minLengths,
    params.endIds, params.batchSlots, params.tokensPerStep, params.finished);
  ```

- 每个Block只处理所属词表分区内容

  ```cpp
  ...
  // 计算vocab分区范围
  SizeType32 vocabPerBlock = (vocabSize + vocabBlocks - 1) / vocabBlocks;
  SizeType32 vocabStart = vocabBlockIdx * vocabPerBlock;
  SizeType32 vocabEnd = min(vocabStart + vocabPerBlock, vocabSize);
  ...
  ```

## updateCacheIndirectionKernel优化

**功能设计**

**updateCacheIndirectionKernel：** 在Beam Search（束搜索）结束时，通过回溯父节点索引，将所有尚未完成的候选路径及其得分，从临时缓冲区完整地还原并搬运到最终的结果数组中。
查看TensorRT-LLM的默认实现：

```cpp
dim3 const grid(common::roundUp(bh.nMaxSeqLen, 32), bh.nBatchSize, bh.nBeamWidthOut);
updateCacheIndirectionKernel<<<grid, 32, 0, stream>>>(tgtCI, srcCI, bh, maxAttentionWindow, sinkTokenLength);
```

其中Block线程块的大小固定为32，在目前的NVIDIA GPU架构中，一个Block（线程块）支持的最大线程数为1024个，存在资源浪费。

**优化点：** 减少Block的数量，增大Block内线程块数量。每一个Block处理更多的数据量，减少需要调度的Block总数。

**功能实现**

- 将`grid size`由`common::roundUp(bh.nMaxSeqLen, 32) *bh.nBatchSize* bh.nBeamWidthOut`减少为`common::roundUp(bh.nMaxSeqLen, PER_STEP) *bh.nBatchSize* (common::roundUp(bh.nBeamWidthOut, BEAMS_PER_BLOCK ) / BEAMS_PER_BLOCK)`
- 将`Block size`由32增加至`BEAMS_PER_BLOCK * PER_STEP`。

```cpp
...
#define BEAMS_PER_BLOCK 96
#define PER_STEP 8
#define THREADS_PER_BLOCK (PER_STEP * BEAMS_PER_BLOCK)
dim3 grid_opt1(common::roundUp(bh.nMaxSeqLen, PER_STEP), bh.nBatchSize, common::roundUp(bh.nBeamWidthOut, BEAMS_PER_BLOCK ) / BEAMS_PER_BLOCK);
updateCacheIndirectionKernelV1<<<grid_opt1, THREADS_PER_BLOCK , 0, stream>>>(tgtCI, srcCI, bh, maxAttentionWindow, sinkTokenLength);
...
```

## masked_multihead_attention_kernel优化

**功能设计**

**masked_multihead_attention_kernel：** masked_multihead_attention实现。
查看TensorRT-LLM的默认实现：

```cpp
...
// The unroll factor for loading from K cache.
unsigned K_LOOP_UNROLL = 4,
// The unroll factor for loading from V cache.
unsigned V_LOOP_UNROLL = 8,
...
#pragma unroll
...
#pragma unroll
...

```

其中存在大量的循环展开，使得每个Block执行kernel时占据大量的寄存器，受寄存器数量影响，导致最终Block并行度下降，GPU的Occupancy达不到最优。

**优化点：** 调整循环展开次数，减少大量的循环展开。

**功能实现**

将K，V循环展开次数调整至2，关闭大量的循环展开。

```cpp
// The unroll factor for loading from K cache.
unsigned K_LOOP_UNROLL = 2,
// The unroll factor for loading from V cache.
unsigned V_LOOP_UNROLL = 2,
...
// #pragma unroll
... 
// #pragma unroll
```

## addBiasSoftMax优化

**功能设计**

**addBiasSoftMax：** 对原始 Logits 进行后处理，将其转化为概率分布（Softmax）或对数概率（Log-Probs），同时应用多种采样策略。
查看TensorRT-LLM的默认实现：

```cpp
...
for (int tid = threadIdx.x; tid < vocabSizePadded; tid += blockDim.x)
    {
        auto logit = logitsPtr[tid];
        logit = temperatures ? logit * tempInv : logit;
        if (tid < vocabSize)
        {
            if (finish && endIds != nullptr)
            {
                // Prefer token EOS if the request has finished
                logit = (tid == endIds[batchSlot]) ? MAX_T_VAL : -MAX_T_VAL;
            }
            else
            {
                // Compute biased logit if the request has not finished, or `endIds` is nullptr
                logit += (bias != nullptr) ? bias[tid] : T{0.0f};
            }
        }
        else
        {
            logit = -MAX_T_VAL;
        }
        maxVal = max(maxVal, static_cast<float>(logit));
        logitsPtr[tid] = logit; // Write back biased logits
    }

for (int tid = threadIdx.x; tid < vocabSizePadded; tid += blockDim.x)
    {
        auto value = __expf(static_cast<float>(logitsPtr[tid]) - sMaxVal);
        // minP : probability of token proportional to the max token
        // compare minP against exp(logit - maxVal) / exp(maxVal - maxVal) = exp(logit - maxVal)
        if (value < minP)
        {
            value = 0.0;
            logitsPtr[tid] = -MAX_T_VAL;
        }
        dst[offset + tid] = value;
        sumVal += value;
    }

    sumVal = blockReduceSum<float>(sumVal);

float entropy{0.f};
for (int tid = threadIdx.x; tid < vocabSizePadded; tid += blockDim.x)
{
    auto const softmaxValue = static_cast<float>(dst[offset + tid]) / (sSumVal + EPSILON);
    auto const probValue = (probs != nullptr) ? softmaxValue : __logf(softmaxValue);
    if (outputEntropy)
    {
        entropy += probValue * __logf(probValue + EPSILON);
    }
    dst[offset + tid] = probValue;
}
...
```

其中softmax采用Safe Softmax，需要循环访问词表3次。

**优化点：** 将Safe Softmax改为Online Softmax，减少一次访存。

**功能实现**

- 定义结构体MD用来保存最大值和指数和，在第一次遍历时每个线程处理多个元素，线程内部维护m和d两个变量，利用公式更新。
- 通过block内规约，将每个线程维护的m和d合并为一个sMaxVal和sSumVal。
- 第二次循环，计算最终的softmax结果。

```cpp

struct __align__(8) MD
{
    float m;
    float d;
};

__device__ __forceinline__ MD reduce_md_op(MD a, MD b)
{
    bool a_bigger = (a.m > b.m);
    MD bigger_m = a_bigger ? a : b;
    MD smaller_m = a_bigger ? b : a;
    MD res;
    res.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    res.m = bigger_m.m;
    return res;
}

MD md_partial;
md_partial.m = -FLT_MAX;
md_partial.d = 0.0F;

for (int tid = threadIdx.x; tid < vocabSizePadded; tid += blockDim.x)
{
    auto logit = logitsPtr[tid];
    logit = temperatures ? logit * tempInv : logit;

    if (tid < vocabSize) {
        if (finish && endIds != nullptr) {
            logit = (tid == endIds[batchSlot]) ? MAX_T_VAL : -MAX_T_VAL;
        } else {
            logit += (bias != nullptr) ? bias[tid] : T{0.0f};
        }
    } else {
        logit = -MAX_T_VAL;
    }

    logitsPtr[tid] = logit;
    MD new_elem;
    new_elem.m = static_cast<float>(logit);
    new_elem.d = 1.0F;
    md_partial = reduce_md_op(md_partial, new_elem);
}

typedef cub::BlockReduce<MD, THREADBLOCK_SIZE> BlockReduce;
__shared__ typename BlockReduce::TempStorage temp_storage;
MD final_md = BlockReduce(temp_storage).Reduce(md_partial, reduce_md_op);

__shared__ float sMaxVal, sSumVal;
if (threadIdx.x == 0) {
    sMaxVal = final_md.m;
    sSumVal = final_md.d;
}

float d_total_inverse = __fdividef(1.0F, sSumVal + EPSILON);
for (int tid = threadIdx.x; tid < vocabSizePadded; tid += blockDim.x)
{
    float val = __expf(static_cast<float>(logitsPtr[tid]) - sMaxVal);
    
    if (val < minP) {
        val = 0.0f;
        logitsPtr[tid] = -MAX_T_VAL; 
    }

    float softmaxValue = val * d_total_inverse;
    float probValue = (probs != nullptr) ? softmaxValue : __logf(softmaxValue);

    if (outputEntropy) {
        entropy += probValue * __logf(probValue + EPSILON);
    }
    dst[offset + tid] = static_cast<T>(probValue);
}

```

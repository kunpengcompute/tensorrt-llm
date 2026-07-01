# Operator Optimization

## insertUnfinishedPathKernel

**Functional Design**

**insertUnfinishedPathKernel:** Saves all unfinished candidate paths to the final candidate pool (Candidate Beam Array, CBA) when the generation task ends (or reaches the maximum length).
View the default implementation of TensorRT-LLM:

```cpp
insertUnfinishedPathKernel<<<bh.nBatchSize, 1, 0, stream>>>(bh);
```

The grid size is set to the batch size, and the block size is set to `1`, meaning each block processes an independent request. However, when the beam width is greater than `1`, each request contains multiple independent beams. Setting the block size to `1` results in underutilized GPU resources.

**Optimization:** Each block still processes one batch, but introduces beam-level parallelism within the block. The block size is set to the beam width, enabling each thread to process a single beam.

**Function Implementation**

- Adjusts the block size at kernel launch:

  ```cpp
  dim3 blockSize(bh.nBeamWidth);
  dim3 gridSize(bh.nBatchSize);
  insertUnfinishedPathKernelParallelV1<<<gridSize, blockSize, 0, stream>>>(bh);
  ```

- Modifies access logic inside the kernel so each thread handles one beam:

  ```cpp
  ...
  size_t const bid = blockIdx.x;       // Batch index
  size_t const tid = threadIdx.x;      // Beam index within block
  // Each thread processes one beam.
  int const i = tid;
  int const srcBeam = bid * nBM + i;
  int const dstBeam = bid * nBM * 2 + i + bh.numBeamsCBA[bid];
  int const step = bh.sequenceLengths[srcBeam] - 1;
  ...
  ```

## batchApplyPenalty

**Functional Design**

**batchApplyPenalty:** Applies various penalties and adjustments (bias/temperature) to the raw logits (unnormalized probability scores) output by the model.
View the default implementation of TensorRT-LLM:

```cpp
dim3 block(512);
dim3 grid(params.batchSize, params.beamWidth, params.maxTokensPerStep);
batchApplyPenalty<T><<<grid, block, 0, params.stream>>>(params.inputLogits, params.outputLogits, params.biases,
    params.penaltyWorkspace, params.penaltyWorkspacePrev, params.temperatures, params.repetitionPenalties,
    params.presencePenalties, params.frequencyPenalties, params.maxSeqLen, params.vocabSize, params.vocabSizePadded,
    params.outputIdsPtr, params.parentIdsPtr, params.inputLengths, params.sequenceLengths, params.minLengths,
    params.endIds, params.batchSlots, params.tokensPerStep, params.finished);
```

In the current scenario, `maxTokensPerStep` is set to `1`, and each block processes the entire vocabulary (151,936 tokens), leading to an excessive workload per block.

**Optimization:** Partitions the vocabulary so that each block processes a segment of it, thereby increasing the inter-block parallelism.

**Function Implementation**

- Divides the vocabulary into `VOCAB_BLOCKS` partitions:

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

- Restricts each block to process only its assigned vocabulary partition:

  ```cpp
  ...
  // Calculate the vocab partition range.
  SizeType32 vocabPerBlock = (vocabSize + vocabBlocks - 1) / vocabBlocks;
  SizeType32 vocabStart = vocabBlockIdx * vocabPerBlock;
  SizeType32 vocabEnd = min(vocabStart + vocabPerBlock, vocabSize);
  ...
  ```

## updateCacheIndirectionKernel

**Functional Design**

**updateCacheIndirectionKernel:** Upon completion of the beam search, backtracks the parent node indices to fully restore and transfer all unfinished candidate paths and their scores from the temporary buffer to the final result array.
View the default implementation of TensorRT-LLM:

```cpp
dim3 const grid(common::roundUp(bh.nMaxSeqLen, 32), bh.nBatchSize, bh.nBeamWidthOut);
updateCacheIndirectionKernel<<<grid, 32, 0, stream>>>(tgtCI, srcCI, bh, maxAttentionWindow, sinkTokenLength);
```

The block size is fixed to `32`. In the current NVIDIA GPU architecture, a single block (thread block) supports a maximum of 1,024 threads. This configuration leaves hardware resources heavily underutilized.

**Optimization:** Reduces the total number of blocks while increasing the number of threads per block. Each block processes a larger volume of data, thereby decreasing the overall number of blocks required for scheduling.

**Function Implementation**

- Reduces the grid size from `common::roundUp(bh.nMaxSeqLen, 32) _bh.nBatchSize_ bh.nBeamWidthOut` to `common::roundUp(bh.nMaxSeqLen, PER_STEP) _bh.nBatchSize_ (common::roundUp(bh.nBeamWidthOut, BEAMS_PER_BLOCK ) / BEAMS_PER_BLOCK)`.
- Increases the block size from `32` to `BEAMS_PER_BLOCK * PER_STEP`.

```cpp
...
#define BEAMS_PER_BLOCK 96
#define PER_STEP 8
#define THREADS_PER_BLOCK (PER_STEP * BEAMS_PER_BLOCK)
dim3 grid_opt1(common::roundUp(bh.nMaxSeqLen, PER_STEP), bh.nBatchSize, common::roundUp(bh.nBeamWidthOut, BEAMS_PER_BLOCK ) / BEAMS_PER_BLOCK);
updateCacheIndirectionKernelV1<<<grid_opt1, THREADS_PER_BLOCK , 0, stream>>>(tgtCI, srcCI, bh, maxAttentionWindow, sinkTokenLength);
...
```

## masked_multihead_attention_kernel

**Functional Design**

**masked_multihead_attention_kernel**: Implements the masked multi-head attention mechanism.
View the default implementation of TensorRT-LLM:

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

The implementation contains excessive loop unrolling, which causes each block to consume a large number of registers during kernel execution. Restricted by register capacity, block-level concurrency drops, preventing the GPU from achieving optimal occupancy.

**Optimization:** Adjusts the number of loop unrollings to reduce excessive loop unrolling.

**Function Implementation**

Tunes K and V unroll factors to `2` and disable excessive compiler unrolling:

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

## addBiasSoftMax

**Functional Design**

**addBiasSoftMax:** Post-processes the original logits, converts them into probability distributions (Softmax) or log-probabilities (Log-Probs), and applies multiple sampling strategies.
View the default implementation of TensorRT-LLM:

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

The softmax implementation relies on Safe Softmax, which requires three separate memory passes over the vocabulary.

**Optimization:** Replaces Safe Softmax with Online Softmax to eliminate one memory pass.

**Function Implementation**

- Defines the `MD` structure to store the maximum value and the exponential sum. During the first traversal, each thread processes multiple elements, maintaining local `m` and `d` variables and updating them using the formula.
- Performs an intra-block reduction to aggregate the `m` and `d` variables maintained by each individual thread into `sMaxVal` and `sSumVal`.
- Executes the second loop to compute the final softmax result.

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

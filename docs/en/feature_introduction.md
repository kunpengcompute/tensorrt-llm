# CUDA Kernel Optimization

## insertUnfinishedPathKernel

**Function:** Saves all unfinished candidate paths to the final candidate pool (Candidate Beam Array, CBA) when the generation task ends (or reaches the maximum length).

**Problem:** In the default TensorRT-LLM implementation, the grid size is set to the batch size, and the block size is set to `1`, with each block processing an independent request. However, when the beam width is greater than `1`, each request contains multiple independent beams. Setting the block size to `1` results in underutilized GPU resources.

**Optimization:** Each block still processes one batch, but introduces beam-level parallelism within the block. The block size is set to the beam width, enabling each thread to process a single beam.

## batchApplyPenalty

**Function:** Applies various penalties and adjustments (bias/temperature) to the raw logits (unnormalized probability scores) output by the model.

**Problem:** In the default TensorRT-LLM implementation, `maxTokensPerStep` is set to `1` for the current scenario. Each block processes the entire vocabulary (151,936 tokens), leading to an excessive processing workload per block.

**Optimization:** Partitions the vocabulary so that each block processes a segment of it, thereby increasing the inter-block parallelism.

## updateCacheIndirectionKernel

**Function:** Upon completion of the beam search, backtracks the parent node indices to fully restore and transfer all unfinished candidate paths and their scores from the temporary buffer to the final result array.

**Problem:** In the default TensorRT-LLM implementation, the block size is set to `32` threads. Since modern NVIDIA GPU architectures support up to 1,024 threads per block, this configuration leaves hardware resources heavily underutilized.

**Optimization:** Reduces the total number of blocks while increasing the number of threads per block. Each block processes a larger volume of data, thereby decreasing the overall number of blocks required for scheduling.

## masked_multihead_attention_kernel

**Problem:** The default TensorRT-LLM implementation relies on excessive loop unrolling, which causes each block to consume too many registers during kernel execution. This high register pressure limits block-level parallelism, preventing the GPU from reaching its optimum occupancy.

**Optimization:** Adjusts the number of loop unrollings to reduce excessive loop unrolling.

## addBiasSoftMax

**Problem:** The native Softmax uses Safe Softmax, which requires three memory passes over the vocabulary.

**Optimization:** Replaces Safe Softmax with Online Softmax to eliminate one memory pass.

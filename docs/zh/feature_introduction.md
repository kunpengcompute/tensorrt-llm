# Cuda Kernel优化

## insertUnfinishedPathKernel

**功能：** 在生成任务结束（或达到最大长度）时，将所有尚未完成（Unfinished）的候选路径保存到最终的候选池（CBA, Candidate Beam Array）中。

**问题：** TensorRT-LLM的默认实现Grid Size设置为batch size的大小，Block Size的线程块大小设置为1，每一个Block处理一个独立的请求。但是在当前Beam width不为1的场景，每一个请求内部可以独立为不同的beam，线程块设置为1时存在资源浪费。

**优化点：** 每个Block仍然处理一个batch，但是在Block内部进行beam级别的并行，线程块大小设置为BeamWidth，每个线程处理一个beam。

## batchApplyPenalty

**功能：** 对模型输出的原始 Logits（未归一化的概率分数）应用各种惩罚（Penalties）和调整（Bias/Temperature）。

**问题：** TensorRT-LLM的默认实现maxTokensPerStep在当前场景设置为1，每一个Block处理整个词表，当前场景词表大小为151936，每个Block处理压力过大。

**优化点：** 进行词表分区，每个Block处理部分词表内容，增大Block间并行度

## updateCacheIndirectionKernel

**功能：** 在 Beam Search（束搜索）结束时，通过回溯父节点索引，将所有尚未完成的候选路径及其得分，从临时缓冲区完整地还原并搬运到最终的结果数组中。

**问题：** TensorRT-LLM的默认实现Block线程块的大小固定为 32，在目前的 NVIDIA GPU 架构中，一个 Block（线程块）支持的最大线程数为 1024 个，存在资源浪费。

**优化点：** 减少Block的数量，增大Block内线程块数量。每一个Block处理更多的数据量，减少需要调度的 Block 总数。

## masked_multihead_attention_kernel

**问题：** TensorRT-LLM的默认实现存在大量的循环展开，使得每个Block执行kernel时占据大量的寄存器，受寄存器数量影响，导致最终Block并行度下降，GPU的Occupancy达不到最优。

**优化点：** 调整循环展开次数，减少大量的循环展开。

## addBiasSoftMax

**问题：** 原生softmax采用Safe Softmax，需要循环访问词表3次。

**优化点：** 将Safe Softmax改为Online Softmax，减少一次访存。

# 安装指南

本文档提供TensorRT-LLM安装方法，介绍了已验证环境要求和编译安装步骤，适用于在鲲鹏服务器安装TensorRT-LLM的用户。

## 已验证环境

为保证您可以顺利安全地使用TensorRT-LLM特性，请确保所使用的环境信息在已验证环境范围内。

**硬件要求<a name="section230mcpsimp"></a>**

已验证的硬件环境如[**表 1** 硬件要求](#硬件要求)所示。

**表 1** 硬件要求<a id="硬件要求"></a>

<a name="_table38928044"></a>

| 项目 | 说明 |
| --- | --- |
| CPU | 鲲鹏950 7592C处理器 |

**操作系统要求<a name="section250mcpsimp"></a>**

已验证的操作系统如[**表 2** 操作系统](#操作系统)所示。

**表 2** 操作系统<a id="操作系统"></a>

<a name="_d0e164"></a>

| 项目 | 版本 | 说明 | 下载地址 |
| --- | --- | --- | --- |
| OS | openEuler 24.03 LTS SP3 | 如果是全新安装操作系统，可选择“Minimal Install”安装方式并勾选Development Tools套件，否则很多软件包需要手动安装。 | [操作系统下载链接](https://repo.openeuler.org/openEuler-24.03-LTS-SP3/ISO/aarch64/) |

**软件要求<a name="section290mcpsimp"></a>**

已验证的软件依赖环境如[**表 3** 软件要求](#软件要求)所示。

**表 3** 软件要求<a id="软件要求"></a>

<a name="_table237115053311"></a>

| 项目 | 版本 | 说明 | 下载地址 |
| --- | --- | --- | --- |
| Python | 3.11.6 | Python是TensorRT-LLM的构建过程中的辅助工具，起到自动下载依赖、配置环境等作用。 | 通过Yum源方式安装。 |
| CMake | 3.22.0 | CMake是TensorRT-LLM的构建工具，要求CMake版本为3.22.0及以上。 | 通过Yum源方式安装。 |
| GCC/G++ | 14.3.1 | GCC（GNU Compiler Collection）是一种编程语言编译器。 | 通过Yum源方式安装。 |
| CUDA | 12.8 | CUDA是一种由NVIDIA推出的并行计算平台和编程模型。 | [CUDA下载链接](https://developer.nvidia.com/cuda-12-8-0-download-archive) |
| torch-cu128 | 2.8.0 | PyTorch的CUDA版本是指与GPU加速兼容的PyTorch构建版本。 | 通过pip源方式安装。 |

## 编译安装

1. 获取TensorRT-LLM开源代码。假设安装路径为“/path/to“。

    ```bash
    cd /path/to
    git clone --branch v1.0.0 https://github.com/NVIDIA/TensorRT-LLM.git
    ```

2. 下载并应用Patch。

    ```bash
    yum install git-lfs
    git lfs install
    git clone https://gitcode.com/boostkit/tensorrt-llm.git k_tensort-llm
    cd TensorRT-LLM
    git submodule update --init --recursive
    git lfs pull
    patch -p1 < ../k_tensort-llm/001-boostsra-tensorrtllm-1.0.0-optimize_kernel.patch
    ```

3. 编译安装

    ```bash
    python3 ./scripts/build_wheel.py --benchmarks
    pip install ./build/tensorrt_llm*.whl
    ```

## 修订记录

|文档版本|发布日期|修改说明|
|--|--|--|
|01|2026-06-30|第一次正式发布。|

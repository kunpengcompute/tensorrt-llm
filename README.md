# TensorRT-LLM介绍

简体中文|[English](./README_en.md)

本文档介绍TensorRT-LLM优化补丁，通过算子优化、访存优化、参数配置等深度性能增强提升模型推理吞吐量和延迟表现，适用于大模型推理场景下的高效执行用户。

## 最新消息

[2026-06-30]：TensorRT-LLM优化补丁发布于Gitcode平台，聚焦于大模型推理场景下的高效执行。

## 项目介绍

本项目基于开源的TensorRT-LLM，聚焦于大模型推理场景下的高效执行。通过算子优化、访存优化、参数配置等进行了深度的性能增强，显著提升了模型推理的吞吐量和延迟表现。

## 目录结构

```bash
tensorrt-llm
├── 001-boostsra-tensorrtllm-1.0.0-optimize_kernel.patch     // TensorRT-LLM补丁文件
├── LICENSE                                                  // License文件
├── README.md                                                // 开源仓介绍
└── docs                                                     // 文档
```

## 版本说明

关于Kunpeng TensorRT-LLM版本更新情况请参见《[版本说明书](./docs/zh/release_notes.md)》。

## 学习文档

| 学习文档名称 | 学习文档简介 |
| --- | --- |
| 《[版本说明书](./docs/zh/release_notes.md)》 | 提供Kunpeng TensorRT-LLM每个发布版本的基础信息和特性更新信息。 |
| 《[特性介绍](./docs/zh/feature_introduction.md)》 | 介绍Kunpeng TensorRT-LLM基于开源TensorRT-LLM的功能点和优化点。 |
| 《[安装指南](./docs/zh/installation_guide.md)》 | 提供Kunpeng TensorRT-LLM编译安装指导。 |

## 免责声明

此代码仓计划参与TensorRT-LLM社区开源，编码风格遵照开源软件，继承开源软件安全设计，不破坏开源软件设计及编码风格和方式，软件的任何漏洞与安全问题，均由相应的上游社区根据其漏洞和安全响应机制解决。请密切关注上游社区发布的通知和版本更新。鲲鹏计算社区对软件的漏洞及安全问题不承担任何责任。

## License

本项目采用Apache License 2.0许可证。具体请参见<a href="./LICENSE">LICENSE</a>文件

本项目文档适用CC-BY 4.0许可证，具体请参见<a href="./docs/LICENSE">LICENSE</a>文件。

## 贡献声明

欢迎大家为社区做贡献，如果使用过程中有任何问题/建议，或者需要反馈特性需求和bug报告，可以提交[Issues](https://gitcode.com/boostkit/community/blob/master/docs/contributor/issue-submit.md)联系我们，具体贡献方法可参考[贡献指南](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。同时也欢迎大家在[讨论专区](https://gitcode.com/boostkit/community/discussions)展开讨论交流。感谢您的支持。

## 致谢

感谢来自社区的每一个PR，欢迎贡献TensorRT-LLM！

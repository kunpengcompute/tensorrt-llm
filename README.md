# TensorRT-LLM介绍<a name="ZH-CN_TOPIC_0000002522619926"></a>

## 项目介绍<a name="ZH-CN_TOPIC_0000002518496736"></a>

本项目基于开源的TensorRT-LLM，聚焦于大模型推理场景下的高效执行。通过算子优化、访存优化、参数配置等进行了深度的性能增强，显著提升了模型推理的吞吐量和时延表现。

## 版本说明<a name="ZH-CN_TOPIC_0000002518656656"></a>
| Kunpeng TensorRT-LLM| TensorRT-LLM |
| --- | --- |
|v1.0.0  |  v1.0.0 |

## 目录结构<a name="ZH-CN_TOPIC_0000002521564878"></a>
 ```
tensorrt-llm
├── 001-boostsra-tensorrtllm-1.0.0-optimize_kernel.patch     // tensorrt-llm补丁文件
├── LICENSE                                                  // License文件
├── README.md                                                // 开源仓介绍
└── docs                                                     // 文档
 ```
## 学习文档<a name="ZH-CN_TOPIC_0000002553555861"></a>

<table>
<thead align="left">
<tr id="row1291816372202">
<th class="cellrowborder" valign="top" width="9.780978097809781%" id="mcps1.1.4.1.1"><p id="p291823714205">学习资源类别</p></th>
<th class="cellrowborder" valign="top" width="17.64176417641764%" id="mcps1.1.4.1.2"><p id="p13918183762016">学习资源名称</p></th>
<th class="cellrowborder" valign="top" width="72.57725772577258%" id="mcps1.1.4.1.3"><p id="p89181437152019">学习资源简介</p></th>
</tr>
</thead>
<tbody>
<tr id="row179181137112015">
<td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1"><p id="p1918123710208">文档</p></td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2"><p id="p2091893722011"><a href="./docs/release_notes.md">版本说明书</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3"><p id="p491893752010">提供TensorRT-LLM每个发布版本的基础信息和特性更新信息。</p></td>
</tr>
<tr id="row179181137112015">
<td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1"><p id="p1918123710208">文档</p></td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2"><p id="p2091893722011"><a href="./docs/feature_guide.md">特性指南</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3"><p id="p491893752010">提供TensorRT-LLM特性介绍。</p></td>
</tr>
<tr id="row939116371143">
<td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1"><p id="p1039163711413">文档</p></td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2"><p id="p03913372046"><a href="./docs/installation_guide.md">安装指南</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3"><p id="p1139217371746">提供TensorRT-LLM编译安装指导。</p></td>
</tr>
</tbody>
</table>

## 免责声明<a name="ZH-CN_TOPIC_0000002550096507"></a>
此代码仓计划参与TensorRT-LLM社区开源，编码风格遵照原生开源软件，继承原生开源软件安全设计，不破坏原生开源软件设计及编码风格和方式，软件的任何漏洞与安全问题，均由相应的上游社区根据其漏洞和安全响应机制解决。请密切关注上游社区发布的通知和版本更新。鲲鹏计算社区对软件的漏洞及安全问题不承担任何责任。
## License<a name="ZH-CN_TOPIC_0000002549976513"></a>
本项目采用Apache License 2.0许可证。详见<a href="./LICENSE">LICENSE</a>文件

本项目文档适用CC-BY 4.0许可证，具体请参见<a href="./docs/LICENSE">LICENSE</a>文件。


## 贡献声明<a name="ZH-CN_TOPIC_0000002549976515"></a>
欢迎大家为社区做贡献，如果使用过程中有任何问题/建议，或者需要反馈特性需求和bug报告，可以提交[Issues](https://gitcode.com/boostkit/community/blob/master/docs/contributor/issue-submit.md)联系我们，具体贡献方法可参考[这里](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。同时也欢迎大家在[讨论专区](https://gitcode.com/boostkit/community/discussions)展开讨论交流。感谢您的支持。
## 致谢<a name="ZH-CN_TOPIC_0000002550096509"></a>
 
TensorRT-LLM由华为公司的下列部门联合贡献：

- 鲲鹏计算Boostkit开发部

感谢来自社区的每一个PR，欢迎贡献TensorRT-LLM！

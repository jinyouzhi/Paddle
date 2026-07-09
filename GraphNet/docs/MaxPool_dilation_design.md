# MaxPool1D/2D/3D Dilation 参数设计文档

## 1. 背景与目标

### 1.1 问题描述
当前PaddlePaddle的MaxPool1D、MaxPool2D、MaxPool3D API不支持dilation（空洞）参数，无法实现空洞池化操作。空洞池化在语义分割、目标检测等任务中具有重要应用价值。

### 1.2 预期效果
新增dilation参数后，用户可以：
```python
# 2D空洞池化示例
import paddle.nn as nn
max_pool = nn.MaxPool2D(kernel_size=3, dilation=2)
output = max_pool(input)

# 1D空洞池化示例
max_pool1d = nn.MaxPool1D(kernel_size=3, dilation=2)
output = max_pool1d(input)
```

## 2. 技术方案

### 2.1 Dilation参数说明
- **参数名称**: `dilation`
- **类型**: int 或 tuple/list
- **默认值**: 1 (无空洞，即标准池化)
- **作用**: 控制池化核元素的采样间隔

### 2.2 数学原理
标准池化（dilation=1）：
```
输出[i] = max(输入[stride*i : stride*i + kernel_size])
```

空洞池化（dilation > 1）：
```
有效kernel_size = (kernel_size - 1) * dilation + 1
输出[i] = max(输入[stride*i : stride*i + 有效kernel_size, 步进=dilation])
```

### 2.3 修改范围

#### 2.3.1 Python层
1. `python/paddle/nn/layer/pooling.py`
   - `MaxPool1D` 类添加 `dilation` 参数
   - `MaxPool2D` 类添加 `dilation` 参数
   - `MaxPool3D` 类添加 `dilation` 参数

2. `python/paddle/nn/functional/pooling.py`
   - `max_pool1d` 函数添加 `dilation` 参数
   - `max_pool2d` 函数添加 `dilation` 参数
   - `max_pool3d` 函数添加 `dilation` 参数

#### 2.3.2 C++ Kernel层
1. `paddle/phi/kernels/pool_kernel.h`
   - 修改 `Pool2dKernel` 签名添加 `dilation` 参数
   - 修改 `Pool3dKernel` 签名添加 `dilation` 参数
   - 新增 `Pool1dKernel` 声明

2. `paddle/phi/kernels/impl/pool_kernel_impl.h`
   - 修改 `PoolRawKernel` 实现支持dilation
   - 新增 `Pool1dKernel` 实现（复用2D实现）

3. `paddle/phi/kernels/cpu/pool_kernel.cc`
   - 注册新Kernel

4. `paddle/phi/kernels/gpu/pool_kernel.cu`
   - 注册新Kernel

5. `paddle/phi/kernels/xpu/pool_kernel.cc`
   - 注册新Kernel

#### 2.3.3 运算符定义
1. `paddle/phi/ops/yaml/ops.yaml`
   - 修改 `pool2d` 操作添加dilation参数
   - 修改 `pool3d` 操作添加dilation参数
   - 新增 `pool1d` 操作定义

2. `paddle/phi/ops/yaml/backward.yaml`
   - 更新反向传播操作

## 3. 实现细节

### 3.1 兼容性设计
- 默认dilation=1，保持向后兼容
- dilation参数必须为正整数
- 对于1D，dilation可以是int或长度为1的list/tuple
- 对于2D，dilation可以是int或长度为2的list/tuple
- 对于3D，dilation可以是int或长度为3的list/tuple

### 3.2 输出尺寸计算
```
output_size = (input_size + 2*padding - ((kernel_size - 1) * dilation + 1)) / stride + 1
```

### 3.3 实现策略
1. **CPU实现**：修改现有的池化函数，在遍历时根据dilation调整步进
2. **GPU实现**：使用cuDNN API（如果支持）或自定义CUDA kernel
3. **XPU实现**：参考CPU实现方式

## 4. 风险与注意事项

### 4.1 兼容性风险
- 保持API向后兼容（dilation默认值为1）
- 不改变现有行为

### 4.2 性能影响
- dilation=1时性能应与现有实现一致
- dilation>1时可能需要额外计算有效kernel size

### 4.3 边界条件
- 需要正确处理padding边界
- 确保dilation值不会导致访问越界

## 5. 测试计划

### 5.1 单元测试
- 基本功能测试
- 边界条件测试
- 梯度计算测试

### 5.2 对比测试
- 与PyTorch实现对比
- 数值正确性验证

### 5.3 性能测试
- 与标准池化性能对比
- 不同dilation值的性能差异

## 6. 时间规划

| 阶段 | 任务 | 预计工作量 |
|------|------|-----------|
| 1 | 设计文档与方案评审 | 2天 |
| 2 | Python层API修改 | 3天 |
| 3 | C++ Kernel层修改（CPU） | 4天 |
| 4 | C++ Kernel层修改（GPU） | 3天 |
| 5 | C++ Kernel层修改（XPU） | 2天 |
| 6 | 单元测试编写 | 3天 |
| 7 | 集成测试与调优 | 3天 |
| **总计** | | **20天** |

## 7. 参考资料

- PyTorch MaxPool Documentation
- ONNX Operators Specification
- 空洞卷积相关论文
- PaddlePaddle现有池化实现
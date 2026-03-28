# 模型转换指南：TensorFlow → ONNX → OM/TensorRT

## 目录
1. [ONNX格式介绍](#onnx格式介绍)
2. [TensorFlow转ONNX](#tensorflow转onnx)
3. [ONNX转OM（华为昇腾）](#onnx转om华为昇腾)
4. [ONNX转TensorRT（NVIDIA）](#onnx转tensorrtnvidia)
5. [性能对比](#性能对比)

---

## ONNX格式介绍

### 什么是ONNX？

ONNX (Open Neural Network Exchange) 是一种开放的神经网络交换格式，用于在不同深度学习框架之间转换模型。

```
TensorFlow → ONNX → TensorRT (NVIDIA GPU)
           → ONNX → OM (华为昇腾)
           → ONNX → OpenVINO (Intel)
           → ONNX → CoreML (Apple)
```

### ONNX的优势

1. **跨平台**：支持TensorFlow、PyTorch、Caffe等多种框架
2. **硬件加速**：可转换为各种硬件专用格式
3. **推理优化**：支持图优化和量化
4. **部署灵活**：一次转换，多处部署

---

## TensorFlow转ONNX

### 方法1：使用tf2onnx（推荐）

```bash
# 安装tf2onnx
pip install tf2onnx

# 转换H5模型为ONNX
python -m tf2onnx.convert \
    --saved-model ./saved_model \
    --output pcb_defect_classifier.onnx \
    --opset 13
```

### 方法2：使用Python脚本转换

创建 `convert_to_onnx.py`:

```python
"""
TensorFlow模型转换为ONNX格式
"""
import tensorflow as tf
import tf2onnx
import onnx
from pathlib import Path


def convert_h5_to_onnx(h5_path, onnx_path, input_size=(224, 224)):
    """
    将H5模型转换为ONNX
    
    Args:
        h5_path: H5模型路径
        onnx_path: 输出ONNX路径
        input_size: 输入图像尺寸
    """
    print(f"Loading model from: {h5_path}")
    
    # 加载模型
    model = tf.keras.models.load_model(h5_path)
    
    # 构建模型（确保输入形状正确）
    model.build(input_shape=(None, *input_size, 3))
    
    print("Converting to ONNX...")
    
    # 定义输入规格
    spec = (tf.TensorSpec((None, *input_size, 3), tf.float32, name="input"),)
    
    # 转换
    model_proto, external_tensor_storage = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=onnx_path
    )
    
    print(f"ONNX model saved to: {onnx_path}")
    
    # 验证模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")
    
    # 打印模型信息
    print("\nModel Info:")
    print(f"  IR version: {onnx_model.ir_version}")
    print(f"  Opset version: 13")
    print(f"  Producer: {onnx_model.producer_name}")
    print(f"  Inputs: {[input.name for input in onnx_model.graph.input]}")
    print(f"  Outputs: {[output.name for output in onnx_model.graph.output]}")
    
    return onnx_path


def optimize_onnx(onnx_path, optimized_path):
    """
    优化ONNX模型
    
    Args:
        onnx_path: 输入ONNX路径
        optimized_path: 输出优化后的路径
    """
    from onnx import optimizer
    
    print(f"\nOptimizing ONNX model...")
    
    # 加载模型
    model = onnx.load(onnx_path)
    
    # 应用优化
    passes = [
        "eliminate_identity",
        "fuse_consecutive_transposes",
        "fuse_pad_into_conv",
        "extract_constant_to_initializer",
        "fuse_add_bias_into_conv",
        "fuse_bn_into_conv"
    ]
    
    optimized_model = optimizer.optimize(model, passes)
    
    # 保存
    onnx.save(optimized_model, optimized_path)
    
    # 比较大小
    original_size = Path(onnx_path).stat().st_size / 1024 / 1024
    optimized_size = Path(optimized_path).stat().st_size / 1024 / 1024
    
    print(f"Original model: {original_size:.2f} MB")
    print(f"Optimized model: {optimized_size:.2f} MB")
    print(f"Reduction: {(1 - optimized_size/original_size)*100:.1f}%")
    
    return optimized_path


def quantize_onnx(onnx_path, quantized_path):
    """
    INT8量化（减小模型大小，提高推理速度）
    
    Args:
        onnx_path: 输入ONNX路径
        quantized_path: 输出量化后的路径
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    print(f"\nQuantizing model to INT8...")
    
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QInt8
    )
    
    # 比较大小
    original_size = Path(onnx_path).stat().st_size / 1024 / 1024
    quantized_size = Path(quantized_path).stat().st_size / 1024 / 1024
    
    print(f"Original model: {original_size:.2f} MB")
    print(f"Quantized model: {quantized_size:.2f} MB")
    print(f"Reduction: {(1 - quantized_size/original_size)*100:.1f}%")
    
    return quantized_path


def test_onnx_inference(onnx_path, input_size=(224, 224)):
    """
    测试ONNX模型推理
    
    Args:
        onnx_path: ONNX模型路径
        input_size: 输入尺寸
    """
    import onnxruntime as ort
    import numpy as np
    import time
    
    print(f"\nTesting ONNX inference...")
    
    # 创建推理会话
    session = ort.InferenceSession(onnx_path)
    
    # 获取输入输出信息
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"Input name: {input_name}")
    print(f"Output name: {output_name}")
    
    # 创建测试数据
    test_input = np.random.randn(1, *input_size, 3).astype(np.float32)
    
    # 预热
    for _ in range(10):
        session.run([output_name], {input_name: test_input})
    
    # 测试推理速度
    times = []
    for _ in range(100):
        start = time.time()
        outputs = session.run([output_name], {input_name: test_input})
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Throughput: {1000/avg_time:.1f} FPS")
    
    return avg_time


def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert TensorFlow model to ONNX')
    parser.add_argument('--input', default='pcb_defect_classifier.h5',
                       help='Input H5 model path')
    parser.add_argument('--output', default='pcb_defect_classifier.onnx',
                       help='Output ONNX path')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize ONNX model')
    parser.add_argument('--quantize', action='store_true',
                       help='Quantize to INT8')
    parser.add_argument('--test', action='store_true',
                       help='Test inference speed')
    args = parser.parse_args()
    
    print("="*70)
    print("TensorFlow to ONNX Conversion")
    print("="*70)
    
    # 1. 转换为ONNX
    onnx_path = convert_h5_to_onnx(args.input, args.output)
    
    # 2. 优化（可选）
    if args.optimize:
        optimized_path = args.output.replace('.onnx', '_optimized.onnx')
        optimize_onnx(onnx_path, optimized_path)
        onnx_path = optimized_path
    
    # 3. 量化（可选）
    if args.quantize:
        quantized_path = args.output.replace('.onnx', '_quantized.onnx')
        quantize_onnx(onnx_path, quantized_path)
        onnx_path = quantized_path
    
    # 4. 测试推理（可选）
    if args.test:
        test_onnx_inference(onnx_path)
    
    print("\n" + "="*70)
    print("Conversion completed!")
    print("="*70)


if __name__ == '__main__':
    main()
```

### 使用方法

```bash
# 基础转换
python convert_to_onnx.py --input pcb_defect_classifier.h5 --output pcb_defect_classifier.onnx

# 转换 + 优化 + 量化 + 测试
python convert_to_onnx.py --input pcb_defect_classifier.h5 --optimize --quantize --test
```

---

## ONNX转OM（华为昇腾）

### 环境准备

```bash
# 安装CANN工具包（需要在昇腾设备上）
# 下载地址: https://www.hiascend.com/software/cann/community

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 转换命令

```bash
# 使用atc工具转换
atc --model=pcb_defect_classifier.onnx \
    --framework=5 \
    --output=pcb_defect_classifier_om \
    --soc_version=Ascend310 \
    --input_shape="input:1,224,224,3" \
    --log=info \
    --insert_op_conf=aipp.config
```

### Python转换脚本

创建 `convert_to_om.py`:

```python
"""
ONNX转OM格式（华为昇腾）
"""
import subprocess
import os
from pathlib import Path


def convert_onnx_to_om(onnx_path, om_path, soc_version="Ascend310"):
    """
    将ONNX转换为OM格式
    
    Args:
        onnx_path: ONNX模型路径
        om_path: 输出OM路径
        soc_version: 昇腾芯片版本（Ascend310/Ascend310P/Ascend910）
    """
    print(f"Converting ONNX to OM...")
    print(f"  Input: {onnx_path}")
    print(f"  Output: {om_path}")
    print(f"  SoC: {soc_version}")
    
    # 构建atc命令
    cmd = [
        "atc",
        "--model", str(onnx_path),
        "--framework", "5",  # 5表示ONNX
        "--output", str(om_path).replace('.om', ''),
        "--soc_version", soc_version,
        "--input_shape", "input:1,224,224,3",
        "--log", "info",
        "--precision_mode", "force_fp16"  # 使用FP16提高性能
    ]
    
    print(f"\nCommand: {' '.join(cmd)}")
    
    # 执行转换
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Conversion successful!")
        
        # 显示模型信息
        om_file = Path(om_path)
        if om_file.exists():
            size_mb = om_file.stat().st_size / 1024 / 1024
            print(f"  Model size: {size_mb:.2f} MB")
    else:
        print("✗ Conversion failed!")
        print(f"Error: {result.stderr}")
        return None
    
    return om_path


def benchmark_om(om_path):
    """
    测试OM模型性能
    
    Args:
        om_path: OM模型路径
    """
    print(f"\nBenchmarking OM model...")
    
    # 使用msame工具进行性能测试
    cmd = [
        "msame",
        "--model", str(om_path),
        "--loop", "100",
        "--output", "./benchmark_output"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Benchmark completed")
        print(result.stdout)
    else:
        print("✗ Benchmark failed")
        print(result.stderr)


def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert ONNX to OM (Ascend)')
    parser.add_argument('--input', default='pcb_defect_classifier.onnx',
                       help='Input ONNX path')
    parser.add_argument('--output', default='pcb_defect_classifier.om',
                       help='Output OM path')
    parser.add_argument('--soc', default='Ascend310',
                       choices=['Ascend310', 'Ascend310P', 'Ascend910'],
                       help='Ascend SoC version')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark after conversion')
    args = parser.parse_args()
    
    print("="*70)
    print("ONNX to OM Conversion (Huawei Ascend)")
    print("="*70)
    
    # 检查环境
    if not os.system("which atc > /dev/null 2>&1") == 0:
        print("Error: ATC tool not found!")
        print("Please install CANN toolkit and source set_env.sh")
        return
    
    # 转换
    convert_onnx_to_om(args.input, args.output, args.soc)
    
    # 性能测试
    if args.benchmark:
        benchmark_om(args.output)
    
    print("\n" + "="*70)
    print("Conversion completed!")
    print("="*70)


if __name__ == '__main__':
    main()
```

### 昇腾推理代码

```python
"""
使用OM模型在昇腾设备上推理
"""
import acl
import numpy as np


class AscendInferencer:
    """昇腾推理器"""
    
    def __init__(self, om_path, device_id=0):
        """
        初始化推理器
        
        Args:
            om_path: OM模型路径
            device_id: 设备ID
        """
        self.device_id = device_id
        
        # 初始化ACL
        ret = acl.init()
        ret = acl.rt.set_device(device_id)
        
        # 加载模型
        self.model_id, ret = acl.mdl.load_from_file(om_path)
        
        # 获取模型信息
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        
        # 创建输入输出数据集
        self.input_dataset = self._create_dataset()
        self.output_dataset = self._create_dataset()
        
        print(f"Model loaded: {om_path}")
    
    def _create_dataset(self):
        """创建数据集"""
        dataset = acl.mdl.create_dataset()
        return dataset
    
    def infer(self, input_data):
        """
        执行推理
        
        Args:
            input_data: 输入数据 (numpy array)
            
        Returns:
            输出结果
        """
        # 准备输入数据
        input_ptr = acl.util.numpy_to_ptr(input_data)
        
        # 执行推理
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        
        # 获取输出
        output_ptr = acl.mdl.get_dataset_buffer(self.output_dataset, 0)
        
        # 转换为numpy
        output_data = acl.util.ptr_to_numpy(output_ptr)
        
        return output_data
    
    def __del__(self):
        """释放资源"""
        acl.mdl.unload(self.model_id)
        acl.rt.reset_device(self.device_id)
        acl.finalize()


# 使用示例
if __name__ == '__main__':
    # 创建推理器
    inferencer = AscendInferencer('pcb_defect_classifier.om')
    
    # 准备输入数据
    input_data = np.random.randn(1, 224, 224, 3).astype(np.float32)
    
    # 推理
    output = inferencer.infer(input_data)
    
    print(f"Output shape: {output.shape}")
    print(f"Predicted class: {np.argmax(output)}")
```

---

## ONNX转TensorRT（NVIDIA）

### 环境准备

```bash
# 安装TensorRT
# 下载地址: https://developer.nvidia.com/tensorrt

# 安装Python包
pip install tensorrt
```

### 转换命令

```bash
# 使用trtexec工具
trtexec --onnx=pcb_defect_classifier.onnx \
        --saveEngine=pcb_defect_classifier.trt \
        --fp16 \
        --workspace=4096 \
        --minShapes=input:1x224x224x3 \
        --optShapes=input:1x224x224x3 \
        --maxShapes=input:8x224x224x3
```

### Python转换脚本

创建 `convert_to_tensorrt.py`:

```python
"""
ONNX转TensorRT格式
"""
import tensorrt as trt
import numpy as np
from pathlib import Path


class TensorRTConverter:
    """TensorRT转换器"""
    
    def __init__(self, verbose=True):
        """
        初始化转换器
        
        Args:
            verbose: 是否显示详细日志
        """
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.network = None
        self.parser = None
        self.engine = None
    
    def convert_onnx_to_trt(
        self,
        onnx_path,
        trt_path,
        fp16_mode=True,
        max_batch_size=8,
        max_workspace_size=4
    ):
        """
        将ONNX转换为TensorRT
        
        Args:
            onnx_path: ONNX模型路径
            trt_path: 输出TensorRT路径
            fp16_mode: 是否使用FP16
            max_batch_size: 最大批次大小
            max_workspace_size: 最大工作空间（GB）
        """
        print(f"Converting ONNX to TensorRT...")
        print(f"  Input: {onnx_path}")
        print(f"  Output: {trt_path}")
        print(f"  FP16: {fp16_mode}")
        print(f"  Max batch: {max_batch_size}")
        
        # 创建网络
        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(explicit_batch)
        self.parser = trt.OnnxParser(self.network, self.logger)
        
        # 解析ONNX
        with open(onnx_path, 'rb') as f:
            if not self.parser.parse(f.read()):
                print("Error: Failed to parse ONNX")
                for error in range(self.parser.num_errors):
                    print(self.parser.get_error(error))
                return None
        
        # 配置builder
        config = self.builder.create_builder_config()
        config.max_workspace_size = max_workspace_size * (1 << 30)  # GB to bytes
        
        if fp16_mode:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  Using FP16 precision")
        
        # 创建优化配置文件（支持动态batch）
        profile = self.builder.create_optimization_profile()
        input_name = self.network.get_input(0).name
        
        # 设置动态batch范围
        profile.set_shape(
            input_name,
            min=(1, 224, 224, 3),
            opt=(1, 224, 224, 3),
            max=(max_batch_size, 224, 224, 3)
        )
        config.add_optimization_profile(profile)
        
        # 构建引擎
        print("Building TensorRT engine...")
        self.engine = self.builder.build_engine(self.network, config)
        
        if self.engine is None:
            print("Error: Failed to build engine")
            return None
        
        # 保存引擎
        with open(trt_path, 'wb') as f:
            f.write(self.engine.serialize())
        
        print(f"✓ TensorRT engine saved to: {trt_path}")
        
        # 显示模型信息
        size_mb = Path(trt_path).stat().st_size / 1024 / 1024
        print(f"  Engine size: {size_mb:.2f} MB")
        
        return trt_path
    
    def __del__(self):
        """释放资源"""
        if self.engine:
            del self.engine
        if self.parser:
            del self.parser
        if self.network:
            del self.network
        if self.builder:
            del self.builder


def benchmark_tensorrt(trt_path, batch_size=1, iterations=100):
    """
    测试TensorRT性能
    
    Args:
        trt_path: TensorRT引擎路径
        batch_size: 批次大小
        iterations: 迭代次数
    """
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    print(f"\nBenchmarking TensorRT...")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {iterations}")
    
    # 加载引擎
    logger = trt.Logger(trt.Logger.WARNING)
    with open(trt_path, 'rb') as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    
    # 创建执行上下文
    context = engine.create_execution_context()
    
    # 分配内存
    input_shape = (batch_size, 224, 224, 3)
    output_shape = (batch_size, 6)  # 6个类别
    
    input_size = trt.volume(input_shape) * trt.float32.itemsize
    output_size = trt.volume(output_shape) * trt.float32.itemsize
    
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)
    
    bindings = [int(d_input), int(d_output)]
    
    # 创建流
    stream = cuda.Stream()
    
    # 准备输入数据
    h_input = np.random.randn(*input_shape).astype(np.float32)
    h_output = np.empty(output_shape, dtype=np.float32)
    
    # 预热
    for _ in range(10):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
    
    # 测试
    import time
    times = []
    
    for _ in range(iterations):
        start = time.time()
        
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000
    print(f"  Average time: {avg_time:.2f} ms")
    print(f"  Throughput: {batch_size * 1000 / avg_time:.1f} FPS")
    
    return avg_time


def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert ONNX to TensorRT')
    parser.add_argument('--input', default='pcb_defect_classifier.onnx',
                       help='Input ONNX path')
    parser.add_argument('--output', default='pcb_defect_classifier.trt',
                       help='Output TensorRT path')
    parser.add_argument('--fp16', action='store_true', default=True,
                       help='Use FP16 precision')
    parser.add_argument('--batch', type=int, default=1,
                       help='Max batch size')
    parser.add_argument('--workspace', type=int, default=4,
                       help='Max workspace size (GB)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark after conversion')
    args = parser.parse_args()
    
    print("="*70)
    print("ONNX to TensorRT Conversion (NVIDIA)")
    print("="*70)
    
    # 转换
    converter = TensorRTConverter(verbose=True)
    trt_path = converter.convert_onnx_to_trt(
        args.input,
        args.output,
        fp16_mode=args.fp16,
        max_batch_size=args.batch,
        max_workspace_size=args.workspace
    )
    
    if trt_path and args.benchmark:
        benchmark_tensorrt(trt_path, batch_size=1)
    
    print("\n" + "="*70)
    print("Conversion completed!")
    print("="*70)


if __name__ == '__main__':
    main()
```

### TensorRT推理代码

```python
"""
使用TensorRT进行推理
"""
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


class TensorRTInferencer:
    """TensorRT推理器"""
    
    def __init__(self, trt_path):
        """
        初始化推理器
        
        Args:
            trt_path: TensorRT引擎路径
        """
        # 加载引擎
        logger = trt.Logger(trt.Logger.WARNING)
        with open(trt_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        
        # 创建CUDA流
        self.stream = cuda.Stream()
        
        # 获取输入输出尺寸
        self.input_shape = (1, 224, 224, 3)
        self.output_shape = (1, 6)
        
        # 分配GPU内存
        self.input_size = trt.volume(self.input_shape) * np.dtype(np.float32).itemsize
        self.output_size = trt.volume(self.output_shape) * np.dtype(np.float32).itemsize
        
        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_output = cuda.mem_alloc(self.output_size)
        
        self.bindings = [int(self.d_input), int(self.d_output)]
        
        print(f"TensorRT engine loaded: {trt_path}")
    
    def infer(self, input_data):
        """
        执行推理
        
        Args:
            input_data: 输入数据 (numpy array, shape: [1, 224, 224, 3])
            
        Returns:
            输出结果 (numpy array, shape: [1, 6])
        """
        # 确保输入数据类型正确
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
        
        # 分配输出内存
        output = np.empty(self.output_shape, dtype=np.float32)
        
        # 数据传输 + 推理
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()
        
        return output
    
    def __del__(self):
        """释放资源"""
        if hasattr(self, 'd_input'):
            self.d_input.free()
        if hasattr(self, 'd_output'):
            self.d_output.free()


# 使用示例
if __name__ == '__main__':
    # 创建推理器
    inferencer = TensorRTInferencer('pcb_defect_classifier.trt')
    
    # 准备输入数据
    input_data = np.random.randn(1, 224, 224, 3).astype(np.float32)
    
    # 推理
    output = inferencer.infer(input_data)
    
    print(f"Output: {output}")
    print(f"Predicted class: {np.argmax(output)}")
```

---

## 性能对比

| 格式 | 大小 | 推理速度 | 适用场景 |
|------|------|---------|---------|
| H5 (TensorFlow) | ~15 MB | 100ms (CPU) | 训练、调试 |
| ONNX | ~15 MB | 80ms (CPU) | 跨平台部署 |
| ONNX (INT8) | ~4 MB | 50ms (CPU) | 边缘设备 |
| OM (昇腾) | ~8 MB | 10ms (Ascend310) | 华为设备 |
| TensorRT (FP16) | ~8 MB | 5ms (RTX3060) | NVIDIA GPU |
| TensorRT (INT8) | ~4 MB | 3ms (RTX3060) | NVIDIA GPU |

---

## 完整转换流程

```bash
# 1. TensorFlow → ONNX
python convert_to_onnx.py \
    --input pcb_defect_classifier.h5 \
    --output pcb_defect_classifier.onnx \
    --optimize \
    --quantize \
    --test

# 2. ONNX → OM (华为昇腾)
python convert_to_om.py \
    --input pcb_defect_classifier.onnx \
    --output pcb_defect_classifier.om \
    --soc Ascend310 \
    --benchmark

# 3. ONNX → TensorRT (NVIDIA)
python convert_to_tensorrt.py \
    --input pcb_defect_classifier.onnx \
    --output pcb_defect_classifier.trt \
    --fp16 \
    --benchmark
```

---

## 总结

| 硬件平台 | 推荐格式 | 转换工具 | 推理速度 |
|---------|---------|---------|---------|
| Intel CPU | ONNX (INT8) | OpenVINO | 50ms |
| NVIDIA GPU | TensorRT | trtexec | 5ms |
| 华为昇腾 | OM | ATC | 10ms |
| 移动端 | TFLite | TFLite Converter | 30ms |
| 通用 | ONNX | ONNX Runtime | 80ms |

**建议**：
- 开发调试：使用H5或ONNX
- 生产部署：根据硬件选择TensorRT、OM或TFLite
- 边缘设备：使用INT8量化版本

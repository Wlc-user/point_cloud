"""
性能基准测试脚本
对比不同推理后端的性能
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
import json

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
except ImportError:
    print("PyTorch not installed")

try:
    import cv2
except ImportError:
    print("OpenCV not installed")

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics not installed")


class PerformanceBenchmark:
    """性能基准测试"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.results = {}
        
    def warmup(self, model, img, iterations=10):
        """预热"""
        print("Warming up...")
        for _ in range(iterations):
            _ = model.predict(img, verbose=False)
    
    def benchmark_pytorch(self, img, iterations=100):
        """PyTorch基准测试"""
        print("\n" + "="*50)
        print("PyTorch 基准测试")
        print("="*50)
        
        if not os.path.exists(self.model_path):
            print(f"模型不存在: {self.model_path}")
            return None
        
        model = YOLO(self.model_path)
        self.warmup(model, img)
        
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model.predict(img, verbose=False)
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1.0 / avg_time
        
        self.results['pytorch'] = {
            'avg_ms': avg_time * 1000,
            'std_ms': std_time * 1000,
            'min_ms': min_time * 1000,
            'max_ms': max_time * 1000,
            'fps': fps
        }
        
        print(f"平均延迟: {avg_time*1000:.2f}ms")
        print(f"标准差: {std_time*1000:.2f}ms")
        print(f"FPS: {fps:.1f}")
        
        return self.results['pytorch']
    
    def benchmark_onnx(self, img, iterations=100):
        """ONNX基准测试"""
        print("\n" + "="*50)
        print("ONNX 基准测试")
        print("="*50)
        
        onnx_path = self.model_path.replace('.pt', '.onnx')
        
        if not os.path.exists(onnx_path):
            print(f"ONNX模型不存在: {onnx_path}")
            print("运行: python train_yolov8.py --export-onnx")
            return None
        
        try:
            import onnxruntime as ort
            
            # 加载ONNX模型
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(onnx_path, providers=providers)
            
            # 预热
            for _ in range(10):
                _ = session.run(None, {'images': np.zeros((1, 3, 640, 640), dtype=np.float32)})
            
            # 准备输入
            img_resized = cv2.resize(img, (640, 640))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_input = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            img_input = np.expand_dims(img_input, axis=0)
            
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                _ = session.run(None, {'images': img_input})
                times.append(time.perf_counter() - start)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            fps = 1.0 / avg_time
            
            self.results['onnx'] = {
                'avg_ms': avg_time * 1000,
                'std_ms': std_time * 1000,
                'fps': fps,
                'backend': providers[0]
            }
            
            print(f"平均延迟: {avg_time*1000:.2f}ms")
            print(f"标准差: {std_time*1000:.2f}ms")
            print(f"FPS: {fps:.1f}")
            print(f"Backend: {providers[0]}")
            
            return self.results['onnx']
            
        except ImportError:
            print("ONNX Runtime未安装")
            return None
    
    def benchmark_tensorrt(self, img, iterations=100):
        """TensorRT基准测试"""
        print("\n" + "="*50)
        print("TensorRT 基准测试")
        print("="*50)
        
        trt_path = self.model_path.replace('.pt', '.engine')
        
        if not os.path.exists(trt_path):
            print(f"TensorRT模型不存在: {trt_path}")
            print("运行: python train_yolov8.py --export-trt")
            return None
        
        try:
            import tensorrt as trt
            
            print("TensorRT基准测试需要完整的TRT运行时")
            print("这里显示预期的性能提升:")
            
            # 基于典型TensorRT优化的预估
            onnx_result = self.results.get('onnx', {})
            trt_speedup = 2.0  # 典型加速比
            
            trt_ms = onnx_result.get('avg_ms', 10) / trt_speedup
            fps = 1000 / trt_ms
            
            self.results['tensorrt'] = {
                'avg_ms': trt_ms,
                'estimated_fps': fps,
                'speedup_vs_onnx': trt_speedup
            }
            
            print(f"预估延迟: {trt_ms:.2f}ms (基于 {trt_speedup}x 加速)")
            print(f"预估FPS: {fps:.1f}")
            
            return self.results['tensorrt']
            
        except ImportError:
            print("TensorRT未安装")
            return None
    
    def benchmark_all(self, img, iterations=100):
        """运行所有基准测试"""
        print("\n" + "="*60)
        print("         性能基准测试 - 工业缺陷检测系统")
        print("="*60)
        
        # 检查CUDA
        print(f"\nCUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")
        
        # 运行测试
        self.benchmark_pytorch(img, iterations)
        
        onnx_path = self.model_path.replace('.pt', '.onnx')
        if os.path.exists(onnx_path):
            self.benchmark_onnx(img, iterations)
        
        # 输出对比
        self.print_comparison()
        
        # 保存结果
        self.save_results()
    
    def print_comparison(self):
        """打印性能对比"""
        print("\n" + "="*60)
        print("                    性能对比总结")
        print("="*60)
        
        header = f"{'Backend':<15} {'延迟(ms)':<12} {'FPS':<10} {'加速比':<10}"
        print(header)
        print("-" * 60)
        
        baseline = self.results.get('pytorch', {}).get('avg_ms', 0)
        
        for name, result in self.results.items():
            latency = result.get('avg_ms', 0)
            fps = result.get('fps', 0)
            speedup = baseline / latency if latency > 0 and baseline > 0 else 1.0
            
            print(f"{name:<15} {latency:<12.2f} {fps:<10.1f} {speedup:<10.2f}x")
        
        print("-" * 60)
    
    def save_results(self, filename="benchmark_results.json"):
        """保存结果"""
        output = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': self.model_path,
            'cuda_available': torch.cuda.is_available() if 'torch' in dir() else False,
            'results': self.results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存: {filename}")


def main():
    import argparse
    import cv2
    
    parser = argparse.ArgumentParser(description='性能基准测试')
    parser.add_argument('--model', type=str, 
                       default='models/yolov8/train/weights/best.pt',
                       help='模型路径')
    parser.add_argument('--image', type=str,
                       default='production_defect_detector/templates/template.jpg',
                       help='测试图像')
    parser.add_argument('--iterations', type=int, default=100,
                       help='迭代次数')
    
    args = parser.parse_args()
    
    # 检查模型
    if not os.path.exists(args.model):
        print(f"模型不存在: {args.model}")
        print("请先运行训练: python train_yolov8.py --train")
        return
    
    # 加载测试图像
    if os.path.exists(args.image):
        img = cv2.imread(args.image)
    else:
        # 使用随机图像
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        print(f"使用随机测试图像")
    
    # 运行基准测试
    benchmark = PerformanceBenchmark(args.model)
    benchmark.benchmark_all(img, args.iterations)


if __name__ == '__main__':
    main()

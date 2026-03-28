"""
PCB缺陷数据集管理器
用于加载、处理和分析真实的PCB缺陷数据
"""
import os
import cv2
import numpy as np
import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import matplotlib.pyplot as plt
import random


@dataclass
class PCBDefect:
    """PCB缺陷数据类"""
    id: str
    defect_type: str
    image_path: str
    pcb_id: str
    severity: str = "unknown"
    
    def to_dict(self):
        return asdict(self)


class PCBDatasetManager:
    """PCB数据集管理器"""
    
    # 缺陷类型映射
    DEFECT_TYPES = {
        'missing_hole': '缺孔',
        'mouse_bite': '鼠咬',
        'open_circuit': '开路',
        'short': '短路',
        'spur': '毛刺',
        'spurious_copper': '多余铜'
    }
    
    def __init__(self, dataset_path: str = "real_images/pcb/images"):
        self.dataset_path = Path(dataset_path)
        self.defects = []
        self.statistics = {}
        
    def load_dataset(self) -> List[PCBDefect]:
        """加载整个数据集"""
        print("Loading PCB defect dataset...")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        
        self.defects = []
        
        # 遍历所有图片文件
        for img_file in sorted(self.dataset_path.glob("*.jpg")):
            # 解析文件名: {pcb_id}_{defect_type}_{number}.jpg
            parts = img_file.stem.split('_')
            
            if len(parts) >= 3:
                pcb_id = parts[0]
                # 处理多单词缺陷类型 (如 spurious_copper)
                defect_type = '_'.join(parts[1:-1])
                number = parts[-1]
                
                defect = PCBDefect(
                    id=f"{pcb_id}_{defect_type}_{number}",
                    defect_type=defect_type,
                    image_path=str(img_file),
                    pcb_id=pcb_id
                )
                self.defects.append(defect)
        
        print(f"Loaded {len(self.defects)} defect images")
        self.calculate_statistics()
        return self.defects
    
    def calculate_statistics(self):
        """计算数据集统计信息"""
        self.statistics = {
            'total_images': len(self.defects),
            'total_pcbs': len(set(d.pcb_id for d in self.defects)),
            'defect_types': defaultdict(int),
            'defects_per_pcb': defaultdict(int)
        }
        
        for defect in self.defects:
            self.statistics['defect_types'][defect.defect_type] += 1
            self.statistics['defects_per_pcb'][defect.pcb_id] += 1
        
        return self.statistics
    
    def get_defects_by_type(self, defect_type: str) -> List[PCBDefect]:
        """按类型获取缺陷"""
        return [d for d in self.defects if d.defect_type == defect_type]
    
    def get_defects_by_pcb(self, pcb_id: str) -> List[PCBDefect]:
        """按PCB ID获取缺陷"""
        return [d for d in self.defects if d.pcb_id == pcb_id]
    
    def get_defect_image(self, defect: PCBDefect) -> Optional[np.ndarray]:
        """加载缺陷图像"""
        img = cv2.imread(defect.image_path)
        if img is not None:
            # 转换为RGB用于显示
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                     split_by='image') -> Tuple[List, List, List]:
        """
        划分训练/验证/测试集
        split_by: 'image' - 按图像划分, 'pcb' - 按PCB板划分
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        if split_by == 'pcb':
            # 按PCB板划分（确保同一PCB的缺陷不会分散在不同集合）
            pcbs = list(set(d.pcb_id for d in self.defects))
            random.shuffle(pcbs)
            
            n_train = int(len(pcbs) * train_ratio)
            n_val = int(len(pcbs) * val_ratio)
            
            train_pcbs = set(pcbs[:n_train])
            val_pcbs = set(pcbs[n_train:n_train+n_val])
            test_pcbs = set(pcbs[n_train+n_val:])
            
            train_set = [d for d in self.defects if d.pcb_id in train_pcbs]
            val_set = [d for d in self.defects if d.pcb_id in val_pcbs]
            test_set = [d for d in self.defects if d.pcb_id in test_pcbs]
        else:
            # 按图像划分
            shuffled = self.defects.copy()
            random.shuffle(shuffled)
            
            n_train = int(len(shuffled) * train_ratio)
            n_val = int(len(shuffled) * val_ratio)
            
            train_set = shuffled[:n_train]
            val_set = shuffled[n_train:n_train+n_val]
            test_set = shuffled[n_train+n_val:]
        
        print(f"Dataset split:")
        print(f"  Train: {len(train_set)} images ({len(train_set)/len(self.defects)*100:.1f}%)")
        print(f"  Val: {len(val_set)} images ({len(val_set)/len(self.defects)*100:.1f}%)")
        print(f"  Test: {len(test_set)} images ({len(test_set)/len(self.defects)*100:.1f}%)")
        
        return train_set, val_set, test_set
    
    def export_to_yolo_format(self, output_dir: str = "yolo_dataset"):
        """导出为YOLO格式（用于目标检测训练）"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 创建目录结构
        for split in ['train', 'val', 'test']:
            (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # 划分数据集
        train_set, val_set, test_set = self.split_dataset()
        
        # 生成类别映射
        class_names = list(self.DEFECT_TYPES.keys())
        class_map = {name: i for i, name in enumerate(class_names)}
        
        # 保存类别名称
        with open(output_path / 'classes.txt', 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")
        
        # 处理每个集合
        for split_name, dataset in [('train', train_set), ('val', val_set), ('test', test_set)]:
            for defect in dataset:
                # 复制图像
                img = self.get_defect_image(defect)
                if img is not None:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img_path = output_path / 'images' / split_name / f"{defect.id}.jpg"
                    cv2.imwrite(str(img_path), img_bgr)
                    
                    # 生成YOLO标注文件
                    # 注意：这里假设整张图就是一个缺陷区域
                    # 实际使用时需要根据真实标注调整
                    h, w = img.shape[:2]
                    class_id = class_map.get(defect.defect_type, 0)
                    
                    # 使用整张图作为边界框（需要根据实际情况调整）
                    x_center, y_center = 0.5, 0.5
                    width, height = 1.0, 1.0
                    
                    label_path = output_path / 'labels' / split_name / f"{defect.id}.txt"
                    with open(label_path, 'w') as f:
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        
        # 生成data.yaml
        yaml_content = f"""train: {output_path}/images/train
val: {output_path}/images/val
test: {output_path}/images/test

nc: {len(class_names)}
names: {class_names}
"""
        with open(output_path / 'data.yaml', 'w') as f:
            f.write(yaml_content)
        
        print(f"YOLO format dataset exported to: {output_dir}")
        return output_path
    
    def visualize_dataset(self, num_samples_per_type=5, save_path: str = None):
        """可视化数据集样本"""
        defect_types = list(self.statistics['defect_types'].keys())
        n_types = len(defect_types)
        
        fig, axes = plt.subplots(n_types, num_samples_per_type, 
                                figsize=(num_samples_per_type*3, n_types*3))
        
        if n_types == 1:
            axes = axes.reshape(1, -1)
        
        for i, defect_type in enumerate(defect_types):
            defects = self.get_defects_by_type(defect_type)
            samples = random.sample(defects, min(num_samples_per_type, len(defects)))
            
            for j, defect in enumerate(samples):
                img = self.get_defect_image(defect)
                if img is not None:
                    axes[i, j].imshow(img)
                    axes[i, j].set_title(f"{defect_type}\n{defect.pcb_id}", fontsize=8)
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "="*60)
        print("PCB Defect Dataset Statistics")
        print("="*60)
        print(f"Total Images: {self.statistics['total_images']}")
        print(f"Total PCBs: {self.statistics['total_pcbs']}")
        print(f"\nDefect Type Distribution:")
        
        for defect_type, count in sorted(self.statistics['defect_types'].items()):
            percentage = count / self.statistics['total_images'] * 100
            chinese_name = self.DEFECT_TYPES.get(defect_type, defect_type)
            print(f"  {defect_type:20s} ({chinese_name:10s}): {count:4d} ({percentage:5.1f}%)")
        
        print("="*60)
    
    def export_statistics(self, output_file: str = "dataset_statistics.json"):
        """导出统计信息到JSON"""
        stats_dict = {
            'total_images': self.statistics['total_images'],
            'total_pcbs': self.statistics['total_pcbs'],
            'defect_types': dict(self.statistics['defect_types']),
            'defects_per_pcb': dict(self.statistics['defects_per_pcb'])
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Statistics exported to: {output_file}")


def main():
    """主程序"""
    print("PCB Defect Dataset Manager")
    print("="*60)
    
    # 创建管理器
    manager = PCBDatasetManager("real_images/pcb/images")
    
    # 加载数据集
    manager.load_dataset()
    
    # 打印统计信息
    manager.print_statistics()
    
    # 导出统计信息
    manager.export_statistics("pcb_dataset_stats.json")
    
    # 可视化样本
    print("\nGenerating visualization...")
    manager.visualize_dataset(num_samples_per_type=5, 
                             save_path="pcb_dataset_samples.png")
    
    # 导出YOLO格式（可选）
    print("\nExporting to YOLO format...")
    manager.export_to_yolo_format("yolo_pcb_dataset")
    
    print("\nDone!")


if __name__ == '__main__':
    main()

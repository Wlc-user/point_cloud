"""
真实图片工业缺陷检测系统
使用用户提供的真实工业缺陷图片进行检测
"""
import sys
import os
import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class RealImageInspection:
    """真实图片工业缺陷检测系统"""
    
    def __init__(self, image_dir="real_images"):
        self.image_dir = image_dir
        self.results_dir = "inspection_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_images_from_folder(self, folder_path):
        """从文件夹加载图片"""
        images = []
        if not os.path.exists(folder_path):
            print(f"[WARNING] Folder not found: {folder_path}")
            return images
            
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        files = [f for f in os.listdir(folder_path) 
                if any(f.lower().endswith(ext) for ext in valid_extensions)]
        
        for file in sorted(files):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is not None:
                images.append({
                    'path': img_path,
                    'filename': file,
                    'image': img
                })
                print(f"  Loaded: {file}")
        
        return images
    
    def detect_defects(self, image, product_type='auto'):
        """
        检测缺陷
        根据产品类型自动选择合适的检测参数
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 根据产品类型选择参数
        if product_type == 'pcb':
            edges = cv2.Canny(gray, 50, 150)
        elif product_type == 'chip':
            edges = cv2.Canny(gray, 30, 100)
        elif product_type == 'wafer':
            edges = cv2.Canny(gray, 20, 80)
        else:  # auto
            edges = cv2.Canny(gray, 40, 120)
        
        # 形态学处理
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 分析缺陷
        defects = []
        result = image.copy()
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 20:  # 过滤太小的噪点
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # 缺陷分类
            if aspect_ratio > 4 or aspect_ratio < 0.25:
                defect_type = "SCRATCH"
                color = (0, 0, 255)
                severity = "HIGH"
            elif area > 1000:
                defect_type = "LARGE_DEFECT"
                color = (0, 0, 255)
                severity = "HIGH"
            elif area > 200:
                defect_type = "MEDIUM_DEFECT"
                color = (0, 165, 255)
                severity = "MEDIUM"
            else:
                defect_type = "SMALL_DEFECT"
                color = (0, 255, 255)
                severity = "LOW"
            
            defects.append({
                'id': i,
                'type': defect_type,
                'position': {'x': int(x + w/2), 'y': int(y + h/2)},
                'size': {'width': int(w), 'height': int(h)},
                'area_pixels': int(area),
                'severity': severity
            })
            
            # 在结果图上标记
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            label = f"{defect_type[:3]}#{i}"
            cv2.putText(result, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return result, defects
    
    def measure_basic(self, image):
        """基本尺寸测量"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = image.copy()
        measurements = []
        
        if contours:
            # 找到最大的轮廓
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # 绘制边界框
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 添加尺寸标注
            cv2.putText(result, f"W: {w}px", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(result, f"H: {h}px", (x + w + 5, y + h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            measurements.append({
                'bounding_box': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                'area_pixels': int(cv2.contourArea(main_contour))
            })
        
        return result, measurements
    
    def inspect_image(self, image_info, product_type='auto'):
        """检测单张图片"""
        print(f"\nInspecting: {image_info['filename']}")
        
        img = image_info['image']
        
        # 缺陷检测
        defect_result, defects = self.detect_defects(img, product_type)
        
        # 尺寸测量
        measure_result, measurements = self.measure_basic(img)
        
        # 保存结果
        base_name = Path(image_info['filename']).stem
        
        defect_path = os.path.join(self.results_dir, f"{base_name}_defects.jpg")
        measure_path = os.path.join(self.results_dir, f"{base_name}_measure.jpg")
        
        cv2.imwrite(defect_path, defect_result)
        cv2.imwrite(measure_path, measure_result)
        
        # 生成报告数据
        report = {
            'filename': image_info['filename'],
            'product_type': product_type,
            'image_size': {'width': img.shape[1], 'height': img.shape[0]},
            'defect_count': len(defects),
            'defects': defects,
            'measurements': measurements,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"  Defects found: {len(defects)}")
        for d in defects[:5]:  # 只显示前5个
            print(f"    - {d['type']} at ({d['position']['x']}, {d['position']['y']}) - {d['severity']}")
        
        return report, defect_result, measure_result
    
    def generate_summary_report(self, all_reports):
        """生成汇总报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary = {
            'inspection_time': timestamp,
            'total_images': len(all_reports),
            'total_defects': sum(r['defect_count'] for r in all_reports),
            'images': all_reports
        }
        
        # 保存JSON报告
        json_path = os.path.join(self.results_dir, f"summary_report_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n[OK] Summary report saved: {json_path}")
        return summary
    
    def run_inspection(self):
        """运行完整检测流程"""
        print("=" * 70)
        print("REAL IMAGE INDUSTRIAL DEFECT INSPECTION")
        print("=" * 70)
        
        # 检查是否有真实图片
        categories = {
            'pcb': os.path.join(self.image_dir, 'pcb'),
            'chip': os.path.join(self.image_dir, 'chip'),
            'wafer': os.path.join(self.image_dir, 'wafer')
        }
        
        all_reports = []
        has_real_images = False
        
        for product_type, folder in categories.items():
            print(f"\n{'='*60}")
            print(f"Loading {product_type.upper()} images...")
            print(f"{'='*60}")
            
            images = self.load_images_from_folder(folder)
            
            if images:
                has_real_images = True
                print(f"[OK] Loaded {len(images)} {product_type} images")
                
                for img_info in images:
                    report, _, _ = self.inspect_image(img_info, product_type)
                    all_reports.append(report)
            else:
                print(f"[INFO] No {product_type} images found in {folder}")
        
        if not has_real_images:
            print("\n" + "=" * 70)
            print("NO REAL IMAGES FOUND")
            print("=" * 70)
            print("\nPlease add your real defect images to:")
            for folder in categories.values():
                print(f"  - {folder}/")
            print("\nSupported formats: .jpg, .jpeg, .png, .bmp, .tiff")
            return
        
        # 生成汇总报告
        summary = self.generate_summary_report(all_reports)
        
        print("\n" + "=" * 70)
        print("INSPECTION COMPLETE")
        print("=" * 70)
        print(f"\nTotal images inspected: {summary['total_images']}")
        print(f"Total defects found: {summary['total_defects']}")
        print(f"\nResults saved to: {self.results_dir}/")
        print("\nGenerated files:")
        print("  - *_defects.jpg (defect detection results)")
        print("  - *_measure.jpg (measurement results)")
        print("  - summary_report_*.json (JSON report)")

def main():
    """主程序"""
    inspector = RealImageInspection()
    inspector.run_inspection()

if __name__ == '__main__':
    main()

import sys
import os
import cv2
import numpy as np
import json
from datetime import datetime

# 确保可以找到src模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class ProfessionalVisionInspection:
    """专业工业视觉检测系统"""
    
    def __init__(self):
        self.calibration_factor = None
        self.unit = 'mm'
        self.inspection_results = []
        
    def create_pcb_image(self):
        """创建PCB电路板图像（更专业的工业场景）"""
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        img[:] = (30, 60, 30)  # PCB绿色背景
        
        # 画基板轮廓
        cv2.rectangle(img, (50, 50), (750, 550), (50, 100, 50), -1)
        cv2.rectangle(img, (50, 50), (750, 550), (20, 40, 20), 3)
        
        # 画铜箔走线
        for i in range(10):
            y = 100 + i * 45
            cv2.line(img, (80, y), (720, y), (180, 120, 30), 3)
        
        # 画焊盘
        pad_positions = []
        for row in range(8):
            for col in range(12):
                x = 120 + col * 50
                y = 120 + row * 50
                pad_positions.append((x, y))
                cv2.circle(img, (x, y), 8, (200, 150, 50), -1)
                cv2.circle(img, (x, y), 3, (100, 70, 20), -1)
        
        # 画IC芯片
        cv2.rectangle(img, (300, 200), (500, 350), (30, 30, 30), -1)
        cv2.rectangle(img, (300, 200), (500, 350), (80, 80, 80), 2)
        
        # IC引脚
        for i in range(10):
            x = 310 + i * 20
            cv2.rectangle(img, (x, 190), (x + 10, 200), (150, 150, 150), -1)
            cv2.rectangle(img, (x, 350), (x + 10, 360), (150, 150, 150), -1)
        
        # 添加一些缺陷
        # 1. 焊盘偏移
        cv2.circle(img, (220, 270), 8, (200, 150, 50), -1)
        cv2.circle(img, (225, 275), 3, (100, 70, 20), -1)
        
        # 2. 短路（两条线连在一起）
        cv2.line(img, (80, 235), (720, 235), (180, 120, 30), 3)
        cv2.rectangle(img, (400, 230), (420, 245), (180, 120, 30), -1)
        
        # 3. 缺件（空焊盘）
        cv2.circle(img, (520, 320), 8, (50, 100, 50), -1)
        
        # 4. 划伤
        cv2.line(img, (150, 450), (250, 480), (10, 20, 10), 2)
        
        # 添加标识文字
        cv2.putText(img, "PCB-2024-001", (550, 520), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        return img
    
    def calibrate(self, image, reference_length_mm):
        """系统校准"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            width_pixels = w
            
            self.calibration_factor = reference_length_mm / width_pixels
            return True
        return False
    
    def detect_defects_advanced(self, image):
        """高级缺陷检测"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 多尺度边缘检测
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 找轮廓
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        result = image.copy()
        defects = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if 5 < area < 1000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h != 0 else 0
                solidity = area / (w * h) if w * h != 0 else 0
                
                # 缺陷分类
                defect_type = "UNKNOWN"
                color = (0, 255, 255)
                
                if aspect_ratio > 3 or aspect_ratio < 0.33:
                    defect_type = "SCRATCH"
                    color = (0, 0, 255)
                elif solidity < 0.3:
                    defect_type = "IRREGULAR"
                    color = (255, 0, 0)
                else:
                    defect_type = "SPOT"
                    color = (0, 255, 0)
                
                defects.append({
                    'id': i,
                    'type': defect_type,
                    'position': (x + w//2, y + h//2),
                    'area_pixels': area,
                    'area_mm2': area * (self.calibration_factor ** 2) if self.calibration_factor else None,
                    'bbox': (x, y, w, h),
                    'aspect_ratio': aspect_ratio,
                    'solidity': solidity,
                    'severity': 'HIGH' if area > 200 else 'MEDIUM' if area > 50 else 'LOW'
                })
                
                # 标记缺陷
                cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
                cv2.putText(result, f"{defect_type[:3]}#{i}", (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return result, defects
    
    def measure_dimensions_pro(self, image):
        """专业尺寸测量"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = image.copy()
        measurements = []
        
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            
            # 边界矩形
            x, y, w, h = cv2.boundingRect(main_contour)
            
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 计算尺寸
            width_pix = w
            height_pix = h
            
            width_mm = width_pix * self.calibration_factor if self.calibration_factor else None
            height_mm = height_pix * self.calibration_factor if self.calibration_factor else None
            
            center_x = x + w // 2
            center_y = y + h // 2
            
            measurements.append({
                'type': 'main_body',
                'center': (center_x, center_y),
                'width_pixels': width_pix,
                'height_pixels': height_pix,
                'width_mm': width_mm,
                'height_mm': height_mm
            })
            
            # 标注尺寸线
            # 宽度标注
            cv2.line(result, (x, center_y - 30), 
                    (x + w, center_y - 30), (0, 255, 0), 2)
            cv2.putText(result, f"W: {width_mm:.1f}mm" if width_mm else f"W: {width_pix}px", 
                       (center_x - 40, center_y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 高度标注
            cv2.line(result, (x + w + 30, y), 
                    (x + w + 30, y + h), (0, 255, 0), 2)
            cv2.putText(result, f"H: {height_mm:.1f}mm" if height_mm else f"H: {height_pix}px", 
                       (x + w + 35, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result, measurements
    
    def generate_report(self, image, defects, measurements, output_dir="reports"):
        """生成检测报告"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 统计数据
        defect_count = len(defects)
        high_severity = sum(1 for d in defects if d['severity'] == 'HIGH')
        medium_severity = sum(1 for d in defects if d['severity'] == 'MEDIUM')
        low_severity = sum(1 for d in defects if d['severity'] == 'LOW')
        
        # 生成JSON报告
        report = {
            'inspection_id': f"INS-{timestamp}",
            'timestamp': datetime.now().isoformat(),
            'status': 'FAIL' if high_severity > 0 else 'PASS' if defect_count == 0 else 'WARNING',
            'summary': {
                'total_defects': defect_count,
                'high_severity': high_severity,
                'medium_severity': medium_severity,
                'low_severity': low_severity
            },
            'defects': defects,
            'measurements': measurements,
            'calibration': {
                'factor': self.calibration_factor,
                'unit': self.unit
            }
        }
        
        json_path = os.path.join(output_dir, f"report_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成可视化报告
        report_img = np.ones((800, 1000, 3), dtype=np.uint8) * 240
        
        # 添加标题
        cv2.putText(report_img, "PROFESSIONAL VISION INSPECTION REPORT", (200, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(report_img, f"ID: {report['inspection_id']}", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
        cv2.putText(report_img, f"Time: {report['timestamp']}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
        
        # 状态指示
        status_color = (0, 255, 0) if report['status'] == 'PASS' else (0, 165, 255) if report['status'] == 'WARNING' else (0, 0, 255)
        cv2.putText(report_img, f"STATUS: {report['status']}", (700, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # 统计摘要
        y_start = 150
        cv2.putText(report_img, "DEFECT SUMMARY:", (50, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(report_img, f"  Total: {defect_count}", (50, y_start + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
        cv2.putText(report_img, f"  High: {high_severity}", (200, y_start + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(report_img, f"  Medium: {medium_severity}", (350, y_start + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(report_img, f"  Low: {low_severity}", (500, y_start + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 缩放并放入原图
        img_resized = cv2.resize(image, (400, 300))
        report_img[200:500, 50:450] = img_resized
        cv2.putText(report_img, "ORIGINAL", (180, 520), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 缺陷检测结果图
        defect_img = self.detect_defects_advanced(image)[0]
        defect_resized = cv2.resize(defect_img, (400, 300))
        report_img[200:500, 550:950] = defect_resized
        cv2.putText(report_img, "DEFECT DETECTION", (680, 520), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 缺陷列表
        cv2.putText(report_img, "DEFECT DETAILS:", (50, 560), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        for i, defect in enumerate(defects[:8]):  # 只显示前8个
            y = 590 + i * 25
            color = (0, 0, 255) if defect['severity'] == 'HIGH' else (0, 165, 255) if defect['severity'] == 'MEDIUM' else (0, 255, 0)
            cv2.putText(report_img, f"#{defect['id']}: {defect['type']} @ {defect['position']} ({defect['severity']})", 
                       (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        report_img_path = os.path.join(output_dir, f"report_{timestamp}.jpg")
        cv2.imwrite(report_img_path, report_img)
        
        return report, json_path, report_img_path

def main():
    print("=" * 70)
    print("PROFESSIONAL INDUSTRIAL VISION INSPECTION SYSTEM")
    print("=" * 70)
    
    inspection = ProfessionalVisionInspection()
    
    # 1. 创建专业PCB图像
    print("\n[1/6] Creating professional PCB image...")
    pcb_img = inspection.create_pcb_image()
    cv2.imwrite('professional_pcb.jpg', pcb_img)
    print("[OK] PCB image saved as 'professional_pcb.jpg'")
    
    # 2. 系统校准
    print("\n[2/6] Performing system calibration...")
    if inspection.calibrate(pcb_img, 100.0):  # 假设参考尺寸100mm
        print(f"[OK] Calibration complete: {inspection.calibration_factor:.6f} mm/pixel")
    else:
        print("[WARNING] Calibration failed, using pixel units")
    
    # 3. 高级缺陷检测
    print("\n[3/6] Performing advanced defect detection...")
    defect_result, defects = inspection.detect_defects_advanced(pcb_img)
    cv2.imwrite('professional_defects.jpg', defect_result)
    print(f"[OK] Defect detection complete, found {len(defects)} defects")
    
    # 统计缺陷
    high = sum(1 for d in defects if d['severity'] == 'HIGH')
    medium = sum(1 for d in defects if d['severity'] == 'MEDIUM')
    low = sum(1 for d in defects if d['severity'] == 'LOW')
    print(f"    - High severity: {high}")
    print(f"    - Medium severity: {medium}")
    print(f"    - Low severity: {low}")
    
    # 4. 专业尺寸测量
    print("\n[4/6] Performing professional dimension measurement...")
    measure_result, measurements = inspection.measure_dimensions_pro(pcb_img)
    cv2.imwrite('professional_measurement.jpg', measure_result)
    print("[OK] Dimension measurement complete")
    for meas in measurements:
        print(f"    - Width: {meas.get('width_mm', 'N/A')} mm")
        print(f"    - Height: {meas.get('height_mm', 'N/A')} mm")
    
    # 5. 生成检测报告
    print("\n[5/6] Generating inspection report...")
    report, json_path, report_img_path = inspection.generate_report(
        pcb_img, defects, measurements
    )
    print(f"[OK] Report generated:")
    print(f"    - JSON: {json_path}")
    print(f"    - Visual: {report_img_path}")
    print(f"    - Status: {report['status']}")
    
    # 6. 创建综合视图
    print("\n[6/6] Creating comprehensive visualization...")
    report_img = cv2.imread(report_img_path)
    
    # 调整所有图像到相同高度
    target_height = 600
    pcb_resized = cv2.resize(pcb_img, (800, target_height))
    defect_resized = cv2.resize(defect_result, (800, target_height))
    measure_resized = cv2.resize(measure_result, (800, target_height))
    report_resized = cv2.resize(report_img, (800, target_height))
    
    h1 = np.hstack((pcb_resized, defect_resized))
    h2 = np.hstack((measure_resized, report_resized))
    combined = np.vstack((h1, h2))
    
    cv2.imwrite('professional_combined.jpg', combined)
    print("[OK] Combined visualization saved as 'professional_combined.jpg'")
    
    # 显示结果
    print("\nDisplaying results...")
    print("Press any key to close windows")
    
    try:
        cv2.imshow('Original PCB', pcb_img)
        cv2.imshow('Defect Detection', defect_result)
        cv2.imshow('Dimension Measurement', measure_result)
        cv2.imshow('Inspection Report', cv2.imread(report_img_path))
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("[OK] Windows closed")
    except Exception as e:
        print(f"[INFO] Could not display GUI windows: {e}")
        print("[INFO] But all result images and reports have been saved!")
    
    print("\n" + "=" * 70)
    print("PROFESSIONAL INSPECTION COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - professional_pcb.jpg (original PCB)")
    print("  - professional_defects.jpg (defect detection)")
    print("  - professional_measurement.jpg (dimension measurement)")
    print("  - professional_combined.jpg (comprehensive view)")
    print("  - reports/ (report directory)")

if __name__ == '__main__':
    main()

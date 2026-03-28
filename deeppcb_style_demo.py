"""
DeepPCB风格的PCB缺陷检测演示
参考DeepPCB数据集的真实缺陷样式
"""
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class DeepPCBStyleDemo:
    """DeepPCB风格的PCB缺陷检测"""
    
    def __init__(self):
        self.width = 640
        self.height = 480
        
    def create_pcb_template(self):
        """创建无缺陷的PCB模板图像（类似DeepPCB风格）"""
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        
        # 绘制PCB走线（黑色线条在白色背景上）
        # 水平走线
        for i in range(8):
            y = 60 + i * 50
            cv2.line(img, (50, y), (590, y), 0, 3)
        
        # 垂直走线和连接
        # 左侧连接器
        for i in range(12):
            y = 100 + i * 25
            cv2.line(img, (50, y), (80, y), 0, 2)
            cv2.circle(img, (50, y), 4, 0, -1)
        
        # 右侧连接器
        for i in range(12):
            y = 100 + i * 25
            cv2.line(img, (560, y), (590, y), 0, 2)
            cv2.circle(img, (590, y), 4, 0, -1)
        
        # 中间复杂走线
        # 斜线连接
        cv2.line(img, (150, 60), (200, 160), 0, 3)
        cv2.line(img, (250, 60), (300, 210), 0, 3)
        cv2.line(img, (350, 110), (400, 260), 0, 3)
        cv2.line(img, (450, 160), (500, 310), 0, 3)
        
        # 垂直连接线
        cv2.line(img, (200, 160), (200, 260), 0, 3)
        cv2.line(img, (300, 210), (300, 360), 0, 3)
        cv2.line(img, (400, 260), (400, 410), 0, 3)
        
        # 焊盘（圆形）
        pads = [
            (150, 60), (250, 60), (350, 60), (450, 60),
            (150, 110), (250, 110), (350, 110), (450, 110),
            (200, 160), (300, 160), (400, 160), (500, 160),
            (150, 210), (250, 210), (350, 210), (450, 210),
            (200, 260), (300, 260), (400, 260), (500, 260),
            (150, 310), (250, 310), (350, 310), (450, 310),
            (200, 360), (300, 360), (400, 360), (500, 360),
            (150, 410), (250, 410), (350, 410), (450, 410),
        ]
        
        for pad in pads:
            cv2.circle(img, pad, 8, 0, 2)
            cv2.circle(img, pad, 3, 0, -1)
        
        # 过孔（小圆点）
        vias = [
            (180, 135), (280, 185), (380, 235), (480, 285),
            (180, 285), (280, 335), (380, 385), (480, 135),
        ]
        for via in vias:
            cv2.circle(img, via, 4, 0, -1)
        
        return img
    
    def add_defects_deeppcb_style(self, template):
        """添加DeepPCB风格的缺陷"""
        defective = template.copy()
        defects_info = []
        
        # 1. Open (开路) - 走线断裂
        # 在水平走线上制造断裂
        cv2.line(defective, (180, 160), (220, 160), 255, 4)  # 擦除一段
        defects_info.append({
            'type': 'open',
            'bbox': (180, 158, 40, 4),
            'description': 'Circuit line break'
        })
        
        # 2. Short (短路) - 走线连接
        # 连接两条平行走线
        cv2.line(defective, (350, 110), (350, 160), 0, 4)
        cv2.line(defective, (345, 135), (355, 135), 0, 4)
        defects_info.append({
            'type': 'short',
            'bbox': (348, 110, 4, 50),
            'description': 'Unwanted connection between traces'
        })
        
        # 3. Mousebite (鼠咬) - 边缘缺口
        # 在走线边缘制造缺口
        cv2.ellipse(defective, (280, 260), (8, 4), 0, 0, 360, 255, -1)
        defects_info.append({
            'type': 'mousebite',
            'bbox': (272, 256, 16, 8),
            'description': 'Edge notch on trace'
        })
        
        # 4. Spur (毛刺) - 多余铜箔突出
        # 从走线突出的小铜箔
        cv2.line(defective, (400, 260), (420, 240), 0, 3)
        cv2.line(defective, (420, 240), (425, 245), 0, 3)
        defects_info.append({
            'type': 'spur',
            'bbox': (400, 240, 25, 20),
            'description': 'Copper protrusion from trace'
        })
        
        # 5. Copper (多余铜) - 残留铜箔
        # 孤立的铜箔区域
        cv2.ellipse(defective, (480, 310), (12, 8), 30, 0, 360, 0, -1)
        defects_info.append({
            'type': 'copper',
            'bbox': (468, 302, 24, 16),
            'description': 'Excess copper residue'
        })
        
        # 6. Pin-hole (针孔) - 孔洞
        # 走线上的小孔
        cv2.circle(defective, (150, 310), 5, 255, -1)
        defects_info.append({
            'type': 'pin-hole',
            'bbox': (145, 305, 10, 10),
            'description': 'Hole in trace'
        })
        
        return defective, defects_info
    
    def visualize_defects(self, defective_img, defects_info):
        """可视化缺陷（类似DeepPCB的标注方式）"""
        # 转换为彩色图像
        vis_img = cv2.cvtColor(defective_img, cv2.COLOR_GRAY2BGR)
        
        # 绘制缺陷标注框和标签
        for i, defect in enumerate(defects_info):
            x, y, w, h = defect['bbox']
            defect_type = defect['type']
            
            # 不同缺陷类型用不同颜色
            color_map = {
                'open': (0, 0, 255),      # 红色
                'short': (0, 165, 255),   # 橙色
                'mousebite': (0, 255, 255), # 黄色
                'spur': (255, 0, 0),      # 蓝色
                'copper': (255, 0, 255),  # 紫色
                'pin-hole': (0, 255, 0)   # 绿色
            }
            color = color_map.get(defect_type, (128, 128, 128))
            
            # 绘制边界框
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
            
            # 绘制标签背景
            label = f"{defect_type}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis_img, (x, y - text_h - 8), (x + text_w, y), color, -1)
            
            # 绘制标签文字
            cv2.putText(vis_img, label, (x, y - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis_img
    
    def detect_defects(self, template, defective):
        """缺陷检测算法（模板对比法）"""
        # 计算差异
        diff = cv2.absdiff(template, defective)
        
        # 二值化差异图像
        _, binary = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
        
        # 形态学处理
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 分析缺陷
        detected_defects = []
        result_img = cv2.cvtColor(defective, cv2.COLOR_GRAY2BGR)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 20:  # 过滤小噪点
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # 根据形状和位置分类
            aspect_ratio = float(w) / h if h > 0 else 0
            
            if aspect_ratio > 3:
                defect_type = "open"
                color = (0, 0, 255)
            elif aspect_ratio < 0.5:
                defect_type = "short"
                color = (0, 165, 255)
            elif area < 100:
                defect_type = "pin-hole"
                color = (0, 255, 0)
            else:
                defect_type = "defect"
                color = (255, 0, 0)
            
            detected_defects.append({
                'id': i,
                'type': defect_type,
                'bbox': (x, y, w, h),
                'area': area
            })
            
            # 绘制检测结果
            cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(result_img, defect_type, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return result_img, detected_defects, diff
    
    def run_demo(self):
        """运行DeepPCB风格演示"""
        print("=" * 70)
        print("DEEPPCB STYLE PCB DEFECT DETECTION DEMO")
        print("=" * 70)
        
        # 1. 创建模板图像
        print("\n[1/4] Creating PCB template image...")
        template = self.create_pcb_template()
        cv2.imwrite('deeppcb_template.jpg', template)
        print("[OK] Template saved: deeppcb_template.jpg")
        
        # 2. 添加缺陷
        print("\n[2/4] Adding defects (DeepPCB style)...")
        defective, defects_info = self.add_defects_deeppcb_style(template)
        cv2.imwrite('deeppcb_defective.jpg', defective)
        print(f"[OK] Defective image saved: deeppcb_defective.jpg")
        print(f"    Added {len(defects_info)} defects:")
        for d in defects_info:
            print(f"      - {d['type']}: {d['description']}")
        
        # 3. 可视化标注
        print("\n[3/4] Creating visualization with annotations...")
        annotated = self.visualize_defects(defective, defects_info)
        cv2.imwrite('deeppcb_annotated.jpg', annotated)
        print("[OK] Annotated image saved: deeppcb_annotated.jpg")
        
        # 4. 自动检测
        print("\n[4/4] Running automatic defect detection...")
        detected_img, detected_defects, diff = self.detect_defects(template, defective)
        cv2.imwrite('deeppcb_detected.jpg', detected_img)
        cv2.imwrite('deeppcb_difference.jpg', diff)
        print(f"[OK] Detection complete, found {len(detected_defects)} defects")
        for d in detected_defects:
            print(f"    - Detected: {d['type']} at ({d['bbox'][0]}, {d['bbox'][1]})")
        
        # 创建综合对比图
        print("\n[5/5] Creating comparison view...")
        
        # 调整大小并转换为彩色
        template_color = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
        defective_color = cv2.cvtColor(defective, cv2.COLOR_GRAY2BGR)
        diff_color = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        
        # 添加标题
        cv2.putText(template_color, "TEMPLATE (Defect-free)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(defective_color, "DEFECTIVE (With defects)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(annotated, "GROUND TRUTH (Annotations)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(detected_img, "DETECTION RESULT", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 拼接
        top = np.hstack((template_color, defective_color))
        bottom = np.hstack((annotated, detected_img))
        combined = np.vstack((top, bottom))
        
        cv2.imwrite('deeppcb_comparison.jpg', combined)
        print("[OK] Comparison view saved: deeppcb_comparison.jpg")
        
        # 显示结果
        print("\nDisplaying results...")
        try:
            cv2.imshow('DeepPCB Style - Template', template_color)
            cv2.imshow('DeepPCB Style - Defective', defective_color)
            cv2.imshow('DeepPCB Style - Annotated', annotated)
            cv2.imshow('DeepPCB Style - Detected', detected_img)
            cv2.imshow('DeepPCB Style - Comparison', combined)
            
            print("Press any key to close windows")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"[INFO] Display error: {e}")
        
        print("\n" + "=" * 70)
        print("DEEPPCB STYLE DEMO COMPLETE!")
        print("=" * 70)
        print("\nGenerated files:")
        print("  - deeppcb_template.jpg (defect-free template)")
        print("  - deeppcb_defective.jpg (image with defects)")
        print("  - deeppcb_annotated.jpg (ground truth annotations)")
        print("  - deeppcb_detected.jpg (automatic detection result)")
        print("  - deeppcb_difference.jpg (template difference)")
        print("  - deeppcb_comparison.jpg (2x2 comparison view)")

def main():
    demo = DeepPCBStyleDemo()
    demo.run_demo()

if __name__ == '__main__':
    main()

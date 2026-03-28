import sys
import os
import cv2
import numpy as np
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class RealDefectsDemo:
    """真实工业缺陷演示系统"""
    
    def __init__(self):
        self.defect_types = {
            'pcb': ['open_circuit', 'short_circuit', 'solder_bridge', 'missing_component', 
                   'wrong_component', 'tombstone', 'cold_solder', 'solder_void'],
            'chip': ['die_crack', 'pad_corrosion', 'wire_bond_lift', 'contamination',
                    'scratch', 'discoloration', 'foreign_particle'],
            'wafer': ['edge_chip', 'scratch', 'particle', 'residue', 'probe_mark',
                     'pattern_defect', 'ring_oscillator']
        }
    
    def create_pcb_with_real_defects(self):
        """创建带有真实PCB缺陷的图像"""
        # PCB基板 (800x600)
        img = np.ones((600, 800, 3), dtype=np.uint8) * 40
        img[:, :] = (40, 80, 40)  # PCB绿色
        
        # 画基板边缘
        cv2.rectangle(img, (50, 50), (750, 550), (60, 120, 60), -1)
        cv2.rectangle(img, (50, 50), (750, 550), (20, 40, 20), 3)
        
        # 正常的铜箔走线
        for i in range(8):
            y = 100 + i * 55
            cv2.line(img, (100, y), (700, y), (180, 140, 60), 4)
        
        # 正常的焊盘
        for row in range(6):
            for col in range(10):
                x = 150 + col * 55
                y = 120 + row * 65
                cv2.circle(img, (x, y), 12, (200, 160, 80), -1)
                cv2.circle(img, (x, y), 5, (140, 100, 40), -1)
        
        # === 真实PCB缺陷 ===
        
        # 1. 开路 (Open Circuit) - 走线断裂
        # 模拟铜箔断裂
        cv2.line(img, (200, 155), (250, 155), (40, 80, 40), 6)  # 擦除一段走线
        cv2.line(img, (195, 150), (205, 160), (100, 80, 40), 2)  # 断裂边缘1
        cv2.line(img, (245, 150), (255, 160), (100, 80, 40), 2)  # 断裂边缘2
        
        # 2. 短路 (Short Circuit) - 两条走线意外连接
        cv2.rectangle(img, (380, 205), (420, 215), (180, 140, 60), -1)  # 桥接两条线
        
        # 3. 焊桥 (Solder Bridge) - 焊锡过多导致相邻焊盘连接
        cv2.ellipse(img, (315, 250), (25, 15), 0, 0, 360, (220, 200, 150), -1)
        
        # 4. 缺件 (Missing Component) - 焊盘上没有元件
        # 空焊盘（比其他焊盘颜色更深）
        cv2.circle(img, (535, 315), 12, (50, 100, 50), -1)
        cv2.circle(img, (535, 315), 5, (30, 60, 30), -1)
        
        # 5. 立碑 (Tombstone) - 元件一端翘起
        # 模拟立起的电阻
        cv2.rectangle(img, (400, 370), (410, 400), (180, 180, 200), -1)  # 竖立的元件
        cv2.circle(img, (405, 405), 8, (200, 160, 80), -1)  # 正常焊盘
        cv2.circle(img, (405, 365), 8, (50, 100, 50), -1)   # 空焊盘（元件翘起）
        
        # 6. 冷焊 (Cold Solder) - 焊点表面粗糙、暗淡
        cv2.circle(img, (205, 445), 12, (120, 100, 60), -1)  # 暗淡的焊点
        cv2.circle(img, (205, 445), 5, (80, 60, 30), -1)
        
        # 7. 焊锡空洞 (Solder Void) - 焊点内部气泡
        cv2.circle(img, (480, 445), 12, (200, 160, 80), -1)
        cv2.circle(img, (480, 445), 4, (60, 120, 60), -1)  # 空洞
        
        # 8. 划伤 (Scratch) - 表面物理损伤
        cv2.line(img, (100, 500), (200, 520), (20, 30, 20), 2)
        cv2.line(img, (105, 498), (205, 518), (20, 30, 20), 2)
        
        # 添加标签
        labels = [
            ("OPEN CIRCUIT", (180, 140)),
            ("SHORT CIRCUIT", (360, 190)),
            ("SOLDER BRIDGE", (290, 230)),
            ("MISSING COMPONENT", (490, 290)),
            ("TOMBSTONE", (370, 350)),
            ("COLD SOLDER", (150, 430)),
            ("SOLDER VOID", (430, 430)),
            ("SCRATCH", (80, 490))
        ]
        
        for label, pos in labels:
            cv2.putText(img, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
        
        cv2.putText(img, "PCB-2024-REAL-DEFECTS", (500, 580), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        return img
    
    def create_chip_with_real_defects(self):
        """创建带有真实芯片缺陷的图像"""
        img = np.ones((600, 800, 3), dtype=np.uint8) * 30
        
        # 芯片封装主体
        chip_x, chip_y = 400, 300
        chip_w, chip_h = 300, 200
        
        # 封装体
        cv2.rectangle(img, (chip_x - chip_w//2, chip_y - chip_h//2), 
                     (chip_x + chip_w//2, chip_y + chip_h//2), (50, 50, 60), -1)
        cv2.rectangle(img, (chip_x - chip_w//2, chip_y - chip_h//2), 
                     (chip_x + chip_w//2, chip_y + chip_h//2), (80, 80, 90), 3)
        
        # 引脚
        for i in range(10):
            x = chip_x - chip_w//2 - 30 + i * 35
            cv2.rectangle(img, (x, chip_y - chip_h//2 - 20), (x + 20, chip_y - chip_h//2), 
                         (180, 180, 190), -1)
            cv2.rectangle(img, (x, chip_y + chip_h//2), (x + 20, chip_y + chip_h//2 + 20), 
                         (180, 180, 190), -1)
        
        # Die (芯片核心)
        cv2.rectangle(img, (chip_x - 80, chip_y - 60), (chip_x + 80, chip_y + 60), 
                     (30, 30, 35), -1)
        
        # === 真实芯片缺陷 ===
        
        # 1. Die Crack (芯片裂纹) - 物理应力导致的裂纹
        cv2.line(img, (chip_x - 40, chip_y - 20), (chip_x + 20, chip_y + 30), (100, 50, 50), 2)
        cv2.line(img, (chip_x - 35, chip_y - 25), (chip_x + 25, chip_y + 25), (150, 80, 80), 1)
        
        # 2. Pad Corrosion (焊盘腐蚀) - 化学腐蚀导致的变色
        cv2.circle(img, (chip_x - 120, chip_y - 80), 15, (100, 80, 60), -1)
        cv2.circle(img, (chip_x - 120, chip_y - 80), 15, (150, 120, 90), 2)
        
        # 3. Wire Bond Lift (金丝键合脱落) - 连接断开
        # 正常的键合线
        cv2.line(img, (chip_x - 60, chip_y - 40), (chip_x - 120, chip_y - 80), 
                (200, 200, 180), 2)
        # 脱落的键合线（弯曲下垂）
        pts = np.array([[chip_x + 40, chip_y - 30], 
                       [chip_x + 80, chip_y - 50],
                       [chip_x + 120, chip_y - 30]], np.int32)
        cv2.polylines(img, [pts], False, (180, 180, 160), 2)
        
        # 4. Contamination (污染) - 异物附着
        cv2.circle(img, (chip_x + 50, chip_y + 40), 8, (80, 70, 60), -1)
        cv2.circle(img, (chip_x + 55, chip_y + 35), 3, (60, 50, 40), -1)
        
        # 5. Scratch (划痕) - 表面划伤
        cv2.line(img, (chip_x - 100, chip_y + 50), (chip_x - 40, chip_y + 80), 
                (100, 100, 110), 2)
        
        # 6. Discoloration (变色) - 过热或老化导致的颜色变化
        cv2.ellipse(img, (chip_x, chip_y), (100, 70), 0, 0, 360, (60, 50, 50), -1)
        
        # 7. Foreign Particle (异物颗粒) - 灰尘或碎屑
        cv2.circle(img, (chip_x + 90, chip_y - 50), 4, (120, 120, 120), -1)
        cv2.circle(img, (chip_x - 90, chip_y + 60), 3, (100, 100, 100), -1)
        
        # 添加标签
        labels = [
            ("DIE CRACK", (chip_x - 60, chip_y + 50)),
            ("PAD CORROSION", (chip_x - 180, chip_y - 100)),
            ("WIRE BOND LIFT", (chip_x + 60, chip_y - 70)),
            ("CONTAMINATION", (chip_x + 70, chip_y + 60)),
            ("SCRATCH", (chip_x - 130, chip_y + 100)),
            ("DISCOLORATION", (chip_x - 40, chip_y + 10)),
            ("FOREIGN PARTICLE", (chip_x + 100, chip_y - 40))
        ]
        
        for label, pos in labels:
            cv2.putText(img, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
        
        cv2.putText(img, "IC-CHIP-REAL-DEFECTS", (500, 580), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        return img
    
    def create_wafer_with_real_defects(self):
        """创建带有真实晶圆缺陷的图像"""
        img = np.ones((700, 700, 3), dtype=np.uint8) * 20
        
        center_x, center_y = 350, 350
        wafer_radius = 280
        
        # 晶圆主体（圆形）
        for r in range(wafer_radius, 0, -1):
            color = int(60 + (r / wafer_radius) * 40)
            cv2.circle(img, (center_x, center_y), r, (color, color, color + 10), -1)
        
        # 晶圆边缘
        cv2.circle(img, (center_x, center_y), wafer_radius, (100, 100, 110), 3)
        
        # 画晶格（Dies）
        die_size = 35
        for row in range(-6, 7):
            for col in range(-6, 7):
                x = center_x + col * die_size
                y = center_y + row * die_size
                
                # 检查是否在圆内
                if np.sqrt((x - center_x)**2 + (y - center_y)**2) < wafer_radius - 20:
                    cv2.rectangle(img, (x - die_size//2, y - die_size//2), 
                                 (x + die_size//2, y + die_size//2), 
                                 (80, 80, 90), 1)
        
        # 平边（Flat）- 晶圆定位边
        cv2.rectangle(img, (center_x - 40, center_y - wafer_radius - 5), 
                     (center_x + 40, center_y - wafer_radius + 15), (20, 20, 20), -1)
        
        # === 真实晶圆缺陷 ===
        
        # 1. Edge Chip (边缘崩边) - 晶圆边缘破损
        chip_points = np.array([
            [center_x + wafer_radius - 30, center_y + 100],
            [center_x + wafer_radius + 10, center_y + 80],
            [center_x + wafer_radius + 5, center_y + 120]
        ], np.int32)
        cv2.fillPoly(img, [chip_points], (20, 20, 20))
        
        # 2. Scratch (划痕) - 处理或运输过程中的划伤
        cv2.line(img, (center_x - 150, center_y - 100), 
                (center_x + 50, center_y + 100), (40, 40, 50), 2)
        cv2.line(img, (center_x - 145, center_y - 105), 
                (center_x + 55, center_y + 95), (60, 60, 70), 1)
        
        # 3. Particle (颗粒污染) - 灰尘或工艺残留物
        cv2.circle(img, (center_x - 80, center_y - 50), 6, (100, 100, 100), -1)
        cv2.circle(img, (center_x + 120, center_y + 80), 4, (90, 90, 90), -1)
        cv2.circle(img, (center_x - 50, center_y + 150), 5, (110, 110, 110), -1)
        
        # 4. Residue (残留物) - 光刻胶或化学品残留
        cv2.ellipse(img, (center_x + 60, center_y - 120), (20, 12), 30, 0, 360, 
                   (70, 70, 80), -1)
        
        # 5. Probe Mark (探针痕迹) - 测试探针留下的压痕
        for i in range(3):
            for j in range(3):
                px = center_x - 100 + i * 15
                py = center_y + 50 + j * 15
                cv2.circle(img, (px, py), 3, (50, 50, 60), -1)
        
        # 6. Pattern Defect (图案缺陷) - 光刻图案缺陷
        # 某个die上的图案不完整
        die_x, die_y = center_x + 70, center_y - 70
        cv2.rectangle(img, (die_x - 15, die_y - 15), (die_x + 15, die_y + 15), 
                     (60, 60, 70), -1)
        cv2.line(img, (die_x - 10, die_y - 10), (die_x + 10, die_y + 10), 
                (100, 100, 110), 2)
        
        # 7. Ring Oscillator (环形振荡器异常) - 测试结构异常
        # 画一个异常的测试结构
        test_x, test_y = center_x - 150, center_y + 120
        cv2.circle(img, (test_x, test_y), 25, (70, 70, 80), 2)
        cv2.circle(img, (test_x, test_y), 20, (70, 70, 80), 2)
        cv2.circle(img, (test_x, test_y), 15, (70, 70, 80), 2)
        cv2.line(img, (test_x - 30, test_y), (test_x + 30, test_y), (150, 50, 50), 2)
        
        # 添加标签
        labels = [
            ("EDGE CHIP", (center_x + wafer_radius - 80, center_y + 130)),
            ("SCRATCH", (center_x - 100, center_y + 20)),
            ("PARTICLE", (center_x - 110, center_y - 60)),
            ("RESIDUE", (center_x + 30, center_y - 140)),
            ("PROBE MARK", (center_x - 140, center_y + 100)),
            ("PATTERN DEFECT", (center_x + 50, center_y - 50)),
            ("RING OSCILLATOR", (center_x - 200, center_y + 160))
        ]
        
        for label, pos in labels:
            cv2.putText(img, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
        
        cv2.putText(img, "WAFER-300MM-REAL-DEFECTS", (400, 680), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        return img
    
    def detect_and_analyze(self, image, product_type):
        """检测并分析缺陷"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用不同的检测策略
        if product_type == 'pcb':
            edges = cv2.Canny(gray, 50, 150)
        elif product_type == 'chip':
            edges = cv2.Canny(gray, 30, 100)
        else:  # wafer
            edges = cv2.Canny(gray, 20, 80)
        
        # 形态学处理
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = image.copy()
        detected_defects = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 10:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # 根据形状特征分类
                if aspect_ratio > 5 or aspect_ratio < 0.2:
                    defect_type = "SCRATCH"
                    color = (0, 0, 255)
                elif area > 500:
                    defect_type = "LARGE_DEFECT"
                    color = (0, 0, 255)
                elif area > 100:
                    defect_type = "MEDIUM_DEFECT"
                    color = (0, 165, 255)
                else:
                    defect_type = "SMALL_DEFECT"
                    color = (0, 255, 255)
                
                detected_defects.append({
                    'id': i,
                    'type': defect_type,
                    'area': area,
                    'position': (x + w//2, y + h//2),
                    'bbox': (x, y, w, h)
                })
                
                cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
                cv2.putText(result, f"{defect_type[:3]}", (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return result, detected_defects
    
    def run_demo(self):
        """运行完整演示"""
        print("=" * 70)
        print("REAL INDUSTRIAL DEFECTS DEMONSTRATION")
        print("=" * 70)
        
        # 1. PCB缺陷
        print("\n[1/3] Creating PCB with real defects...")
        pcb_img = self.create_pcb_with_real_defects()
        pcb_detected, pcb_defects = self.detect_and_analyze(pcb_img, 'pcb')
        cv2.imwrite('real_pcb_defects.jpg', pcb_img)
        cv2.imwrite('real_pcb_detected.jpg', pcb_detected)
        print(f"[OK] PCB: {len(pcb_defects)} defects detected")
        
        # 2. 芯片缺陷
        print("\n[2/3] Creating IC Chip with real defects...")
        chip_img = self.create_chip_with_real_defects()
        chip_detected, chip_defects = self.detect_and_analyze(chip_img, 'chip')
        cv2.imwrite('real_chip_defects.jpg', chip_img)
        cv2.imwrite('real_chip_detected.jpg', chip_detected)
        print(f"[OK] IC Chip: {len(chip_defects)} defects detected")
        
        # 3. 晶圆缺陷
        print("\n[3/3] Creating Wafer with real defects...")
        wafer_img = self.create_wafer_with_real_defects()
        wafer_detected, wafer_defects = self.detect_and_analyze(wafer_img, 'wafer')
        cv2.imwrite('real_wafer_defects.jpg', wafer_img)
        cv2.imwrite('real_wafer_detected.jpg', wafer_detected)
        print(f"[OK] Wafer: {len(wafer_defects)} defects detected")
        
        # 创建综合视图
        print("\n[4/4] Creating comprehensive view...")
        
        # 调整大小并拼接
        h, w = 600, 800
        pcb_resized = cv2.resize(pcb_img, (w, h))
        chip_resized = cv2.resize(chip_img, (w, h))
        wafer_resized = cv2.resize(wafer_img, (h, h))  # 晶圆是正方形
        wafer_resized = cv2.resize(wafer_resized, (w, h))
        
        row1 = np.hstack((pcb_resized, chip_resized))
        row2 = np.hstack((wafer_resized, wafer_resized))  # 复制晶圆填充
        combined = np.vstack((row1, row2))
        
        cv2.imwrite('real_defects_combined.jpg', combined)
        print("[OK] Combined view saved")
        
        # 显示结果
        print("\nDisplaying results...")
        try:
            cv2.imshow('PCB Defects', pcb_img)
            cv2.imshow('IC Chip Defects', chip_img)
            cv2.imshow('Wafer Defects', wafer_img)
            cv2.imshow('Combined View', combined)
            
            print("Press any key to close windows")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"[INFO] Display error: {e}")
        
        print("\n" + "=" * 70)
        print("REAL DEFECTS DEMO COMPLETE!")
        print("=" * 70)
        print("\nGenerated files:")
        print("  - real_pcb_defects.jpg")
        print("  - real_pcb_detected.jpg")
        print("  - real_chip_defects.jpg")
        print("  - real_chip_detected.jpg")
        print("  - real_wafer_defects.jpg")
        print("  - real_wafer_detected.jpg")
        print("  - real_defects_combined.jpg")

if __name__ == '__main__':
    demo = RealDefectsDemo()
    demo.run_demo()

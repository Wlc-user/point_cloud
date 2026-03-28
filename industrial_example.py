import sys
import os
import cv2
import numpy as np

# 确保可以找到src模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_industrial_workpiece_image():
    """创建一个工业工件图像"""
    # 创建黑色背景
    img = np.zeros((500, 700, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)  # 深色背景
    
    # 画一个金属工件（圆形基板）
    center = (350, 250)
    radius = 180
    # 金属渐变效果
    for i in range(radius, 0, -1):
        color = int(150 + (i / radius) * 105)
        cv2.circle(img, center, i, (color, color, color), -1)
    
    # 画中心孔
    cv2.circle(img, center, 30, (30, 30, 30), -1)
    
    # 画4个安装孔
    hole_positions = [
        (350, 250 - 100),  # 上
        (350 + 100, 250),  # 右
        (350, 250 + 100),  # 下
        (350 - 100, 250)   # 左
    ]
    for pos in hole_positions:
        cv2.circle(img, pos, 15, (30, 30, 30), -1)
    
    # 添加一些表面缺陷（划痕和凹坑）
    # 划痕1
    cv2.line(img, (200, 150), (280, 200), (80, 80, 80), 2)
    # 划痕2
    cv2.line(img, (420, 300), (500, 350), (80, 80, 80), 2)
    # 凹坑
    cv2.circle(img, (300, 180), 8, (100, 100, 100), -1)
    cv2.circle(img, (400, 320), 6, (100, 100, 100), -1)
    
    # 添加文字标签
    cv2.putText(img, "INDUSTRIAL PART", (220, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    
    return img

def detect_defects(image):
    """工业缺陷检测"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)
    
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = image.copy()
    defects = []
    
    # 检测可能的缺陷区域
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if 10 < area < 500:  # 过滤过小和过大的轮廓
            x, y, w, h = cv2.boundingRect(contour)
            defects.append({
                'id': i,
                'position': (x + w//2, y + h//2),
                'area': area,
                'bbox': (x, y, w, h)
            })
            # 用红色矩形标记缺陷
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(result, f"DEF#{i}", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    return result, defects

def measure_dimensions(image):
    """工业尺寸测量"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # 找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = image.copy()
    measurements = []
    
    if contours:
        # 找到最大的轮廓（工件主体）
        main_contour = max(contours, key=cv2.contourArea)
        
        # 计算最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(main_contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # 画外接圆和直径线
        cv2.circle(result, center, radius, (0, 255, 0), 2)
        cv2.line(result, (center[0] - radius, center[1]), 
                (center[0] + radius, center[1]), (0, 255, 0), 2)
        
        # 画中心十字
        cv2.line(result, (center[0] - 20, center[1]), 
                (center[0] + 20, center[1]), (255, 255, 0), 2)
        cv2.line(result, (center[0], center[1] - 20), 
                (center[0], center[1] + 20), (255, 255, 0), 2)
        
        measurements.append({
            'type': 'main_circle',
            'center': center,
            'diameter_pixels': radius * 2,
            'radius_pixels': radius
        })
        
        # 标注尺寸
        cv2.putText(result, f"DIA: {radius * 2}px", 
                   (center[0] - 60, center[1] - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return result, measurements

def main():
    print("=" * 60)
    print("INDUSTRIAL MACHINE VISION EXAMPLE")
    print("=" * 60)
    
    # 1. 创建工业工件图像
    print("\n[1/4] Creating industrial workpiece image...")
    workpiece_img = create_industrial_workpiece_image()
    cv2.imwrite('industrial_workpiece.jpg', workpiece_img)
    print("[OK] Industrial workpiece image saved as 'industrial_workpiece.jpg'")
    
    # 2. 缺陷检测
    print("\n[2/4] Performing defect detection...")
    defect_result, defects = detect_defects(workpiece_img)
    cv2.imwrite('defect_detection_result.jpg', defect_result)
    print(f"[OK] Defect detection complete, found {len(defects)} potential defects")
    for defect in defects:
        print(f"    - Defect #{defect['id']}: Position {defect['position']}, Area {defect['area']:.1f}px2")
    
    # 3. 尺寸测量
    print("\n[3/4] Performing dimension measurement...")
    measure_result, measurements = measure_dimensions(workpiece_img)
    cv2.imwrite('measurement_result.jpg', measure_result)
    print("[OK] Dimension measurement complete")
    for meas in measurements:
        if meas['type'] == 'main_circle':
            print(f"    - Main diameter: {meas['diameter_pixels']} pixels")
            print(f"    - Center at: {meas['center']}")
    
    # 4. 创建综合结果图
    print("\n[4/4] Creating combined visualization...")
    # 拼接所有结果
    h1 = np.hstack((workpiece_img, defect_result))
    h2 = np.hstack((measure_result, defect_result))  # 使用缺陷结果作为第四张
    combined = np.vstack((h1, h2))
    
    # 添加标题
    cv2.putText(combined, "ORIGINAL", (50, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "DEFECT DETECTION", (750, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "DIMENSION MEASUREMENT", (50, 530), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "DEFECT DETECTION", (750, 530), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imwrite('industrial_combined_result.jpg', combined)
    print("[OK] Combined result saved as 'industrial_combined_result.jpg'")
    
    # 尝试显示图像
    print("\nDisplaying results...")
    print("Press any key to close windows")
    
    try:
        cv2.imshow('Original Industrial Workpiece', workpiece_img)
        cv2.imshow('Defect Detection', defect_result)
        cv2.imshow('Dimension Measurement', measure_result)
        cv2.imshow('Combined View', combined)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("[OK] Windows closed")
    except Exception as e:
        print(f"[INFO] Could not display GUI windows: {e}")
        print("[INFO] But all result images have been saved!")
    
    print("\n" + "=" * 60)
    print("INDUSTRIAL EXAMPLE COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - industrial_workpiece.jpg (original)")
    print("  - defect_detection_result.jpg (defects marked)")
    print("  - measurement_result.jpg (dimensions measured)")
    print("  - industrial_combined_result.jpg (all views combined)")

if __name__ == '__main__':
    main()

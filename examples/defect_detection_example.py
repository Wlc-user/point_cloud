import cv2
import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.defect_detection import DefectDetector

def demo_scratch_detection(image_path):
    """
    划痕检测演示
    """
    print("\n划痕检测演示")
    print("=" * 50)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像")
        return
    
    # 创建检测器
    detector = DefectDetector()
    
    # 检测划痕
    scratches = detector.detect_scratches(image, 
                                     min_length=50, 
                                     max_width=10, 
                                     min_aspect_ratio=5.0)
    
    print(f"检测到 {len(scratches)} 条划痕")
    for i, scratch in enumerate(scratches):
        print(f"  划痕 {i+1}: 位置=({scratch['x']}, {scratch['y']}), "
              f"长度={scratch['length']:.1f}, 宽度={scratch['width']:.1f}")
    
    # 可视化
    result = detector.visualize_defects(image, scratches, show_label=True)
    
    # 显示结果
    cv2.imshow('Original', image)
    cv2.imshow('Scratch Detection', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo_surface_defects(image_path):
    """
    表面缺陷综合检测演示
    """
    print("\n表面缺陷综合检测演示")
    print("=" * 50)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像")
        return
    
    # 创建检测器
    detector = DefectDetector()
    
    # 检测各种表面缺陷
    scratches = detector.detect_scratches(image)
    dents = detector.detect_dents(image, min_area=100, max_area=5000)
    bumps = detector.detect_bumps(image, min_area=50, max_area=2000)
    
    print(f"划痕: {len(scratches)}")
    print(f"凹陷: {len(dents)}")
    print(f"凸起: {len(bumps)}")
    
    # 合并所有缺陷
    all_defects = scratches + dents + bumps
    
    # 可视化
    result = detector.visualize_defects(image, all_defects, show_label=True)
    
    # 显示结果
    cv2.imshow('Original', image)
    cv2.imshow('Surface Defects', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo_color_defects(image_path, reference_path=None):
    """
    颜色缺陷检测演示
    """
    print("\n颜色缺陷检测演示")
    print("=" * 50)
    
    # 读取图像
    image = cv2.imread(image_path)
    reference = cv2.imread(reference_path) if reference_path else None
    
    if image is None:
        print("无法读取图像")
        return
    
    # 创建检测器
    detector = DefectDetector()
    
    # 检测颜色缺陷
    color_defects = detector.detect_color_defects(image, reference, threshold=30)
    
    print(f"检测到 {len(color_defects)} 个颜色缺陷")
    for i, defect in enumerate(color_defects):
        print(f"  缺陷 {i+1}: 位置=({defect['x']}, {defect['y']}), "
              f"面积={defect['area']}")
    
    # 可视化
    result = detector.visualize_defects(image, color_defects, show_label=True)
    
    # 显示结果
    cv2.imshow('Original', image)
    if reference is not None:
        cv2.imshow('Reference', reference)
    cv2.imshow('Color Defects', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo_texture_defects(image_path):
    """
    纹理缺陷检测演示
    """
    print("\n纹理缺陷检测演示")
    print("=" * 50)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像")
        return
    
    # 创建检测器
    detector = DefectDetector()
    
    # 检测纹理缺陷
    texture_defects = detector.detect_texture_defects(image, 
                                                  radius=3, 
                                                  block_size=32,
                                                  threshold=0.5)
    
    print(f"检测到 {len(texture_defects)} 个纹理缺陷")
    for i, defect in enumerate(texture_defects):
        print(f"  缺陷 {i+1}: 位置=({defect['x']}, {defect['y']}), "
              f"纹理差异={defect['texture_diff']:.3f}")
    
    # 可视化
    result = detector.visualize_defects(image, texture_defects, show_label=True)
    
    # 显示结果
    cv2.imshow('Original', image)
    cv2.imshow('Texture Defects', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo_edge_defects(image_path):
    """
    边缘缺陷检测演示
    """
    print("\n边缘缺陷检测演示")
    print("=" * 50)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像")
        return
    
    # 创建检测器
    detector = DefectDetector()
    
    # 检测边缘缺陷
    edge_defects = detector.detect_edge_defects(image, 
                                             min_length=10, 
                                             max_length=100)
    
    print(f"检测到 {len(edge_defects)} 个边缘缺陷")
    for i, defect in enumerate(edge_defects):
        print(f"  缺陷 {i+1}: 类型={defect['type']}, "
              f"凸性={defect['convexity']:.2f}")
    
    # 可视化
    result = detector.visualize_defects(image, edge_defects, show_label=True)
    
    # 显示结果
    cv2.imshow('Original', image)
    cv2.imshow('Edge Defects', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo_welding_defects(image_path):
    """
    焊接缺陷检测演示
    """
    print("\n焊接缺陷检测演示")
    print("=" * 50)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像")
        return
    
    # 创建检测器
    detector = DefectDetector()
    
    # 检测焊接缺陷
    welding_defects = detector.detect_welding_defects(image, 
                                                   min_radius=2, 
                                                   max_radius=10)
    
    # 分类统计
    porosity = [d for d in welding_defects if d['type'] == 'porosity']
    cracks = [d for d in welding_defects if d['type'] == 'crack']
    
    print(f"气孔: {len(porosity)}")
    print(f"裂纹: {len(cracks)}")
    
    # 可视化
    result = detector.visualize_defects(image, welding_defects, show_label=True)
    
    # 显示结果
    cv2.imshow('Original', image)
    cv2.imshow('Welding Defects', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo_pcb_defects(image_path):
    """
    PCB缺陷检测演示
    """
    print("\nPCB缺陷检测演示")
    print("=" * 50)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像")
        return
    
    # 创建检测器
    detector = DefectDetector()
    
    # 检测PCB缺陷
    pcb_defects = detector.detect_pcb_defects(image, min_area=100)
    
    # 分类统计
    short_circuits = [d for d in pcb_defects if d['type'] == 'short_circuit']
    open_circuits = [d for d in pcb_defects if d['type'] == 'open_circuit']
    
    print(f"短路: {len(short_circuits)}")
    print(f"开路: {len(open_circuits)}")
    
    # 可视化
    result = detector.visualize_defects(image, pcb_defects, show_label=True)
    
    # 显示结果
    cv2.imshow('Original', image)
    cv2.imshow('PCB Defects', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo_dimensional_defects(image_path):
    """
    尺寸缺陷检测演示
    """
    print("\n尺寸缺陷检测演示")
    print("=" * 50)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像")
        return
    
    # 创建检测器
    detector = DefectDetector()
    
    # 定义期望尺寸
    expected_dimensions = {
        'width': 200,
        'height': 150
    }
    
    # 检测尺寸缺陷
    dimensional_defects = detector.detect_dimensional_defects(
        image, expected_dimensions, tolerance=0.1)
    
    print(f"检测到 {len(dimensional_defects)} 个尺寸缺陷")
    for i, defect in enumerate(dimensional_defects):
        defect_type = defect['type']
        if defect_type == 'width_defect':
            print(f"  缺陷 {i+1}: 宽度缺陷, "
                  f"期望={defect['expected_width']}, "
                  f"实际={defect['actual_width']}, "
                  f"误差={defect['width_error']*100:.1f}%")
        else:
            print(f"  缺陷 {i+1}: 高度缺陷, "
                  f"期望={defect['expected_height']}, "
                  f"实际={defect['actual_height']}, "
                  f"误差={defect['height_error']*100:.1f}%")
    
    # 可视化
    result = detector.visualize_defects(image, dimensional_defects, show_label=True)
    
    # 显示结果
    cv2.imshow('Original', image)
    cv2.imshow('Dimensional Defects', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo_comprehensive_defect_detection(image_path):
    """
    综合缺陷检测演示
    """
    print("\n综合缺陷检测演示")
    print("=" * 50)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像")
        return
    
    # 创建检测器
    detector = DefectDetector()
    
    # 执行所有检测
    all_defects = []
    
    # 1. 表面缺陷
    scratches = detector.detect_scratches(image)
    dents = detector.detect_dents(image)
    bumps = detector.detect_bumps(image)
    all_defects.extend(scratches + dents + bumps)
    
    # 2. 颜色和纹理缺陷
    color_defects = detector.detect_color_defects(image)
    texture_defects = detector.detect_texture_defects(image)
    all_defects.extend(color_defects + texture_defects)
    
    # 3. 边缘缺陷
    edge_defects = detector.detect_edge_defects(image)
    all_defects.extend(edge_defects)
    
    # 统计结果
    defect_types = {}
    for defect in all_defects:
        defect_type = defect['type']
        defect_types[defect_type] = defect_types.get(defect_type, 0) + 1
    
    print("\n缺陷统计:")
    for defect_type, count in defect_types.items():
        print(f"  {defect_type}: {count}")
    
    print(f"\n总计: {len(all_defects)} 个缺陷")
    
    # 可视化
    result = detector.visualize_defects(image, all_defects, show_label=True)
    
    # 显示结果
    cv2.imshow('Original', image)
    cv2.imshow('All Defects', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_sample_defect_image():
    """
    创建包含各种缺陷的示例图像
    """
    # 创建白色背景
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # 添加划痕
    cv2.line(image, (50, 100), (300, 100), (0, 0, 0), 2)
    cv2.line(image, (350, 200), (550, 200), (0, 0, 0), 3)
    
    # 添加凹陷（暗色圆形）
    cv2.circle(image, (150, 250), 20, (100, 100, 100), -1)
    cv2.circle(image, (450, 150), 15, (100, 100, 100), -1)
    
    # 添加凸起（亮色圆形）
    cv2.circle(image, (250, 350), 25, (200, 200, 200), -1)
    
    # 添加污渍
    cv2.ellipse(image, (400, 300), (40, 30), 0, 0, 360, (150, 150, 150), -1)
    
    return image

def demo_with_sample_image():
    """
    使用示例图像进行演示
    """
    print("\n使用示例图像进行缺陷检测演示")
    print("=" * 50)
    
    # 创建示例图像
    image = create_sample_defect_image()
    
    # 保存示例图像
    cv2.imwrite('sample_defects.jpg', image)
    print("已创建示例图像: sample_defects.jpg")
    
    # 创建检测器
    detector = DefectDetector()
    
    # 执行综合检测
    all_defects = []
    
    # 检测各种缺陷
    scratches = detector.detect_scratches(image)
    dents = detector.detect_dents(image)
    bumps = detector.detect_bumps(image)
    stains = detector.detect_stains(image)
    
    all_defects.extend(scratches + dents + bumps + stains)
    
    # 统计结果
    defect_types = {}
    for defect in all_defects:
        defect_type = defect['type']
        defect_types[defect_type] = defect_types.get(defect_type, 0) + 1
    
    print("\n检测结果:")
    for defect_type, count in defect_types.items():
        print(f"  {defect_type}: {count}")
    
    print(f"\n总计: {len(all_defects)} 个缺陷")
    
    # 可视化
    result = detector.visualize_defects(image, all_defects, show_label=True)
    
    # 显示结果
    cv2.imshow('Sample Image', image)
    cv2.imshow('Detected Defects', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='缺陷检测演示')
    parser.add_argument('--mode', type=str, default='sample',
                       choices=['scratch', 'surface', 'color', 'texture', 
                               'edge', 'welding', 'pcb', 'dimensional', 
                               'comprehensive', 'sample'],
                       help='演示模式')
    parser.add_argument('--image', type=str, default=None,
                       help='输入图像路径')
    parser.add_argument('--reference', type=str, default=None,
                       help='参考图像路径（用于颜色缺陷检测）')
    
    args = parser.parse_args()
    
    if args.mode == 'sample':
        demo_with_sample_image()
    elif args.image is None:
        print("请指定输入图像路径 (--image)")
    elif args.mode == 'scratch':
        demo_scratch_detection(args.image)
    elif args.mode == 'surface':
        demo_surface_defects(args.image)
    elif args.mode == 'color':
        demo_color_defects(args.image, args.reference)
    elif args.mode == 'texture':
        demo_texture_defects(args.image)
    elif args.mode == 'edge':
        demo_edge_defects(args.image)
    elif args.mode == 'welding':
        demo_welding_defects(args.image)
    elif args.mode == 'pcb':
        demo_pcb_defects(args.image)
    elif args.mode == 'dimensional':
        demo_dimensional_defects(args.image)
    elif args.mode == 'comprehensive':
        demo_comprehensive_defect_detection(args.image)
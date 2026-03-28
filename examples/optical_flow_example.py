import cv2
import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.optical_flow import OpticalFlowAnalyzer

def demo_optical_flow_video(video_path=None):
    """
    光流分析视频演示
    
    Args:
        video_path: 视频文件路径，如果为None则使用摄像头
    """
    # 打开视频源
    if video_path is None:
        cap = cv2.VideoCapture(0)
        print("使用摄像头进行演示")
    else:
        cap = cv2.VideoCapture(video_path)
        print(f"使用视频文件: {video_path}")
    
    if not cap.isOpened():
        print("无法打开视频源")
        return
    
    # 创建光流分析器
    analyzer = OpticalFlowAnalyzer(max_history=30)
    
    # 创建显示窗口
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Optical Flow', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Flow Arrows', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Motion Regions', cv2.WINDOW_NORMAL)
    
    frame_count = 0
    
    print("\n光流分析演示")
    print("按 'q' 退出")
    print("按 'r' 重置分析器")
    print("-" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if video_path is not None:
                # 视频结束，重新开始
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        
        frame_count += 1
        
        # 计算稠密光流
        flow = analyzer.calc_optical_flow_farneback(frame)
        
        if flow is not None:
            # 1. 光流统计分析
            stats = analyzer.analyze_flow_statistics(flow)
            
            # 2. 运动区域检测
            motion_regions = analyzer.detect_motion_regions(flow, threshold=2.0, min_area=100)
            
            # 3. 异常检测
            anomaly = analyzer.detect_anomaly(flow, threshold_sigma=3.0)
            
            # 4. 运动模式分类
            pattern = analyzer.classify_motion_pattern(flow)
            
            # 5. 速度估计
            velocity = analyzer.estimate_velocity(flow, pixel_scale=0.1, fps=30.0)
            
            # 打印统计信息（每10帧）
            if frame_count % 10 == 0:
                print(f"\n帧 {frame_count}:")
                print(f"  平均光流幅度: {stats['mean_magnitude']:.2f}")
                print(f"  运动模式: {pattern['pattern']} (置信度: {pattern['confidence']:.2f})")
                print(f"  检测到的运动区域: {len(motion_regions)}")
                
                if anomaly['is_anomaly']:
                    print(f"  ⚠️  异常检测: {anomaly['anomaly_rules']}")
                    print(f"     Z-score: {anomaly['z_score']:.2f}")
                
                print(f"  估计速度: {velocity['velocity_magnitude']:.2f} units/s, 方向: {velocity['direction']}")
            
            # 可视化
            # 光流场可视化
            flow_vis = analyzer.visualize_flow(frame, flow, overlay=True, alpha=0.5)
            
            # 光流向量箭头
            flow_arrows = analyzer.draw_flow_arrows(frame, flow, step=16, 
                                                   min_magnitude=1.0, color=(0, 255, 0))
            
            # 运动区域标注
            motion_vis = frame.copy()
            for i, region in enumerate(motion_regions):
                x, y, w, h = region['x'], region['y'], region['width'], region['height']
                cv2.rectangle(motion_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(motion_vis, f"Region {i+1}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 添加文字信息
            info_text = [
                f"Frame: {frame_count}",
                f"Mean Magnitude: {stats['mean_magnitude']:.2f}",
                f"Pattern: {pattern['pattern']}",
                f"Regions: {len(motion_regions)}"
            ]
            
            if anomaly['is_anomaly']:
                info_text.append("ANOMALY DETECTED!")
            
            y_offset = 30
            for text in info_text:
                color = (0, 0, 255) if "ANOMALY" in text else (255, 255, 255)
                cv2.putText(motion_vis, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25
            
            # 显示结果
            cv2.imshow('Original', frame)
            cv2.imshow('Optical Flow', flow_vis)
            cv2.imshow('Flow Arrows', flow_arrows)
            cv2.imshow('Motion Regions', motion_vis)
        
        # 键盘控制
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            analyzer.reset()
            print("\n分析器已重置")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("\n演示结束")

def demo_lucas_kanade(video_path=None):
    """
    Lucas-Kanade稀疏光流演示
    
    Args:
        video_path: 视频文件路径
    """
    if video_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("无法打开视频源")
        return
    
    analyzer = OpticalFlowAnalyzer()
    
    # 特征点参数
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )
    
    # LK参数
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # 创建掩码用于绘制轨迹
    mask = None
    
    print("\nLucas-Kanade光流演示")
    print("按 'q' 退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 计算稀疏光流
        p0, p1, st = analyzer.calc_optical_flow_lucas_kanade(frame, feature_params, lk_params)
        
        if p0 is not None and p1 is not None and st is not None:
            # 选择好的点
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            # 初始化掩码
            if mask is None:
                mask = np.zeros_like(frame)
            
            # 绘制轨迹
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                
                # 绘制轨迹线
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), 
                              (0, 255, 0), 2)
                # 绘制当前点
                frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
            
            # 叠加轨迹
            img = cv2.add(frame, mask)
            
            cv2.imshow('Lucas-Kanade Optical Flow', img)
        else:
            cv2.imshow('Lucas-Kanade Optical Flow', frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def demo_motion_analysis():
    """
    运动分析综合演示
    """
    print("\n运动分析综合演示")
    print("=" * 50)
    
    # 创建模拟数据
    np.random.seed(42)
    
    # 模拟不同类型的运动
    motion_types = ['static', 'uniform', 'acceleration', 'rotation', 'vibration']
    
    for motion_type in motion_types:
        print(f"\n模拟运动类型: {motion_type}")
        
        # 创建模拟光流场
        h, w = 100, 100
        
        if motion_type == 'static':
            flow = np.random.randn(h, w, 2) * 0.1
        
        elif motion_type == 'uniform':
            flow = np.ones((h, w, 2)) * 2.0 + np.random.randn(h, w, 2) * 0.2
        
        elif motion_type == 'acceleration':
            flow = np.random.randn(h, w, 2) * 3.0
        
        elif motion_type == 'rotation':
            # 创建旋转场
            y, x = np.mgrid[0:h, 0:w]
            cx, cy = w // 2, h // 2
            dx = -(y - cy) * 0.1
            dy = (x - cx) * 0.1
            flow = np.stack([dx, dy], axis=2) + np.random.randn(h, w, 2) * 0.5
        
        elif motion_type == 'vibration':
            flow = np.random.randn(h, w, 2) * 2.0
        
        # 创建分析器
        analyzer = OpticalFlowAnalyzer()
        analyzer.prev_frame = np.zeros((h, w), dtype=np.uint8)
        
        # 分析
        stats = analyzer.analyze_flow_statistics(flow)
        pattern = analyzer.classify_motion_pattern(flow)
        
        print(f"  平均幅度: {stats['mean_magnitude']:.2f}")
        print(f"  幅度标准差: {stats['std_magnitude']:.2f}")
        print(f"  检测到的模式: {pattern['pattern']} (置信度: {pattern['confidence']:.2f})")
    
    print("\n演示完成")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='光流分析演示')
    parser.add_argument('--mode', type=str, default='video',
                       choices=['video', 'lk', 'analysis'],
                       help='演示模式')
    parser.add_argument('--source', type=str, default=None,
                       help='视频文件路径')
    
    args = parser.parse_args()
    
    if args.mode == 'video':
        demo_optical_flow_video(args.source)
    elif args.mode == 'lk':
        demo_lucas_kanade(args.source)
    elif args.mode == 'analysis':
        demo_motion_analysis()
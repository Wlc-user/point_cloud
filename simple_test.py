import sys
import os
import cv2
import numpy as np

# 确保可以找到src模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing OpenCV import...")
print("[OK] NumPy imported successfully")
print("[OK] OpenCV imported successfully: version", cv2.__version__)

print("\nTrying to read test image...")
if os.path.exists('test_image.jpg'):
    img = cv2.imread('test_image.jpg')
    if img is not None:
        print(f"[OK] Image read successfully, size: {img.shape}")
        
        # Simple image processing test
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 将边缘图像转换为彩色以便并排显示
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 并排拼接原始图像和边缘检测结果
        combined = np.hstack((img, edges_color))
        
        # Save results
        cv2.imwrite('output_edges.jpg', edges)
        cv2.imwrite('output_combined.jpg', combined)
        print("[OK] Results saved")
        
        # 显示图像窗口
        print("\nDisplaying images...")
        print("Press any key to close the windows")
        
        try:
            cv2.imshow('Original Image', img)
            cv2.imshow('Edge Detection', edges)
            cv2.imshow('Combined View', combined)
            
            # 等待按键
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("[OK] Windows closed")
        except Exception as e:
            print(f"[INFO] Could not display GUI windows: {e}")
            print("[INFO] But results have been saved to files:")
            print("  - test_image.jpg (original)")
            print("  - output_edges.jpg (edges)")
            print("  - output_combined.jpg (combined)")
    else:
        print("[ERROR] Failed to read image")
else:
    print("[ERROR] Test image not found")

print("\nProject environment test complete!")

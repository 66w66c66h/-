import os
from pathlib import Path

def check_data_path():
    """检查数据路径是否正确"""
    
    # 尝试不同的路径格式
    test_paths = [
        r"D:\大文件\文献\齿轮箱故障诊断\东南大学齿轮数据集",
        "D:/大文件/文献/齿轮箱故障诊断/东南大学齿轮数据集",
        r"D:\\大文件\\文献\\齿轮箱故障诊断\\东南大学齿轮数据集",
    ]
    
    for i, path in enumerate(test_paths, 1):
        print(f"\n{i}. 测试路径: {path}")
        
        if os.path.exists(path):
            print("   ✓ 路径存在")
            if os.path.isdir(path):
                print("   ✓ 是有效目录")
                # 检查子文件夹
                sub_folders = ['gearset(1)', 'bearingset(1)']
                for sub in sub_folders:
                    sub_path = os.path.join(path, sub)
                    if os.path.exists(sub_path):
                        print(f"   ✓ 找到子文件夹: {sub}")
                    else:
                        print(f"   ✗ 未找到子文件夹: {sub}")
            else:
                print("   ✗ 不是目录")
        else:
            print("   ✗ 路径不存在")
    
    print("\n" + "="*50)
    print("建议：")
    print("1. 如果路径不存在，请检查文件夹位置")
    print("2. 右键点击文件夹 → 属性 → 复制完整路径")
    print("3. 将复制的路径粘贴到 main.py 中")

if __name__ == "__main__":
    check_data_path()
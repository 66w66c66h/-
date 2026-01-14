import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from natsort import natsorted

def load_seu_gearbox_data(root_path):
    sub_folders = ['bearingset(1)', 'gearset(1)']
    csv_acc_data = []
    
    for folder in sub_folders:
        folder_path = os.path.join(root_path, folder)
        if not os.path.exists(folder_path):
            print(f"警告：子文件夹不存在 → {folder_path}")
            continue
        
        # 1. 获取文件夹中的所有CSV文件
        csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
        if not csv_files:
            print(f"警告：子文件夹中无CSV文件 → {folder_path}")
            continue
        
        # 2. 遍历CSV文件，拼接完整路径
        for file_name in csv_files:
            # 拼接完整文件路径：folder_path + file_name
            file_path = os.path.join(folder_path, file_name)
            if not os.path.isfile(file_path):
                print(f"警告：不是有效文件 → {file_path}")
                continue
            
            # 3. 读取文件（增加异常捕获）
            try:
             # 尝试读取数据
                
                csv_acc_all = pd.read_csv(
                    file_path, 
                    header=None, 
                    skiprows=19,  # 跳过前19行，从第20行开始
                    nrows=1000,  # 仅读取1000行数据
                    encoding='utf-8', 
                    low_memory=False,  # 关闭分块读取，避免类型推断错误
                    on_bad_lines='skip'  # 跳过错误行
                ).apply(pd.to_numeric, errors='coerce') 
                
                csv_acc_all = csv_acc_all.fillna(0)  # 用0填充空值
                csv_acc_all = csv_acc_all.values  # 转为numpy数组
            except UnicodeDecodeError:
                try:
                    csv_acc_all = pd.read_csv(
                        file_path, 
                        header=None, 
                        skiprows=19,
                        nrows=1000,  # 仅读取1000行数据  
                        encoding='utf-8', 
                        low_memory=False,  # 关闭分块读取，避免类型推断错误
                        on_bad_lines='skip'  # 跳过错误行
                     ).apply(pd.to_numeric, errors='coerce').fillna(0).values
                except UnicodeDecodeError as e:
                    print(f"错误：文件编码不支持 → {file_path}，错误：{str(e)}")
                    continue
            except pd.errors.EmptyDataError:
                print(f"警告：文件为空 → {file_path}")
                continue
            except pd.errors.ParserError:
                print(f"警告：文件解析错误 → {file_path}")
                continue
            except Exception as e:
                print(f"错误：读取文件失败 → {file_path}，错误：{str(e)}")
                continue
            
            # 5. 处理数据并添加到列表
            data = csv_acc_all[9:982, :8]
            load_config = '20-0' if '20_0' in file_name else '30-2'
            csv_acc_data.append({
                'sub_folder': folder,
                'file_name': file_name,
                'load_config': load_config,
                'data': data,
                'signal_info': {
                    0: '电机振动',
                    1: '行星齿轮箱x方向振动',
                    2: '行星齿轮箱y方向振动',
                    3: '行星齿轮箱z方向振动',
                    4: '电机扭矩',
                    5: '并联齿轮箱x方向振动',
                    6: '并联齿轮箱y方向振动',
                    7: '并联齿轮箱z方向振动'
                }
            })
    
    print(f"数据读取完成！共成功加载 {len(csv_acc_data)} 个CSV文件")
    return csv_acc_data

# 保持plot_time_domain_signal函数不变
def plot_time_domain_signal(csv_acc_data):
    if len(csv_acc_data) == 0:
        print("错误：无数据可绘制时域图！")
        return
    
    plt.rcParams["font.family"] = ["SimHei","Microsoft YaHei",  "Arial Unicode MS"] 
    plt.rcParams["axes.unicode_minus"] = False 
   
    try:
        data = csv_acc_data[0]['data']
        if data.size == 0:
            print("错误：数据为空，无法绘图！")
            return
        f = data[:, 1]
    except (IndexError, KeyError) as e:
        print(f"数据格式错误：{str(e)}")
        return
    
    # 3. 绘图并显式指定字体
    plt.figure(figsize=(10, 4))
    # f = f[::10]  # 每隔10个点取一个，减少数据量
    plt.plot(f)
    
    # 显式设置字体（关键：覆盖坐标轴/标题的字体）
    font = {'family': 'SimHei', 'size': 10}  # 定义字体字典
    plt.title(f'时域振动信号（{csv_acc_data[0]["signal_info"][1]}）', fontdict=font)
    plt.xlabel('采样点', fontdict=font)
    plt.ylabel('幅值', fontdict=font) 
    
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
    # 替换plt.show()为保存图片
    plt.savefig('time_domain_signal.png', dpi=300)  # 保存到当前目录
    print("时域图已保存为：time_domain_signal.png")
    plt.close()  # 关闭画布释放资源




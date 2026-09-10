import os
import matplotlib.pyplot as plt

def create_folder_and_save_plot():
    """
    在当前目录创建 'fig' 文件夹，绘制一个简单的图，并保存到该文件夹。
    """
    # 1. 定义要创建的文件夹名称
    folder_name = "fig"

    # 2. 在当前脚本所在位置创建文件夹
    # exist_ok=True 意味着如果文件夹已经存在，os.makedirs 不会报错
    try:
        os.makedirs(folder_name, exist_ok=True)
        print(f"文件夹 '{folder_name}' 已创建或已存在。")
    except OSError as e:
        print(f"创建文件夹 '{folder_name}' 失败: {e}")
        return # 如果创建失败，则退出函数

    # 3. 准备一些简单的数据来画图
    x_values = [1, 2, 3, 4, 5]
    y_values = [2, 4, 1, 5, 3]

    # 4. 绘制一个简单的图
    plt.figure(figsize=(8, 6)) # 可以设置图片大小
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='blue')

    # 添加标题和轴标签
    plt.title("我的第一个Python图")
    plt.xlabel("X轴")
    plt.ylabel("Y轴")

    # 添加网格
    plt.grid(True)

    # 5. 定义保存文件的完整路径
    file_name = "simple_plot.png"
    save_path = os.path.join(folder_name, file_name)

    # 6. 保存图到指定的文件夹
    try:
        plt.savefig(save_path)
        print(f"图表已成功保存到: {save_path}")
    except Exception as e:
        print(f"保存图表失败: {e}")

    # 7. 显示图表 (可选，如果你想在脚本运行时看到图)
    plt.show()

# 调用函数来执行操作
if __name__ == "__main__":
    create_folder_and_save_plot()

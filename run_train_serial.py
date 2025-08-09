# run_train_serial.py

import subprocess
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Run train command multiple times, one after another.")
    parser.add_argument('--count', type=int, required=True, help='Number of times to run the command')

    args = parser.parse_args()

    if args.count <= 0:
        print("Count must be a positive integer.")
        sys.exit(1)

    # 构建命令（使用列表形式更安全，但这里涉及重定向，使用 shell=True + 字符串）
    command = (
        "python train.py "
        "--model_type dual_branch_enhanced "
        "--fusion_levels 0 1 "
        "--num_classes 2 "
        "--model_name UC "
        "--edge_attention none "
        "--fusion_mode gate"
    )

    # 每次运行输出到不同的日志文件，避免覆盖
    for i in range(args.count):
        log_file = f"train_{i + 1}.log"
        cmd_with_log = f"{command} > {log_file} 2>&1"

        print(f"第 {i + 1} 次运行，日志输出到: {log_file}")
        print(f"执行命令: {cmd_with_log}")

        try:
            # 同步运行，会等待训练结束
            result = subprocess.run(cmd_with_log, shell=True)

            # 可选：检查返回码
            if result.returncode == 0:
                print(f"第 {i + 1} 次运行成功完成。")
            else:
                print(f"第 {i + 1} 次运行失败，返回码: {result.returncode}")
                # 如果你想在失败时停止，取消下面这行的注释
                # break

        except KeyboardInterrupt:
            print("\n用户中断，停止运行。")
            sys.exit(0)
        except Exception as e:
            print(f"运行第 {i + 1} 次时出错: {e}")
            break

    print("所有任务运行完成。")


if __name__ == "__main__":
    main()
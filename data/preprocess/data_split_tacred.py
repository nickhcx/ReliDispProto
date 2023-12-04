num='41'

# 打开原始文件
with open(r'D:\FSRE\Demo\IncreProtoNet-main\IncreProtoNet-main\data\init_data\TACRED\tacred.txt', 'r') as infile:
    # 打开新文件用于保存符合条件的行
    with open(r'D:\FSRE\Demo\IncreProtoNet-main\IncreProtoNet-main\data\tacred\novel6\novel6_tacred.txt', 'a') as outfile:
        # 遍历原始文件的每一行
        for line in infile:
            # 使用split()函数分割每一行的文本，并获取前两个文本
            parts = line.split()
            if len(parts) >= 2 and parts[0] == num and parts[1] not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                outfile.write(line)

# 定义计数器变量
count_A = 0
count_B = 0
count_C = 0

# 打开原始文件
with open(r'D:\FSRE\Demo\IncreProtoNet-main\IncreProtoNet-main\data\tacred\novel6\novel6_tacred.txt', 'r') as infile:
    # 打开新文件用于保存符合条件的行
    with open(r'D:\FSRE\Demo\IncreProtoNet-main\IncreProtoNet-main\data\tacred\novel6\novel6_train_tacred.txt', 'a') as file_A, open(r'D:\FSRE\Demo\IncreProtoNet-main\IncreProtoNet-main\data\tacred\novel6\novel6_test_tacred.txt', 'a') as file_B, open(r'D:\FSRE\Demo\IncreProtoNet-main\IncreProtoNet-main\data\tacred\novel6\novel6_val_tacred.txt', 'a') as file_C:
        # 遍历原始文件的每一行
        for line in infile:
            # 使用 split() 函数分割每一行的文本，并获取第一个文本
            parts = line.split()
            if len(parts) >= 2 and parts[0] == '41' and parts[1] not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                # 如果第一个文本为'1'，第二个文本不是指定的数字
                if count_A < 10:
                    # 将前70个符合条件的行添加到 A.txt
                    file_A.write(line)
                    count_A += 1
                elif count_B < 3:
                    # 将接下来的20个符合条件的行添加到 B.txt
                    file_B.write(line)
                    count_B += 1
                elif count_C < 1:
                    # 将接下来的10个符合条件的行添加到 C.txt
                    file_C.write(line)
                    count_C += 1
                else:
                    # 如果已经满足要求的数量，跳出循环
                    break
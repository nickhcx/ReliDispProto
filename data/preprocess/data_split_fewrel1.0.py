# coding=utf-8
import sys
from os.path import normpath,join,dirname
# 先引入根目录路径，以后的导入将直接使用当前项目的绝对路径
sys.path.append(normpath(join(dirname(__file__), '../../')))
from utils import json_util
from utils import path_util
import numpy as np
from random import sample

def __relsIndex__(rels):
    rel2index = {}
    for i, rel in enumerate(rels):
        rel2index[rel] = i
    return rel2index


def construct_novel_train(base_train_file, novel_train_file, novel_train_file_new):

    base_train_data = json_util.load(base_train_file)

    novel_train_data = json_util.load(novel_train_file)

    base_rels = [rel for rel in base_train_data.keys()]  # 20

    novel_rels = [rel for rel in novel_train_data.keys()]

    novel_rels_new = {}

    for rel in novel_rels:
        if rel not in base_rels:
            novel_rels_new[rel] = novel_train_data[rel]

    json_util.dump(novel_rels_new, novel_train_file_new)

    return None

#划分relations
def session_split(data_file, base_rel_num, novel_rel_num, base_file, novel1_file, novel2_file, novel3_file, novel4_file, novel5_file, novel6_file):
    base_json_data = {}
    novel1_json_data = {}
    novel2_json_data = {}
    novel3_json_data = {}
    novel4_json_data = {}
    novel5_json_data = {}
    novel6_json_data = {}


    json_data = json_util.load(data_file)

    base_rels = np.random.choice(list(json_data.keys()), base_rel_num, False) #20

    selected_rels = [rel for rel in json_data.keys() if rel in base_rels]

    novel_rels = [rel for rel in json_data.keys() if rel not in selected_rels]

    novel1_rels = sample(novel_rels, novel_rel_num) + selected_rels

    selected_rels = novel1_rels

    novel_rels = [rel for rel in json_data.keys() if rel not in selected_rels]

    novel2_rels = sample(novel_rels, novel_rel_num) + selected_rels

    selected_rels = novel2_rels

    novel_rels = [rel for rel in json_data.keys() if rel not in selected_rels]

    novel3_rels = sample(novel_rels, novel_rel_num) + selected_rels

    selected_rels = novel3_rels

    novel_rels = [rel for rel in json_data.keys() if rel not in selected_rels]

    novel4_rels = sample(novel_rels, novel_rel_num) + selected_rels

    selected_rels = novel4_rels

    novel_rels = [rel for rel in json_data.keys() if rel not in selected_rels]

    novel5_rels = sample(novel_rels, novel_rel_num) + selected_rels

    selected_rels =novel5_rels

    novel_rels = [rel for rel in json_data.keys() if rel not in selected_rels]

    novel6_rels = sample(novel_rels, novel_rel_num) + selected_rels


    for rel in json_data.keys():
        if rel in base_rels:
            base_json_data[rel] = json_data[rel]
        if rel in novel1_rels:
            novel1_json_data[rel] = json_data[rel]
        if rel in novel2_rels:
            novel2_json_data[rel] = json_data[rel]
        if rel in novel3_rels:
            novel3_json_data[rel] = json_data[rel]
        if rel in novel4_rels:
            novel4_json_data[rel] = json_data[rel]
        if rel in novel5_rels:
            novel5_json_data[rel] = json_data[rel]
        if rel in novel6_rels:
            novel6_json_data[rel] = json_data[rel]

    # save
    json_util.dump(base_json_data, base_file)
    json_util.dump(novel1_json_data, novel1_file)
    json_util.dump(novel2_json_data, novel2_file)
    json_util.dump(novel3_json_data, novel3_file)
    json_util.dump(novel4_json_data, novel4_file)
    json_util.dump(novel5_json_data, novel5_file)
    json_util.dump(novel6_json_data, novel6_file)

    # 固定classes_name的顺序
    json_util.dump(__relsIndex__(base_rels), path_util.from_project_root("data/fewrel1.0/continual/baserel2index.json"))
    json_util.dump(__relsIndex__(novel1_rels), path_util.from_project_root("data/fewrel1.0/continual/novel1rel2index.json"))
    json_util.dump(__relsIndex__(novel2_rels), path_util.from_project_root("data/fewrel1.0/continual/novel2rel2index.json"))
    json_util.dump(__relsIndex__(novel3_rels), path_util.from_project_root("data/fewrel1.0/continual/novel3rel2index.json"))
    json_util.dump(__relsIndex__(novel4_rels), path_util.from_project_root("data/fewrel1.0/continual/novel4rel2index.json"))
    json_util.dump(__relsIndex__(novel5_rels), path_util.from_project_root("data/fewrel1.0/continual/novel5rel2index.json"))
    json_util.dump(__relsIndex__(novel6_rels), path_util.from_project_root("data/fewrel1.0/continual/novel6rel2index.json"))

    print("base_fewrel rels_nums: {}".format(len(base_json_data.keys())))
    print("novel1_fewrel rels_nums: {}".format(len(novel1_json_data.keys())))
    print("novel2_fewrel rels_nums: {}".format(len(novel2_json_data.keys())))
    print("novel3_fewrel rels_nums: {}".format(len(novel3_json_data.keys())))
    print("novel4_fewrel rels_nums: {}".format(len(novel4_json_data.keys())))
    print("novel5_fewrel rels_nums: {}".format(len(novel5_json_data.keys())))
    print("novel6_fewrel rels_nums: {}".format(len(novel6_json_data.keys())))

    return None

#划分samples
def train_test_split(data_file, train_file, val_file, test_file, train_rel_num):
    train_json_data = {}
    val_json_data = {}
    test_json_data = {}

    json_data = json_util.load(data_file)
    for rel in json_data.keys():
        instances = json_data[rel]
        train_json_data[rel] = instances[:train_rel_num]
        val_json_data[rel] = instances[train_rel_num:train_rel_num+val_rel_num]
        test_json_data[rel] = instances[train_rel_num+val_rel_num:]

    # save
    json_util.dump(train_json_data, train_file)
    json_util.dump(val_json_data, val_file)
    json_util.dump(test_json_data, test_file)

    print("base_train_rel_num:{}".format(train_rel_num))
    print("base_val_rel_num:{}".format(val_rel_num))
    print("base_test_rel_num:{}".format(700 - train_rel_num - val_rel_num))

    return None

if __name__ == '__main__':

    init_file = path_util.from_project_root("data/init_FewRel_data/fewrel1.0/all.json")

    base_file = path_util.from_project_root("data/fewrel1.0/continual/base_fewrel.json")
    novel1_file = path_util.from_project_root("data/fewrel1.0/continual/novel1_fewrel.json")
    novel2_file = path_util.from_project_root("data/fewrel1.0/continual/novel2_fewrel.json")
    novel3_file = path_util.from_project_root("data/fewrel1.0/continual/novel3_fewrel.json")
    novel4_file = path_util.from_project_root("data/fewrel1.0/continual/novel4_fewrel.json")
    novel5_file = path_util.from_project_root("data/fewrel1.0/continual/novel5_fewrel.json")
    novel6_file = path_util.from_project_root("data/fewrel1.0/continual/novel6_fewrel.json")

    base_rels_num = 20
    novel_rels_num = 10
    # session_split(init_file, base_rels_num, novel_rels_num, base_file, novel1_file, novel2_file, novel3_file, novel4_file, novel5_file, novel6_file)

    """
        每个关系划分， 550: 50：100， 作为训练集、验证集、测试集
    """
    all_file = path_util.from_project_root("data/fewrel1.0/continual/novel6_fewrel.json")
    train_file = path_util.from_project_root("data/fewrel1.0/continual/fewrel_data/novel6/novel6_train_fewrel.json")
    val_file = path_util.from_project_root("data/fewrel1.0/continual/fewrel_data/novel6/novel6_val_fewrel.json")
    test_file = path_util.from_project_root("data/fewrel1.0/continual/fewrel_data/novel6/novel6_test_fewrel.json")

    train_rel_num = 550
    val_rel_num = 50
    test_rel_num = 100
    #train_test_split(all_file, train_file, val_file, test_file, train_rel_num)

    base_train_file = path_util.from_project_root("data/fewrel1.0/continual/fewrel_data/base/base_train_fewrel.json")
    novel1_train_file = path_util.from_project_root("data/fewrel1.0/continual/fewrel_data/novel1/novel1_train_fewrel.json")
    novel1_train_file_new = path_util.from_project_root("data/fewrel1.0/continual/fewrel_data/novel1/novel1_train_fewrel_new.json")
    #construct_novel_train(base_train_file, novel1_train_file, novel1_train_file_new)

    base_train_file = path_util.from_project_root("data/fewrel1.0/continual/fewrel_data/novel1/novel1_train_fewrel.json")
    novel2_train_file = path_util.from_project_root("data/fewrel1.0/continual/fewrel_data/novel2/novel2_train_fewrel.json")
    novel2_train_file_new = path_util.from_project_root("data/fewrel1.0/continual/fewrel_data/novel2/novel2_train_fewrel_new.json")
    #construct_novel_train(base_train_file, novel2_train_file, novel2_train_file_new)

    base_train_file = path_util.from_project_root("data/fewrel1.0/continual/fewrel_data/novel2/novel2_train_fewrel.json")
    novel3_train_file = path_util.from_project_root("data/fewrel1.0/continual/fewrel_data/novel3/novel3_train_fewrel.json")
    novel3_train_file_new = path_util.from_project_root("data/fewrel1.0/continual/fewrel_data/novel3/novel3_train_fewrel_new.json")
    #construct_novel_train(base_train_file, novel3_train_file, novel3_train_file_new)

    base_train_file = path_util.from_project_root("data/fewrel1.0/continual/fewrel_data/novel3/novel3_train_fewrel.json")
    novel4_train_file = path_util.from_project_root("data/fewrel1.0/continual/fewrel_data/novel4/novel4_train_fewrel.json")
    novel4_train_file_new = path_util.from_project_root("data/fewrel1.0/continual/fewrel_data/novel4/novel4_train_fewrel_new.json")
    #construct_novel_train(base_train_file, novel4_train_file, novel4_train_file_new)

    base_train_file = path_util.from_project_root("data/fewrel1.0/continual/fewrel_data/novel4/novel4_train_fewrel.json")
    novel5_train_file = path_util.from_project_root("data/fewrel1.0/continual/fewrel_data/novel5/novel5_train_fewrel.json")
    novel5_train_file_new = path_util.from_project_root("data/fewrel1.0/continual/fewrel_data/novel5/novel5_train_fewrel_new.json")
    #construct_novel_train(base_train_file, novel5_train_file, novel5_train_file_new)


    base_train_file = path_util.from_project_root("data/fewrel1.0/continual/fewrel_data/novel5/novel5_train_fewrel.json")
    novel6_train_file = path_util.from_project_root("data/fewrel1.0/continual/fewrel_data/novel6/novel6_train_fewrel.json")
    novel6_train_file_new = path_util.from_project_root("data/fewrel1.0/continual/fewrel_data/novel6/novel6_train_fewrel_new.json")
    #construct_novel_train(base_train_file, novel6_train_file, novel6_train_file_new)
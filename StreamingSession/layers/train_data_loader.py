import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json


class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, data_file, baserel2index_file, encoder, baseN, novelN, K, Q, triplet_num, bi_pos_num, bi_neg_num):
        if not os.path.exists(data_file):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(data_file))
        self.classes = list(self.json_data.keys())
        self.baseClass2index = json.load(open(baserel2index_file))
        self.encoder = encoder
        self.baseN = baseN
        self.novelN = novelN
        self.K = K
        self.Q = Q
        self.triplet_num = triplet_num
        self.bi_pos_num = bi_pos_num
        self.bi_neg_num = bi_neg_num

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word, pos1, pos2, mask 

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):

        novelSupport_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_label = []




        # 随机抽取novelN个关系作为novelRels
        novelClasses = random.sample(self.classes, self.novelN)

        # selectedClasses = []
        # selectedClasses.extend(baseClasses)
        # selectedClasses.extend(novelClasses)



        """
            构建novelRels Support Set
        """

        for i, class_name in enumerate(novelClasses):

            indices = np.random.choice(list(range(len(self.json_data[class_name]))), self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                if count < self.K:
                    self.__additem__(novelSupport_set, word, pos1, pos2, mask)
                else:
                    self.__additem__(query_set, word, pos1, pos2, mask)
                count += 1

            query_label += [i] * self.Q


        return novelSupport_set, query_set, query_label

    def __len__(self):
        return 999999999999999


def collate_fn(data):

    batch_novelSupportSet = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_novel_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_queryLabel = []
    batch_base_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}


    novelSupport_set, query_set, query_label = zip(*data)

    for i in range(len(novelSupport_set)):
        for k in novelSupport_set[i]:
            batch_novelSupportSet[k] += novelSupport_set[i][k]
        for k in query_set[i]:
            batch_query[k] += query_set[i][k]


        batch_queryLabel += query_label[i]


    for k in batch_novelSupportSet:
        batch_novelSupportSet[k] = torch.stack(batch_novelSupportSet[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)



    batch_queryLabel = torch.tensor(batch_queryLabel)

    return batch_novelSupportSet, batch_query, batch_queryLabel


def get_loader(data_file, baserel2index_file, encoder, batch_size,  baseN, novelN, K, Q, triplet_num, bi_pos_num, bi_neg_num, num_workers=4,  collate_fn=collate_fn):

    dataset = FewRelDataset(data_file, baserel2index_file, encoder, baseN, novelN, K, Q, triplet_num, bi_pos_num, bi_neg_num)

    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)

    return iter(data_loader)

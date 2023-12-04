# coding=utf-8
import random
import sys
from torch import nn
import os
import torch
from torch import optim
from transformers import AdamW, get_linear_schedule_with_warmup

class IncreFewShotREModel(nn.Module):

    def __init__(self, MetaCNN):
        nn.Module.__init__(self)
        self.MetaCNN_Encoder = nn.DataParallel(MetaCNN)

    def forward(self, novelSupport_set, query_set, query_label, base_label, K, hidden_size, baseN,
                novelN, Q, triplet_base, triplet_novel, triplet_num, margin, biQ, triplet_loss_w, is_train):

        raise NotImplementedError


class IncreFewShotREFramework:

    def __init__(self,train_data_loader, test_data_loader):
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self, model, bert_optim, learning_rate, train_iter, val_step, val_iter, save_ckpt, K, hidden_size,
              train_baseN, train_novelN, test_baseN, test_novelN, train_Q, test_Q, triplet_num, margin, biQ, triplet_w,
              warmup_step=300,
              weight_decay=1e-5,
              lr_step_size=20000, pytorch_optim=optim.SGD):

        print("Start training...")

        #  过滤之前冻结的参数，不参与第二阶段的训练
        # if bert_optim:
        #     print('Use bert optim!')
        #     parameters_to_optimize = list(model.named_parameters())
        #     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        #     parameters_to_optimize = [
        #         {'params': [p for n, p in parameters_to_optimize
        #                     if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #         {'params': [p for n, p in parameters_to_optimize
        #                     if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        #     ]
        #     optimizer = AdamW(parameters_to_optimize, lr=2e-5, correct_bias=False)
        #     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step,
        #                                                 num_training_steps=train_iter)
        # else:
        optimizer = pytorch_optim(filter(lambda p: p.requires_grad, model.parameters()), learning_rate, weight_decay=weight_decay)
        # optimizer = pytorch_optim(model.parameters(), learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)

        """
            training process  
        """

        model.train()  # 设置模型训练状态

        start_iter = 0
        best_acc = 0.0
        best_novel_acc = 0.0
        best_base_acc = 0.0
        iter_both_acc = 0.0
        iter_base_acc = 0.0
        iter_novel_acc = 0.0
        iter_sample = 0.0
        iter_loss = 0.0
        self.test_novel_support = {}
        self.test_base_label = torch.zeros(1,dtype=torch.int)
        self.test_query = {}
        self.test_query_label = torch.zeros(1,dtype=torch.int)
        for it in range(start_iter, start_iter + train_iter):
            novelSupport_set, query_set, query_label = next(self.train_data_loader)

            if torch.cuda.is_available():  # 实现gpu训练
                for k in novelSupport_set:
                    novelSupport_set[k] = novelSupport_set[k].cuda()
                for k in query_set:
                    query_set[k] = query_set[k].cuda()

                query_label = query_label.cuda()


            both_acc, preds, loss, logits = model(novelSupport_set, query_set, query_label, None, K, hidden_size, train_baseN,
                                     train_novelN, train_Q, triplet_num, margin, biQ, triplet_w, True)

            # loss.requires_grad_(True)
            loss.backward()
            # 优化
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            iter_both_acc += self.item(both_acc.data)

            iter_sample += 1
            iter_loss += self.item(loss.data)

            print('step: {0:4} | loss: {1:2.6f}, both_acc: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_both_acc / iter_sample,) + '\r')

            # with open(r"D:\FSRE\IncreProtoNet-main\log.txt", 'a') as f:
            #     print( 'step: {0:4} | loss: {1:2.6f}, both_acc: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_both_acc / iter_sample,) + '\r')

            # 验证模型
            if (it + 1) % val_step == 0:
                both_acc = self.eval(model, val_iter, K, hidden_size, test_baseN, test_novelN, test_Q, biQ)
                model.train()

                iter_sample = 0.0
                iter_both_acc = 0.0
                iter_base_acc = 0.0
                iter_novel_acc = 0.0
                iter_loss = 0.0
                if both_acc > best_acc:
                    print("Best checkpoint")
                    torch.save({'state_dict': model.state_dict()}, save_ckpt + '{}.pth.tar'.format(it + 1))
                    best_acc = both_acc

                # if os.path.isfile(
                #         r"D:\FSRE\IncreProtoNet-main\IncrementalFewShotModel\checkpoint\IncreFewShotProto-bert-K-1-m-10.0-triplet_w-1.0-att-False.pth.tar"):
                #     print("file is here")
                #     base_dict = model.state_dict()
                #     pre_state_dict = \
                #         torch.load(
                #             r"D:\FSRE\IncreProtoNet-main\IncrementalFewShotModel\checkpoint\IncreFewShotProto-bert-K-1-m-10.0-triplet_w-1.0-att-False.pth.tar")[
                #             'state_dict']
                #     new_state_dict = {k: v for k, v in pre_state_dict.items() if k in base_dict}
                #     base_dict.update(new_state_dict)
                #     model.load_state_dict(base_dict)
                #     model.train()
                #
                # else:
                #     print("file is not here")
                #
                #     model.train()

    def eval(self, model, val_iter, K, hidden_size, test_baseN, test_novelN, test_Q, biQ):
        print("")



        # # Load pre-trained BERT (HCX)
        # base_dict = model.state_dict()
        # pre_state_dict = torch.load(r"D:\FSRE\IncreProtoNet-main\IncrementalFewShotModel\checkpoint\IncreFewShotProto-bert-K-5-m-10.0-triplet_w-1.0-att-False1100.pth.tar")['state_dict']
        # new_state_dict = {k: v for k, v in pre_state_dict.items() if k in base_dict}
        # base_dict.update(new_state_dict)
        # model.load_state_dict(base_dict)

        model.eval()

        model.baseModel.eval()

        iter_sample = 0.0
        iter_both_acc = 0.0
        iter_base_acc = 0.0
        iter_novel_acc = 0.0

        with torch.no_grad():
            for it in range(val_iter):

                novelSupport_set, query_set, query_label = next(self.test_data_loader)

                # self.test_query_label = torch.cat((self.test_query_label, query_label), dim=0)
                #
                # self.test_base_label = torch.cat((self.test_base_label, base_label), dim=0)
                #
                #
                # if self.test_query == {}:
                #     for key,value in query_set.items():
                #         self.test_query[key] = value
                #     for key, value in novelSupport_set.items():
                #         self.test_novel_support[key] = value
                # else:
                #     random_index = [random.randint(0, self.test_query_label.size()[0]-2) for _ in range(query_label.size()[0])]
                #     for key in self.test_query.keys():
                #         self.test_query[key] = torch.cat((self.test_query[key],query_set[key]), dim=0)
                #     for key in query_set.keys():
                #         for i in range(query_set[key].size()[0]):
                #             query_set[key][i] = self.test_query[key][random_index[i]]
                #     query_label[i] = self.test_query_label[1:][random_index[i]]
                #
                #     for key in self.test_novel_support.keys():
                #         self.test_novel_support[key] = torch.cat((self.test_novel_support[key], novelSupport_set[key]), dim=0)
                #     for key in novelSupport_set.keys():
                #         for i in range(novelSupport_set[key].size()[0]):
                #             novelSupport_set[key][i] = self.test_novel_support[key][random_index[i]]
                #     base_label[i] = self.test_base_label[1:][random_index[i]]

                if torch.cuda.is_available():  # 实现gpu训练
                    for k in novelSupport_set:
                        novelSupport_set[k] = novelSupport_set[k].cuda()
                    for k in query_set:
                        query_set[k] = query_set[k].cuda()
                    query_label = query_label.cuda()


                both_acc, preds, loss, logits = model(novelSupport_set, query_set, query_label, None, K, hidden_size, test_baseN,
                                     test_novelN, test_Q,  0,  0.0, biQ, 0.0, False)

                iter_sample += 1
                iter_both_acc += self.item(both_acc.data)


                print('[EVAL] step: {0:4} | both_acc: {1:3.2f}%'.format(it + 1,
                                                                       100 * iter_both_acc / iter_sample) + '\r')



        return iter_both_acc/iter_sample








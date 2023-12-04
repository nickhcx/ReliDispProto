# coding=utf-8
import sys
from os.path import normpath,join,dirname
sys.path.append(normpath(join(dirname(__file__), '..')))
from torch import nn
import torch
from StreamingSession.layers import framework as framework
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import torch.distributions as dist
import random
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

class IncreProto(framework.IncreFewShotREModel):

    def __init__(self, baseModel, Meta_CNN, topKEmbed, top_K):
        framework.IncreFewShotREModel.__init__(self, Meta_CNN)

        self.baseModel = baseModel

        # Load pre-trained BERT
        base_bert_dict = self.baseModel.state_dict()
        pre_state_dict = torch.load(r"your checkpointpath")['state_dict']
        new_state_dict = {k: v for k, v in pre_state_dict.items() if k in base_bert_dict}
        base_bert_dict.update(new_state_dict)
        self.baseModel.load_state_dict(base_bert_dict)

        hidden_size = self.baseModel.prototypes.size(-1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.base_linear_layer = nn.Linear(hidden_size, 1)
        self.novel_linear_layer = nn.Linear(hidden_size, 1)

        # shape = [num_rels, topk, hidden_size]
        self.top_K = top_K
        self.topKEmbed = torch.nn.Parameter(torch.tensor(topKEmbed), requires_grad=False).float()

        self.cost = nn.CrossEntropyLoss()

    def forward(self, novelSupport_set, query_set, query_label, base_label, K, hidden_size, baseN, novelN, Q, triplet_num, margin, biQ, triplet_loss_w, is_train):
        """
        :param novelSupport_set: shape=[B*novelN*K, max_length]
        :param query_set: shape=[B*(baseN+novelN)*Q, hidden_size]
        :param query_label: shape=[B*(baseN+novelN)*Q]
        :param base_label: shape=[B*baseN]
        :param K: k shot
        :param hidden_size:
        :param baseN:
        :param novelN:
        :param base_query: shape=[B*baseN*2*Q, hidden_size]
        :param base_query_label: shape =[B*baseN*2*Q]
        :param novel_query: shape=[B*novelN*2*Q, hidden_size]
        :param novel_query_label: shape =[B*novelN*2Q]
        :param is_train: True or False
        :return:
        """
        """
            sentence embedding 
        """
        with torch.no_grad():
            # shape = [B*novelN*K, sen_len, hidden_size]

            # novel_support_teacher.shape=[200,230]
            novel_support_w, novel_support_teacher = self.baseModel.sentence_encoder(novelSupport_set) #

            # shape=[B*(baseN+novelN)*Q, hidden_size] both_query.shape=[800,230]
            both_query_w, both_query = self.baseModel.sentence_encoder(query_set)


        #[4, 10, 230] 由baseEncoder得到的novel prototypes 由 both_query 来评估
        teacher_prototypes, disps_teacher = self.teacher_prototype_generator(novel_support_teacher, novelN, K, hidden_size)

        # novel_prototypes.shape=[B, novelN, hidden_size]  #[4, 10, 230]         # student_query.shape=[4,1,10,230]
        student_prototypes, disps_student, novel_support_student = self.student_prototypes_generator(novel_support_w, novelN, K, hidden_size)

        """
            novel_prototype_adapter  
        """
        if is_train:
            """
                         Teacher classifier
            """
            # teacher_query.shape=[4,200,1,230]
            teacher_query = both_query.view(-1, (baseN + novelN) * Q, hidden_size).unsqueeze(2)
            # teacher_prototypes.shape=[4,1,10,230]
            teacher_prototypes = teacher_prototypes.unsqueeze(1)
            reliability_teacher_dista = self.reliability_dista(teacher_query, teacher_prototypes, K)
            reliability_teacher_distr = self.reliability_distr(teacher_prototypes, novel_support_teacher, K, hidden_size, novelN)
            # logits_teacher.shape=[4, 200, 10]
            logits_teacher = self.logits(torch.sqrt(torch.pow(teacher_query - teacher_prototypes, 2).sum(-1)), reliability_teacher_dista, reliability_teacher_distr, disps_teacher)
            # teacher_preds=[4, 200]
            sum_dist_teacher = torch.sum(logits_teacher, -1).unsqueeze(2)
            soft_labels_teacher_dist = F.softmax((torch.ones(sum_dist_teacher.size()).cuda() - (torch.argmin(logits_teacher, -1).unsqueeze(2) / sum_dist_teacher)), dim=-1)

            """
                 Student classifier
            """
            # student_query.shape=[4,200,1,230]
            student_query = self.MetaCNN_Encoder(both_query_w).view(-1, (baseN + novelN) * Q, hidden_size).unsqueeze(2)
            student_prototypes = student_prototypes.unsqueeze(1)
            reliability_student_dista = self.reliability_dista(student_query, student_prototypes, K)
            reliability_student_distr = self.reliability_distr(student_prototypes.squeeze(1), novel_support_student.view(-1, hidden_size), K, hidden_size, novelN)
            # logits_student.shape=[4, 200, 10]
            logits_student = self.logits(torch.sqrt(torch.pow(student_query - student_prototypes, 2).sum(-1)), reliability_student_dista, reliability_student_distr, disps_student)
            # student_preds=[4, 200] soft_labels_student.shape=[4, 200, 10]
            sum_dist_student = torch.sum(logits_student, -1).unsqueeze(2)
            soft_labels_student_dist = F.log_softmax((torch.ones(sum_dist_student.size()).cuda() - (torch.argmin(logits_student, -1).unsqueeze(2) / sum_dist_student)), dim=-1)

            """
                 Merge Loss
            """
            loss_kd_dist = F.kl_div(soft_labels_student_dist, soft_labels_teacher_dist, reduction='batchmean')
            loss_kd = loss_kd_dist

            """
                classifier
            """

            # student_query.shape=[4,200,1,230]
            # merge_prototypes.shape=[4, 1, 40, 230]
            logits = self.logits(torch.sqrt(torch.pow(student_query - student_prototypes, 2).sum(-1)), reliability_student_dista, reliability_student_distr, disps_student)
            both_preds = torch.argmin(logits, -1)  # 取最短距离
            both_acc = self.accuracy(both_preds, query_label)
            both_loss = self.softmax_loss(-logits, query_label)
            is_true = (both_preds.view(-1) == query_label.view(-1)).float()  # 将布尔值转换为浮点数
            is_true = (is_true * 2) - 1
            reliability_dista = (torch.ones(torch.sum(logits, -1).unsqueeze(2).size()).cuda() - (torch.argmin(logits, -1).unsqueeze(2) * torch.sum(logits, -1).unsqueeze(2)))
            reliability_dista = (reliability_dista.view(-1) * is_true).sum(-1) / query_label.size()[0]
            zero_tensors = torch.zeros(1).to(self.device)
            loss_reli_dista, _ = torch.max(torch.cat((torch.ones(1).to(self.device) - reliability_dista, zero_tensors), -1), 0)
            loss_reli_dista = loss_reli_dista.item()
            loss_reli_distr = (torch.gather(reliability_student_distr, 1, both_preds)* is_true.view(both_preds.size()[0],-1)).sum(-1).sum(-1)


            """
                 加入dispersion的论证
            """

            # # 在最后一维上找到第二小的值及其索引
            # values, indices = torch.topk(logits.view(-1,novelN), k=2, dim=-1, largest=False)
            #
            # # 获取第二小的值
            # second_smallest_value = values[:, 1:2]
            #
            # smallest_value = values[:, 0:1]
            #
            #
            # min_and_secondmin_rate = smallest_value/ second_smallest_value
            #
            # count_above_threshold1 = torch.sum(min_and_secondmin_rate > 0.9).item()
            # count_above_threshold2 = torch.sum(min_and_secondmin_rate > 0.95).item()
            #
            # rate1 = count_above_threshold1/min_and_secondmin_rate.size()[0]
            # rate2 = count_above_threshold2/min_and_secondmin_rate.size()[0]
            #
            # matching_indices = (both_preds.view(-1,1)[:, 0] == query_label.view(-1,1)[:, 0]).nonzero().squeeze()
            #
            #
            # index1 = (min_and_secondmin_rate[:, 0] > 0.9).nonzero().squeeze()
            #
            # index2 = (min_and_secondmin_rate[:, 0] > 0.95).nonzero().squeeze()
            #
            #
            # not_in_A_indices1 = torch.nonzero(~torch.isin(index1, matching_indices)).squeeze()
            #
            # not_in_A_indices2 = torch.nonzero(~torch.isin(index2, matching_indices)).squeeze()
            #
            #
            # # 计算不在 tensorA 中的元素的个数
            # not_in_A_count1 = not_in_A_indices1.numel()
            #
            # not_in_A_count2 = not_in_A_indices2.numel()


            loss = both_loss +  loss_kd + loss_reli_dista + loss_reli_distr

        else:

            """
                 Student classifier
            """
            # student_query.shape=[4,200,1,230]
            student_query = self.MetaCNN_Encoder(both_query_w).view(-1, (baseN + novelN) * Q, hidden_size).unsqueeze(2)
            student_prototypes = student_prototypes.unsqueeze(1)
            reliability_student_dista = self.reliability_dista(student_query, student_prototypes, K)
            reliability_student_distr = self.reliability_distr(student_prototypes.squeeze(1), novel_support_student.view(-1, hidden_size), K, hidden_size, novelN)

            """
                classifier
            """

            # student_query.shape=[4,200,1,230]
            # merge_prototypes.shape=[4, 1, 40, 230]
            logits = self.logits(torch.sqrt(torch.pow(student_query - student_prototypes, 2).sum(-1)), reliability_student_dista, reliability_student_distr, disps_student)
            #logits = self.logits_onlyD(torch.sqrt(torch.pow(student_query - student_prototypes, 2).sum(-1)))
            #logits = self.logits_withoutDistr(torch.sqrt(torch.pow(student_query - student_prototypes, 2).sum(-1)), reliability_student_dista, disps_student)

            both_preds = torch.argmin(logits, -1)  # 取最短距离
            both_acc = self.accuracy(both_preds, query_label)

            both_loss = self.softmax_loss(-logits, query_label)



            loss = both_loss



        return  both_acc, both_preds, loss, logits


    def teacher_prototype_generator(self, novel_support, novelN, K, hidden_size):
        """
        :param novel_support: shape = [B*novelN*K, sen_len, hidden_size]
        :param novelN:
        :param K:
        :param hidden_size:
        :return:
        """
        """
            novel prototypes 
        """
        #novel_support=[200,230]
        # shape = [B*novelN*K, hidden_size]
        #novel_support = self.MetaCNN_Encoder(novel_support)
        #print(novel_support.shape)
        # shape = [B*novelN, K, hidden_size] 40,5,230
        novel_support_teacher = novel_support.view(-1, K, hidden_size)
        # shape = [B, novelN, hidden_size]
        teacher_prototypes = torch.mean(novel_support_teacher, 1).view(-1, novelN, hidden_size)

        variances = torch.var(novel_support_teacher, dim=1).sum(-1).view(-1, novelN)

        variances = torch.sigmoid(variances)

        #variances = F.softplus(variances)

        return teacher_prototypes, variances

    def logits(self, d, rel_ra, rel_rr, disp):

        logits = d - torch.log((disp.unsqueeze(1) * rel_ra) + 0.1 * (disp.unsqueeze(1) * rel_rr.unsqueeze(1)))

        return logits

    def logits_onlyD(self, d):

        logits = d

        return logits



    def logits_withoutDistr(self, d, rel_ra, disp):

        logits = d - torch.log((disp.unsqueeze(1) * rel_ra))

        return logits


    def student_prototypes_generator(self, novel_support, novelN, K, hidden_size):
        """
        :param novel_support: shape = [B*novelN*K, sen_len, hidden_size]
        :param novelN:
        :param K:
        :param hidden_size:
        :return:
        """
        """
            novel prototypes 
        """
        # shape = [B*novelN*K, hidden_size]
        novel_support_student = self.MetaCNN_Encoder(novel_support)
        # shape = [B*novelN, K, hidden_size] 40,5,230
        novel_support_student = novel_support_student.view(-1, K, hidden_size)
        # shape = [novel_supportB, novelN, hidden_size]
        novel_prototypes = torch.mean(novel_support_student, 1).view(-1, novelN, hidden_size)

        variances = torch.var(novel_support_student, dim=1).sum(-1).view(-1, novelN)

        variances = torch.sigmoid(variances)

        #variances = F.softplus(variances)

        return novel_prototypes, variances, novel_support_student

    def reliability_dista(self, teacher_query, teacher_prototypes, K): # novel_support_teacher=[200,230]
        """
        :param novel_support: shape = [B*novelN*K, sen_len, hidden_size]
        :param novelN:
        :param K:
        :param hidden_size:
        :return:
        """
        """
            novel prototypes 
        """


        # logits_teacher.shape=[4, 200, 10] teacher_query=[4,50,1,230],teacher_prototypes=[4,1,10,230]
        logits_teacher = torch.sqrt(torch.pow(teacher_query - teacher_prototypes, 2).sum(-1))

        reliability = torch.ones(logits_teacher.size()).cuda() - logits_teacher / logits_teacher.sum(-1).unsqueeze(2)

        reliability = torch.sigmoid(reliability)

        return reliability

    def reliability_distr(self, prototypes, support, K, hidden_size, novelN):  # novel_support_teacher=[200,230] teacher_prototypes = [4,1,10,230]
        """
        :param novel_support: shape = [B*novelN*K, sen_len, hidden_size]
        :param novelN:
        :param K:
        :param hidden_size:
        :return:
        """
        """
            novel prototypes 
        """
        # teacher_prototypes = [4,10,230]
        teacher_prototypes = prototypes.squeeze(1)

        # 4,10,5,230
        novel_support = support.view(-1, novelN, K, hidden_size)

        # 计算均值张量
        mean_support = teacher_prototypes  # 计算每5个样本的均值，保持维度

        # 重塑张量以便计算协方差矩阵
        reshaped_for_covariance_support = (novel_support).permute(0, 1, 3, 2)  # 将维度调整为[4, 10, 230, 5]

        # 计算协方差矩阵 4,10,230,230
        covariance_matrix_support = torch.matmul(reshaped_for_covariance_support, reshaped_for_covariance_support.transpose(-1, -2)) / K  # 计算协方差矩阵

        # 定义空的张量用于存放结果
        GMM_means = torch.zeros(mean_support.size()[0], novelN, hidden_size).cuda()
        GMM_covariances = torch.zeros(mean_support.size()[0], novelN, hidden_size, hidden_size).cuda()

        for i in range(mean_support.size()[0]):  # 针对每组样本
            samples = novel_support.view(-1, K * novelN, hidden_size)[i]  # 获取当前组的样本数据

            gmm = GaussianMixture(n_components=novelN, random_state=0)  # 指定5个簇
            gmm.fit(samples.detach().cpu())

            # 获取每个簇的均值和协方差矩阵
            GMM_means[i] = torch.from_numpy(gmm.means_).cuda()  # 转换为PyTorch Tensor 4,10,230
            GMM_covariances[i] = torch.from_numpy(gmm.covariances_).cuda() # 4,10,230,230

        num_clusters = novelN  # 每个聚类结果中的簇数量

        # 计算每对簇之间的KL散度并求和得到总的KL散度
        total_kl_divergence = 0.0

        KLs = torch.zeros(mean_support.size()[0], novelN, 1).cuda()

        for k in range(mean_support.size()[0]):

            for i in range(num_clusters):
                KL_list = []
                for j in range(num_clusters):

                    # 计算KL散度
                    kl_divergence = self.multivariate_gaussian_kl_divergence(mean_support[k][i].squeeze(0), covariance_matrix_support[k][i], GMM_means[k][j], GMM_covariances[k][j])

                    KL_list.append(kl_divergence)

                KLs[k][i] = min(KL_list)

        KLs = torch.sigmoid(-KLs.squeeze(-1)/ KLs.squeeze(-1).sum(-1).unsqueeze(-1))
        #4,10
        return KLs

    def accuracy(self, pred, label):

        acc = torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

        return acc


    def softmax_loss_BL(self, dist, y):
        N = dist.size(-1)

        logits = dist.view(-1,N)
        loss_fn = nn.CrossEntropyLoss()
        smooth_targets = torch.full(logits.size(), 0.0, device=logits.device)
        for i in range(smooth_targets.size(0)):
            smooth_targets[i,y[i]] = 1.0

        predictions = nn.functional.softmax(logits, dim=-1).view(-1,N)


        loss = loss_fn(predictions,smooth_targets) / y.size(-1)


        return loss

    def softmax_loss(self, dist, y):
        N = dist.size(-1)

        logits = dist.view(-1,N)

        # N = dist.size(-1)
        epsilon = 0.1

        loss_fn = nn.CrossEntropyLoss()
        smooth_targets = torch.full(logits.size(), epsilon / N, device=logits.device)
        for i in range(smooth_targets.size(0)):
            smooth_targets[i,y[i]] = 1.0 - epsilon + epsilon / N

        predictions = nn.functional.softmax(logits, dim=-1).view(-1,N)


        loss = loss_fn(predictions,smooth_targets) / y.size(-1)


        return loss

    def multivariate_gaussian_kl_divergence(self, mean_P, covariance_P, mean_Q, covariance_Q):
        # Compute the dimensions
        dim = mean_P.shape[0]
        covariance_P = self.regular_covar(covariance_P,dim)
        covariance_Q = self.regular_covar(covariance_Q,dim)

        if torch.det(covariance_P) == torch.det(covariance_Q):
            # Compute the KL divergence
            kl_div = 0.5 * (  # Log term
                    - dim  # Dimension term
                    + torch.trace(torch.inverse(covariance_Q) @ covariance_P)  # Trace term
                    + (mean_Q - mean_P) @ torch.inverse(covariance_Q) @ (mean_Q - mean_P)  # Mean term
            )
        else:
            kl_div = 0.5 * (
                    torch.log(torch.det(covariance_Q) / torch.det(covariance_P))  # Log term
                    - dim  # Dimension term
                    + torch.trace(torch.inverse(covariance_Q) @ covariance_P)  # Trace term
                    + (mean_Q - mean_P) @ torch.inverse(covariance_Q) @ (mean_Q - mean_P)  # Mean term
            )

        return kl_div.item()




    def regular_covar(self, covariance, hidden_size):

        # # 计算最小值和最大值
        # min_val = torch.min(covariance)
        # max_val = torch.max(covariance)
        #
        # # 使用最小-最大归一化将矩阵缩放到[0, 1]范围
        # covariance = (covariance - min_val) / (max_val - min_val)

        tensor_upper = torch.triu(covariance)  # 获取张量的上三角部分
        cov = tensor_upper.T + tensor_upper - torch.diag(covariance.diagonal())
        if torch.min(torch.diag(cov)) <= 0.0:
            cov = cov - torch.eye(hidden_size).cuda() * torch.min(torch.diag(cov)) + torch.eye(hidden_size).cuda() * 0.00001
        if torch.min(torch.diag(cov)) == 0.0:
            cov = cov + torch.eye(hidden_size).cuda() * 0.00001
        return cov
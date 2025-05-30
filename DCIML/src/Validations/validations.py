
# top1
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
import collections
import ipdb

import numpy as np
import logging
import torch.backends.cudnn as cudnn
import os
import pickle
def info(msg):
    print(msg)
    logging.info(msg)
from tqdm import tqdm
import torch

from Utils.utils import gpu
import json

from Models.loss import  cal_nll_loss

def top_1_metric(pred, label):
    result = {}
    bsz = pred.shape[0]
    iou = calculate_IoU_batch((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result



# def top_1_metric(pred, label):
#     result = {}
#     bsz = pred.shape[0]
#
#     # 计算IoU
#     iou = calculate_IoU_batch((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
#     result['mIoU'] = np.mean(iou)
#
#     # 存储IoU阈值掩码
#     iou_masks = {}
#
#     # 循环计算每个阈值（0.1, 0.3, 0.5, 0.7, 0.9）
#     for i in range(1, 10, 2):
#         threshold = i / 10
#
#         # 生成掩码，满足阈值条件的为1，否则为0
#         mask = (iou >= threshold).astype(int)
#
#         # 将掩码保存到iou_masks字典中
#         iou_masks[f'IoU@{threshold}'] = torch.tensor(mask)  # 使用torch.tensor转换为tensor
#
#         # 计算符合该阈值条件的准确率并保存在result中
#         result[f'IoU@0.{i}'] = 1.0 * np.sum(mask) / bsz
#
#
#     return result



def top_n_metric(preds, label):
    result = {}
    bsz = preds[0].shape[0]
    top_iou = []
    for pred in preds:
        iou = calculate_IoU_batch((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
        top_iou.append(iou)
    iou = np.max(np.stack(top_iou, 1), 1)
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz


    # 初始化 mask 字典
    mask_dict = {}

    # 遍历不同的 IoU 阈值
    for threshold in [0.3, 0.5, 0.7]:
        # 生成 mask：1 表示 IoU >= 阈值，0 表示 IoU < 阈值
        mask = (iou >= threshold).astype(int)
        mask_dict[f'IoU@{threshold}'] = mask


    # 验证
    # mask_dict = {}
    #
    # # 遍历不同的 IoU 阈值
    # for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
    #     # 生成 mask：1 表示 IoU >= 阈值，0 表示 IoU < 阈值
    #     mask = (iou >= threshold).astype(int)
    #     mask_dict[f'IoU@{threshold}'] = mask
    #
    #     # 计算满足条件的样本占比
    #     ratio = np.mean(mask)  # mask 中 1 的比例
    #     mask_dict[f'IoU@{threshold}_ratio'] = ratio

    return result,mask_dict



def calculate_IoU_batch(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou

def get_gt(video_metas, query_metas):
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        for i, query_id in enumerate(query_metas):
            if query_id.split('#', 1)[0] == vid_id:
                v2t_gt[-1].append(i)

    t2v_gt = {}
    for i, t_gts in enumerate(v2t_gt):
        for t_gt in t_gts:
            t2v_gt.setdefault(t_gt, [])
            t2v_gt[t_gt].append(i)

    return v2t_gt, t2v_gt


# def eval_q2m(scores, q2m_gts):
#     n_q, n_m = scores.shape
#
#     gt_ranks = torch.zeros((n_q), dtype=torch.int32).cuda()
#     aps = torch.zeros(n_q).cuda()
#     for i in range(n_q):
#         s = scores[i]
#         sorted_idxs = torch.argsort(s)
#         rank = n_m + 1
#         tmp_set = []
#         for k in q2m_gts[i]:
#             tmp = torch.where(sorted_idxs == k)[0][0] + 1
#             if tmp < rank:
#                 rank = tmp
#
#         gt_ranks[i] = rank
#
#     # compute metrics
#     r1 = 100.0 * len(torch.where(gt_ranks <= 1)[0]) / n_q
#     r5 = 100.0 * len(torch.where(gt_ranks <= 5)[0]) / n_q
#     r10 = 100.0 * len(torch.where(gt_ranks <= 10)[0]) / n_q
#     r100 = 100.0 * len(torch.where(gt_ranks <= 100)[0]) / n_q
#     # print('h') 数量15585
#     return (r1, r5, r10, r100)

def eval_q2m(scores, q2m_gts):
    n_q, n_m = scores.shape

    gt_ranks = torch.zeros((n_q), dtype=torch.int32).cuda()
    aps = torch.zeros(n_q).cuda()
    for i in range(n_q):
        s = scores[i]
        sorted_idxs = torch.argsort(s)
        rank = n_m + 1
        tmp_set = []
        for k in q2m_gts[i]:
            tmp = torch.where(sorted_idxs == k)[0][0] + 1
            if tmp < rank:
                rank = tmp

        gt_ranks[i] = rank

    # compute metrics
    r1 = 100.0 * len(torch.where(gt_ranks <= 1)[0]) / n_q
    r5 = 100.0 * len(torch.where(gt_ranks <= 5)[0]) / n_q
    r10 = 100.0 * len(torch.where(gt_ranks <= 10)[0]) / n_q
    r100 = 100.0 * len(torch.where(gt_ranks <= 100)[0]) / n_q

    # Create a tensor where 1 means the rank is within r10 (<= 10), and 0 means it is not
    r10_tensor = (gt_ranks <= 10).int()

    # Create a tensor where 1 means the rank is within r100 (<= 100), and 0 means it is not
    r100_tensor = (gt_ranks <= 100).int()

    # # Validate the tensors by calculating the ratio of 1's in r10_tensor and r100_tensor
    # r10_ratio = torch.sum(r10_tensor).item() / n_q
    # r100_ratio = torch.sum(r100_tensor).item() / n_q


    # print(gt_ranks.shape) 15585
    return (r1, r5, r10, r100, r10_tensor, r100_tensor)


def cal_perf(t2v_all_errors, t2v_gt):
    # video retrieval
    (t2v_r1, t2v_r5, t2v_r10, t2v_r100, r10_tensor, r100_tensor) = eval_q2m(t2v_all_errors, t2v_gt)

    return (t2v_r1, t2v_r5, t2v_r10, t2v_r100, r10_tensor, r100_tensor)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class validations(nn.Module):
    def __init__(self, cfg):
        super(validations, self).__init__()

        self.cfg = cfg

    def forward(self, model, context_dataloader, query_eval_loader,test_loader):

        model.eval()

        context_info = self.compute_context_info(model, context_dataloader)
        score_sum, query_metas = self.compute_query2ctx_info(model,
                                                             query_eval_loader,
                                                             context_info)


        video_metas = context_info['video_metas']

        v2t_gt, t2v_gt = get_gt(video_metas, query_metas)

        t2v_r1, t2v_r5, t2v_r10, t2v_r100, r10_retrieval, r100_retrieval = cal_perf(-1 * score_sum, t2v_gt)

        t2v_rsum = 0
        t2v_rsum += (t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100)

        ############加上grounding的模型测试########################

        metrics_logger,iou_03_tensor,iou_05_tensor,iou_07_tensor = self.compute_grounding_info(model,test_loader)
        # print('oooooo')
        # print(grounding_results)
        r10_retrieval = r10_retrieval.cuda()
        r100_retrieval = r100_retrieval.cuda()
        iou_03_tensor = iou_03_tensor.cuda()
        iou_05_tensor = iou_05_tensor.cuda()
        iou_07_tensor = iou_07_tensor.cuda()

        # 计算r10_retrieval下不同iou的正确比例
        def calculate_success_ratio(retrieval_tensor, iou_tensor):
            # 只有当检索成功(r == 1)并且定位成功(iou == 1)时，才算成功
            joint_success = (retrieval_tensor == 1) & (iou_tensor == 1)
            return joint_success.sum().item() / joint_success.size(0)

        # r10_retrieval下的成功比率
        r10_iou_03_ratio = calculate_success_ratio(r10_retrieval, iou_03_tensor)
        r10_iou_05_ratio = calculate_success_ratio(r10_retrieval, iou_05_tensor)
        r10_iou_07_ratio = calculate_success_ratio(r10_retrieval, iou_07_tensor)

        # 计算r100_retrieval下不同iou的成功比率
        r100_iou_03_ratio = calculate_success_ratio(r100_retrieval, iou_03_tensor)
        r100_iou_05_ratio = calculate_success_ratio(r100_retrieval, iou_05_tensor)
        r100_iou_07_ratio = calculate_success_ratio(r100_retrieval, iou_07_tensor)

        # # 输出结果
        # print(f"r10_retrieval下不同iou的成功比率：")
        # print(f"IoU@0.3: {r10_iou_03_ratio:.4f}")
        # print(f"IoU@0.5: {r10_iou_05_ratio:.4f}")
        # print(f"IoU@0.7: {r10_iou_07_ratio:.4f}")
        #
        # print(f"\nr100_retrieval下不同iou的成功比率：")
        # print(f"IoU@0.3: {r100_iou_03_ratio:.4f}")
        # print(f"IoU@0.5: {r100_iou_05_ratio:.4f}")
        # print(f"IoU@0.7: {r100_iou_07_ratio:.4f}")



        return [t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_rsum,r10_iou_03_ratio,r10_iou_05_ratio,r10_iou_07_ratio,r100_iou_03_ratio,r100_iou_05_ratio,r100_iou_07_ratio],metrics_logger

    def compute_query2ctx_info(self, model, query_eval_loader, ctx_info):

        query_metas = []
        score_sum = []
        for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing q embedding",
                               total=len(query_eval_loader)):
            batch = gpu(batch)
            query_metas.extend(batch[-1])
            query_feat = batch[0]
            query_mask = batch[1]
            _clip_scale_scores, _frame_scale_scores,clipss,clipss2,clip_level_query_context_scores = model.get_pred_from_raw_query(
                query_feat, query_mask, None, ctx_info["video_proposal_feat"], ctx_info["video_feat"])
            _score_sum = self.cfg['clip_scale_w'] * _clip_scale_scores + self.cfg['frame_scale_w'] * _frame_scale_scores

            score_sum.append(_score_sum)

        score_sum = torch.cat(score_sum, dim=0)

        return score_sum, query_metas

    def compute_context_info(self, model, context_dataloader):

        n_total_vid = len(context_dataloader.dataset)
        bsz = self.cfg['eval_context_bsz']
        metas = []  # list(dicts)
        vid_proposal_feat = []
        frame_feat, frame_mask = [], []
        for idx, batch in tqdm(enumerate(context_dataloader), desc="Computing query2video scores",
                               total=len(context_dataloader)):
            batch = gpu(batch)
            metas.extend(batch[-1])
            clip_video_feat_ = batch[0]
            frame_video_feat_ = batch[1]
            frame_mask_ = batch[2]
            _frame_feat, _video_proposal_feat = model.encode_context(clip_video_feat_, frame_video_feat_, frame_mask_)

            frame_feat.append(_frame_feat)
            frame_mask.append(frame_mask_)

            vid_proposal_feat.append(_video_proposal_feat)

        vid_proposal_feat = torch.cat(vid_proposal_feat, dim=0)

        def cat_tensor(tensor_list):
            if len(tensor_list) == 0:
                return None
            else:
                seq_l = [e.shape[1] for e in tensor_list]
                b_sizes = [e.shape[0] for e in tensor_list]
                b_sizes_cumsum = np.cumsum([0] + b_sizes)
                if len(tensor_list[0].shape) == 3:
                    hsz = tensor_list[0].shape[2]
                    res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l), hsz)
                elif len(tensor_list[0].shape) == 2:
                    res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l))
                else:
                    raise ValueError("Only support 2/3 dimensional tensors")
                for i, e in enumerate(tensor_list):
                    res_tensor[b_sizes_cumsum[i]:b_sizes_cumsum[i + 1], :seq_l[i]] = e
                return res_tensor

        return dict(
            video_metas=metas,  # list(dict) (N_videos)
            video_proposal_feat=vid_proposal_feat,
            video_feat=cat_tensor(frame_feat),
            video_mask=cat_tensor(frame_mask)
        )


    def compute_grounding_info(self, model, test_loader):


        model.eval()
        test_bar = tqdm(test_loader, desc="Computing_grounding" , total=len(test_loader),
                         unit="batch", dynamic_ncols=True)
        metrics_logger = collections.defaultdict(lambda: AverageMeter())
        # 初始化空字典来存储合并的结果
        merged_mask = {'IoU@0.3': [], 'IoU@0.5': [], 'IoU@0.7': []}

        for idx, batch in enumerate(test_bar):

            batch = gpu(batch)

            frames_len0 = batch['frames_len']
            frames_len0 = torch.tensor([item[0] for item in frames_len0])

            duration=batch['durations']
            # 将列表转换为一维ndarray
            duration = np.concatenate([np.array(sublist) for sublist in  duration])

            gt=batch['timestamps']
            # 将每个子列表展平并合并成一个二维数组
            gt = np.concatenate([np.array(sublist) for sublist in gt], axis=0)

            eee = 0
            output = model.grounding_task2(batch,eee, num_proposals=10, random_p=0, tau=0.70)

            bsz = len(duration)
            num_props = 10
            k = 10

            words_mask = output['words_mask'].unsqueeze(1) \
                .expand(bsz, num_props, -1).contiguous().view(bsz * num_props, -1)
            words_id = output['words_id'].unsqueeze(1) \
                .expand(bsz, num_props, -1).contiguous().view(bsz * num_props, -1)

            nll_loss, acc = cal_nll_loss(output['words_logit'], words_id, words_mask)
            idx = nll_loss.view(bsz, num_props).argsort(dim=-1)

            width = output['width'].view(bsz, num_props).gather(index=idx, dim=-1)
            center = output['center'].view(bsz, num_props).gather(index=idx, dim=-1)
            selected_props = torch.stack([torch.clamp(center - width / 2, min=0),
                                          torch.clamp(center + width / 2, max=1)], dim=-1)
            # print(selected_props.shape)32,8,23
            selected_props = selected_props.cpu().numpy()

            gt = gt / duration[:, np.newaxis]

            # if 'vote' in self.args and self.args['vote']:
            #     if self.args['dataset']['dataset'] == 'CharadesSTA':
            #         # On charades, the IoU of many proposals is small, and it doesn't make sense to get these proposals to vote.
            #         # So we weight the voting results of each proposal according to it's IoU with the first proposal.
            #         c = np.zeros((bsz, num_props))
            #         for i in range(num_props):
            #             iou = calculate_IoU_batch((selected_props[:, 0, 0], selected_props[:, 0, 1]),
            #                                       (selected_props[:, i, 0], selected_props[:, i, 1]))
            #             c[:, i] = iou
            #     else:
            #         c = np.ones((bsz, num_props))
            #     votes = np.zeros((bsz, num_props))
            #     for i in range(num_props):
            #         for j in range(num_props):
            #             iou = calculate_IoU_batch((selected_props[:, i, 0], selected_props[:, i, 1]),
            #                                       (selected_props[:, j, 0], selected_props[:, j, 1]))
            #             iou = iou * c[:, j]
            #             votes[:, i] = votes[:, i] + iou
            #     idx = np.argmax(votes, axis=1)
            #     res = top_1_metric(selected_props[np.arange(bsz), idx], gt)
            # else:
            res = top_1_metric(selected_props[:, 0], gt)

            for key, v in res.items():
                metrics_logger['R@1,' + key].update(v, bsz)
            res,mask_dict = top_n_metric(selected_props[:, :k].transpose(1, 0, 2), gt)
            for iou_key in mask_dict:
                merged_mask[iou_key].append(mask_dict[iou_key])
            for key, v in res.items():
                metrics_logger['R@%d,' % (k) + key].update(v, bsz)

        # 将列表拼接成一个大的数组
        for iou_key in merged_mask:
            merged_mask[iou_key] = np.concatenate(merged_mask[iou_key])
        # 将NumPy数组转换为PyTorch张量
        iou_03_tensor = torch.tensor(merged_mask['IoU@0.3'])
        iou_05_tensor = torch.tensor(merged_mask['IoU@0.5'])
        iou_07_tensor = torch.tensor(merged_mask['IoU@0.7'])
        # msg = '|'.join([' {} {:.4f} '.format(k, v.avg) for k, v in metrics_logger.items()])
        # info('|'+msg+'|')

        return metrics_logger,iou_03_tensor,iou_05_tensor,iou_07_tensor


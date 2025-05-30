
import collections
import os
import argparse
import numpy as np
import random
import sys
import time
from tqdm import tqdm
import ipdb
import pickle

import torch
import torch.nn as nn

from Configs.builder import get_configs
from Models.builder import get_models
from Datasets.builder import get_datasets
from Opts.builder import get_opts,get_opts2
from Losses.builder import get_losses
from Validations.builder import get_validations

from Utils.basic_utils import AverageMeter, BigFile, read_dict, log_config
from Utils.utils import set_seed, set_log, gpu, save_ckpt, load_ckpt
from Models.loss import rec_loss,ivc_loss,containment_loss,center_loss,kl_divergence_loss
from Models.loss import  cal_nll_loss,GroupedTripletLoss

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

parser = argparse.ArgumentParser(description="Partially Relevant Video Retrieval")
parser.add_argument(
    '-d', '--dataset_name', default='act', type=str, metavar='DATASET', help='dataset name',
    choices=['tvr', 'act', 'cha']
)
parser.add_argument(
    '--gpu', default='0', type=str, help='specify gpu device'
)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--resume', default='', type=str)
args = parser.parse_args()


def generate_mask(tensor):
    """
    生成掩码
    :param tensor: 形状为 (batch_size, 32) 的张量
    :return: 形状为 (batch_size, 32) 的掩码张量
    """
    # 计算每个批次的平均值
    mean_values = tensor.mean(dim=1, keepdim=True)  # (batch_size, 1)

    # 生成掩码：大于等于均值的为 1，小于均值的为 0
    mask = (tensor >= mean_values).float()  # (batch_size, 32)

    return mask

def create_mask(start_end_times, num_frames=32):
    """
    根据开始和结束时间生成掩码
    :param start_end_times: 形状为 (batch_size, 2) 的张量，表示每个样本的开始和结束时间（归一化到 [0, 1]）
    :param num_frames: 视频的总帧数，默认为 32
    :return: 形状为 (batch_size, num_frames) 的掩码张量
    """
    batch_size = start_end_times.size(0)

    # 生成帧索引 (0 到 num_frames-1)
    frame_indices = torch.arange(num_frames, dtype=torch.float32, device=start_end_times.device)  # (num_frames,)

    # 将开始和结束时间从 [0, 1] 映射到 [0, num_frames-1]
    start_times = start_end_times[:, 0] * (num_frames - 1)  # (batch_size,)
    end_times = start_end_times[:, 1] * (num_frames - 1)  # (batch_size,)

    # 扩展 frame_indices 到 (batch_size, num_frames)
    frame_indices = frame_indices.unsqueeze(0).expand(batch_size, -1)  # (batch_size, num_frames)

    # 扩展 start_times 和 end_times 到 (batch_size, num_frames)
    start_times = start_times.unsqueeze(1).expand(-1, num_frames)  # (batch_size, num_frames)
    end_times = end_times.unsqueeze(1).expand(-1, num_frames)  # (batch_size, num_frames)

    # 生成掩码：区间内的帧为 1，区间外的帧为 0
    mask = (frame_indices >= start_times) & (frame_indices <= end_times)  # (batch_size, num_frames)

    # 将布尔掩码转换为浮点数（1 或 0）
    mask = mask.float()

    return mask


def train_one_epoch(epoch, train_loader, model, criterion, cfg, optimizer, optimizer2):

    if epoch >= cfg['hard_negative_start_epoch']:
        criterion.cfg['use_hard_negative'] = True
    else:
        criterion.cfg['use_hard_negative'] = False

    loss_meter = AverageMeter()
    loss_meter_grounding= collections.defaultdict(lambda: AverageMeter())
    model.train()
    rewards = torch.from_numpy(np.asarray([0, 0.5, 1.0])).cuda()

    # 初始化 num_updates
    num_updates = 0

    train_bar = tqdm(train_loader, desc="epoch " + str(epoch), total=len(train_loader),
                     unit="batch", dynamic_ncols=True)
    # 计算随机性系数 random_p
    random_p = 0.5 * np.exp(-num_updates / 2000)
    for idx, batch in enumerate(train_bar):

        batch = gpu(batch)  # 将批次数据转移到GPU

        # ################### 训练检索模块 ###################
        duration=batch['durations']
        # 将列表转换为一维ndarray
        duration = np.concatenate([np.array(sublist) for sublist in duration])

        bsz = len(duration)

        num_proposals1 = 10
        optimizer.zero_grad()
        tau = 0.65
        eee = epoch + 1
        input_list, grounding_s = model(batch, eee, num_proposals=num_proposals1, random_p=random_p, tau=tau, single=1)


########################################定位显式促进检索#####################################
        num_props = 10
        ###################0和1的交叉墒损失#####
        words_mask = grounding_s['words_mask'].unsqueeze(1) \
            .expand(bsz, num_props, -1).contiguous().view(bsz * num_props, -1)
        words_id = grounding_s['words_id'].unsqueeze(1) \
            .expand(bsz, num_props, -1).contiguous().view(bsz * num_props, -1)

        nll_loss, acc = cal_nll_loss(grounding_s['words_logit'], words_id, words_mask)
        idx = nll_loss.view(bsz, num_props).argsort(dim=-1)

        width = grounding_s['width'].view(bsz, num_props).gather(index=idx, dim=-1)
        center = grounding_s['center'].view(bsz, num_props).gather(index=idx, dim=-1)
        selected_props = torch.stack([torch.clamp(center - width / 2, min=0),
                                      torch.clamp(center + width / 2, max=1)], dim=-1)
        selected_props = selected_props[:, 0]
        # 生成掩码
        selected_props_mask = create_mask(selected_props, num_frames=32)
        selected_props_mask = selected_props_mask.detach()
        text_labels1 = batch['text_labels']
        text_labels = torch.tensor(text_labels1,dtype=torch.long).cuda()
        clip_level_query_context_scores = input_list[-5]
        clip_level_query_context_scores2 = clip_level_query_context_scores[torch.arange(bsz), :, text_labels]  # [20, 32]
        clip_level_query_context_scores2= generate_mask(clip_level_query_context_scores2)

        import torch.nn.functional as F

        loss_binary_cross_entropy = F.binary_cross_entropy_with_logits(clip_level_query_context_scores2, selected_props_mask.float())

############################################################################################################
        # print(TripletLoss)
        # 获取前 n-1 个元素
        loss_r = 0.8*input_list[-4]+input_list[-3]+input_list[-2]
        n = len(input_list)  # 列表的总长度
        input_list1 = input_list[:n-7]
        loss1 = criterion(input_list1, batch)




        if eee > 15:
            loss1 =  loss1 +loss_binary_cross_entropy+loss_r
        else:
            loss1 = loss1
        loss1.backward()
        optimizer.step()


        # ################### 训练定位模块 ###################
        optimizer2.zero_grad()  # 重置梯度
        tau = 0.65
        num_proposals1 = 10
        eee =epoch+1
        # 传递 random_p 和其他参数给模型的定位部分
        input_list, grounding_s = model(batch,eee, num_proposals=num_proposals1, random_p=random_p, tau=tau, single=2)

        ####检索显式促进定位任务######

        clip_indices = input_list[-6]
        clip_indices_second = input_list[-7]
        # 逐元素除以 32 进行缩放
        scaled_clip_indices = clip_indices / 32
        scaled_clip_indices1=scaled_clip_indices.detach()

        # 逐元素除以 32 进行缩放
        scaled_clip_indices_second = clip_indices_second / 32
        scaled_clip_indices2=scaled_clip_indices_second.detach()



        num_props = 10

        width = grounding_s['width'].view(bsz, num_props)
        center = grounding_s['center'].view(bsz, num_props)
        selected_props = torch.stack([torch.clamp(center - width / 2, min=0),
                                      torch.clamp(center + width / 2, max=1)], dim=-1)
        # loss_containment = (containment_loss(selected_props,scaled_clip_indices1)+containment_loss(selected_props,scaled_clip_indices2))/2

        loss_containment = containment_loss(selected_props, scaled_clip_indices1)

        #############################################################################################
        # print( loss_center_loss)
        aaa = {
            "margin_1": 0.1,
            "margin_2": 0.15,
            "lambda": 0.13,
            "alpha_1": 2,
            "alpha_2": 1
        }


        nll_loss_t_c = grounding_s['nll_loss_t_c']
        mse_loss_t_c = grounding_s['mse_loss_t_c']

        # print(mse_loss_t_c)

        aa = 10
        loss, loss_dict = rec_loss(**grounding_s, num_props=aa, **aaa)
        rnk_loss, rnk_loss_dict = ivc_loss(**grounding_s, num_props=aa, **aaa)
        loss_dict.update(rnk_loss_dict)
        # loss_g = loss_containment+ nll_loss_t_c+mse_loss_t_c

        if eee > 15:
            loss = loss + rnk_loss +  loss_containment*3+0.8*nll_loss_t_c +  mse_loss_t_c
        else:
            loss = loss + rnk_loss

        #loss = loss + rnk_loss +  loss_containment
        # loss = loss + rnk_loss
        loss.backward()
        optimizer2.step()  # 更新定位模块的参数

        for k, v in loss_dict.items():
            loss_meter_grounding[k].update(v)

        # # 更新 loss_meter
        # loss_meter.update(loss1.cpu().item())
        #
        # # 更新训练进度条
        # train_bar.set_description('exp: {} epoch:{:2d} iter:{:3d} loss:{:.4f}'.format(cfg['model_name'], epoch, idx, loss1))

        # 更新 num_updates
        num_updates += 1

    return loss_meter.avg


def val_one_epoch(epoch, context_dataloader, query_eval_loader, test_loader, model, val_criterion, cfg, optimizer,optimizer2,
                  best_val, loss_meter, logger):
    val_meter,metrics_logger = val_criterion(model, context_dataloader, query_eval_loader, test_loader)

    # 保存当前epoch的模型
    save_ckpt(model, optimizer,optimizer2, cfg, os.path.join(cfg['model_root'], f'epoch_{epoch}.ckpt'), epoch, best_val)

    if val_meter[4] > best_val[4]:
        es = False
        sc = 'New Best Model !!!'
        best_val = val_meter
        # 保存最好的模型
        save_ckpt(model, optimizer,optimizer2, cfg, os.path.join(cfg['model_root'], 'best.ckpt'), epoch, best_val)
    else:
        es = True
        sc = 'A Relative Failure Epoch'

    logger.info(
        '==========================================================================================================')
    logger.info('单独检索任务')
    logger.info('Epoch: {:2d}    {}'.format(epoch, sc))
    logger.info('Average Loss: {:.4f}'.format(loss_meter))
    logger.info('R@1: {:.1f}'.format(val_meter[0]))
    logger.info('R@5: {:.1f}'.format(val_meter[1]))
    logger.info('R@10: {:.1f}'.format(val_meter[2]))
    logger.info('R@100: {:.1f}'.format(val_meter[3]))
    logger.info('Rsum: {:.1f}'.format(val_meter[4]))
    logger.info('Best: R@1: {:.1f} R@5: {:.1f} R@10: {:.1f} R@100: {:.1f} Rsum: {:.1f}'.format(best_val[0], best_val[1],
                                                                                               best_val[2], best_val[3],
                                                                                               best_val[4]))
    logger.info(
        '==========================================================================================================')
    logger.info(
        '==========================================================================================================')
    logger.info('单独定位任务')
    # 提取 metrics_logger 中的指标并写入 logger
    for metric_name, metric_value in metrics_logger.items():
        logger.info(f"{metric_name}: {metric_value.avg:.4f}")
    # 输出结果到 logger
    logger.info('检索联合定位任务')

    logger.info("r10_retrieval下不同iou的成功比率：")
    logger.info(f"IoU@0.3: {val_meter[5]:.4f}")
    logger.info(f"IoU@0.5: {val_meter[6]:.4f}")
    logger.info(f"IoU@0.7: {val_meter[7]:.4f}")

    logger.info("\nr100_retrieval下不同iou的成功比率：")
    logger.info(f"IoU@0.3: {val_meter[8]:.4f}")
    logger.info(f"IoU@0.5: {val_meter[9]:.4f}")
    logger.info(f"IoU@0.7: {val_meter[10]:.4f}")

    logger.info(
        '==========================================================================================================')

    return val_meter, best_val, es


def validation(context_dataloader, query_eval_loader, model, val_criterion, cfg, logger, resume):
    val_meter = val_criterion(model, context_dataloader, query_eval_loader)

    logger.info(
        '==========================================================================================================')
    logger.info('Testing from: {}'.format(resume))
    logger.info('R@1: {:.1f}'.format(val_meter[0]))
    logger.info('R@5: {:.1f}'.format(val_meter[1]))
    logger.info('R@10: {:.1f}'.format(val_meter[2]))
    logger.info('R@100: {:.1f}'.format(val_meter[3]))
    logger.info('Rsum: {:.1f}'.format(val_meter[4]))
    logger.info(
        '==========================================================================================================')


def main():
    cfg = get_configs(args.dataset_name)

    # set logging
    logger = set_log(cfg['model_root'], 'log.txt')
    logger.info('Partially Relevant Video Retrieval Training: {}'.format(cfg['dataset_name']))

    # set seed
    set_seed(cfg['seed'])
    logger.info('set seed: {}'.format(cfg['seed']))

    # hyper parameter
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device_ids = range(torch.cuda.device_count())
    logger.info('used gpu: {}'.format(args.gpu))

    logger.info('Hyper Parameter ......')
    logger.info(cfg)

    # dataset
    logger.info('Loading Data ......')
    cfg, train_loader, test_loader, context_dataloader, query_eval_loader, test_context_dataloader, test_query_eval_loader = get_datasets(
        cfg)

    # model
    logger.info('Loading Model ......')
    model = get_models(cfg)

    # initial
    current_epoch = -1
    es_cnt = 0
    best_val = [0., 0., 0., 0., 0.]
    if args.resume != '':
        logger.info('Resume from {}'.format(args.resume))
        _, model_state_dict, optimizer_state_dict,optimizer_state_dict2,current_epoch, best_val = load_ckpt(args.resume)
        model.load_state_dict(model_state_dict)
    model = model.cuda()
    if len(device_ids) > 1:
        model = nn.DataParallel(model)

    criterion = get_losses(cfg)
    val_criterion = get_validations(cfg)

    if args.eval:
        if args.resume == '':
            logger.info('No trained ckpt load !!!')
        else:
            with torch.no_grad():
                validation(test_context_dataloader, test_query_eval_loader, model, val_criterion, cfg, logger,
                           args.resume)
        exit(0)

    optimizer = get_opts(cfg, model, train_loader)
    optimizer2 = get_opts2(cfg, model, train_loader)
    if args.resume != '':
        optimizer.load_state_dict(optimizer_state_dict)
        optimizer2.load_state_dict(optimizer_state_dict2)

    for epoch in range(current_epoch + 1, cfg['n_epoch']):
        ############## train###################################################################
        loss_meter = train_one_epoch(epoch, train_loader, model, criterion, cfg, optimizer, optimizer2)

        ############## val
        with torch.no_grad():
            val_meter, best_val, es = val_one_epoch(epoch, context_dataloader, query_eval_loader, test_loader, model,
                                                    val_criterion, cfg, optimizer,optimizer2, best_val, loss_meter, logger)


if __name__ == '__main__':
    main()

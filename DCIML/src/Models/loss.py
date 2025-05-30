# # import torch
# #
# #
# # def cal_nll_loss2(logit, idx, mask, weights=None):
# #     logit = logit.log_softmax(dim=-1)
# #     nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
# #     if weights is None:
# #         nll_loss = nll_loss.masked_fill(mask == 0, 0)
# #         nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
# #     else:
# #         # [nb * nw, seq_len]
# #         nll_loss = (nll_loss * weights).sum(dim=-1)
# #     # nll_loss = nll_loss.mean()
# #     return nll_loss.contiguous()
# #
# #
# # def cal_nll_loss(logit, idx, mask, weights=None):
# #     eps = 0.1
# #     logit = logit.log_softmax(dim=-1).to(torch.int64)
# #     nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)  # [nb * nw, seq_len]
# #     smooth_loss = -logit.sum(dim=-1)  # [nb * nw, seq_len]
# #     nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
# #     if weights is None:
# #         nll_loss = nll_loss.masked_fill(mask == 0, 0)
# #         nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
# #     else:
# #         # [nb * nw, seq_len]
# #         nll_loss = (nll_loss * weights).sum(dim=-1)
# #     # nll_loss = nll_loss.mean()
# #     return nll_loss.contiguous()
# #
# #
# # def weakly_supervised_loss(props_align, words_logit, words_id, words_mask, rewards,
# #                            weights=None, neg_words_logit=None):
# #     num_proposals = rewards.size(0)
# #     words_logit = words_logit.log_softmax(dim=-1)
# #     nll_loss = cal_nll_loss(words_logit, words_id, words_mask)
# #     nll_loss = nll_loss.view(-1, num_proposals)
# #
# #     if neg_words_logit is not None:
# #         neg_words_logit = neg_words_logit.log_softmax(dim=-1)
# #         neg_nll_loss = cal_nll_loss(neg_words_logit, words_id, words_mask, weights)
# #         neg_nll_loss = neg_nll_loss.mean()
# #
# #     idx = torch.argsort(nll_loss, dim=-1, descending=True)
# #     _, idx = torch.sort(idx, dim=-1)
# #     rewards = rewards[idx]
# #     prop_loss = -(rewards * props_align.log_softmax(dim=-1))
# #
# #     nll_loss = nll_loss.mean()
# #     prop_loss = prop_loss.mean(dim=-1).mean(dim=-1)
# #     final_loss = nll_loss + 1.0 * prop_loss
# #     if neg_words_logit is not None:
# #         final_loss += -1e-1 * neg_nll_loss
# #     return final_loss, {
# #         'nll': nll_loss.item(),
# #         'neg_nll': neg_nll_loss.item() if neg_words_logit is not None else 0.0,
# #         'prop': prop_loss.item()
# #     }
#
# #
# #
# # import torch
# #
# #
# # def cal_nll_loss2(logit, idx, mask, weights=None):
# #     logit = logit.log_softmax(dim=-1)  # 对logit进行log_softmax
# #     idx = idx.to(torch.int64)  # 确保 idx 是 int64 类型
# #     nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)  # [nb * nw, seq_len]
# #
# #     if weights is None:
# #         nll_loss = nll_loss.masked_fill(mask == 0, 0)
# #         nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
# #     else:
# #         # [nb * nw, seq_len]
# #         nll_loss = (nll_loss * weights).sum(dim=-1)
# #
# #     return nll_loss.contiguous()
# #
# #
# # def cal_nll_loss(logit, idx, mask, weights=None):
# #     eps = 0.1
# #     logit = logit.log_softmax(dim=-1)  # 对logit进行log_softmax
# #     idx = idx.to(torch.int64)  # 确保 idx 是 int64 类型
# #     nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)  # [nb * nw, seq_len]
# #     smooth_loss = -logit.sum(dim=-1)  # [nb * nw, seq_len]
# #     nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
# #
# #     if weights is None:
# #         nll_loss = nll_loss.masked_fill(mask == 0, 0)
# #         nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
# #     else:
# #         # [nb * nw, seq_len]
# #         nll_loss = (nll_loss * weights).sum(dim=-1)
# #
# #     return nll_loss.contiguous()
# #
# #
# # def weakly_supervised_loss(props_align, words_logit, words_id, words_mask, rewards,
# #                            weights=None, neg_words_logit=None):
# #     num_proposals = rewards.size(0)
# #     words_logit = words_logit.log_softmax(dim=-1)  # 对 words_logit 进行 log_softmax
# #     nll_loss = cal_nll_loss(words_logit, words_id, words_mask, weights)  # 调用修改后的 cal_nll_loss
# #     nll_loss = nll_loss.view(-1, num_proposals)
# #
# #     if neg_words_logit is not None:
# #         neg_words_logit = neg_words_logit.log_softmax(dim=-1)  # 对 neg_words_logit 进行 log_softmax
# #         neg_nll_loss = cal_nll_loss(neg_words_logit, words_id, words_mask, weights)  # 调用修改后的 cal_nll_loss
# #         neg_nll_loss = neg_nll_loss.mean()
# #
# #     # 排序并计算 prop_loss
# #     idx = torch.argsort(nll_loss, dim=-1, descending=True)
# #     _, idx = torch.sort(idx, dim=-1)
# #     rewards = rewards[idx]  # 根据排序后的 idx 重排 rewards
# #     prop_loss = -(rewards * props_align.log_softmax(dim=-1))  # 计算 prop_loss
# #
# #     # 计算最终的损失
# #     nll_loss = nll_loss.mean()
# #     prop_loss = prop_loss.mean(dim=-1).mean(dim=-1)
# #     final_loss = nll_loss + 1.0 * prop_loss  # 最终损失
# #
# #     if neg_words_logit is not None:
# #         final_loss += -1e-1 * neg_nll_loss  # 如果有 neg_words_logit，则加上负样本损失
# #
# #     return final_loss, {
# #         'nll': nll_loss.item(),
# #         'neg_nll': neg_nll_loss.item() if neg_words_logit is not None else 0.0,
# #         'prop': prop_loss.item()
# #     }
#
# ############v1##########
#
# #
# # import torch
# #
# # def cal_nll_loss(logit, idx, mask, weights=None):
# #     """
# #     计算负对数似然损失 (Negative Log-Likelihood Loss)
# #
# #     Args:
# #         logit (Tensor): 模型预测的logits，形状为 [batch_size, seq_len, num_classes]
# #         idx (Tensor): 真实标签的索引，形状为 [batch_size, seq_len]
# #         mask (Tensor): 形状为 [batch_size, seq_len] 的掩码，标记有效的标签位置
# #         weights (Tensor, optional): 权重，形状为 [batch_size, seq_len]，可选参数，默认值为 None
# #
# #     Returns:
# #         Tensor: 计算出的损失值
# #     """
# #     eps = 0.1
# #     logit = logit.log_softmax(dim=-1)  # 对 logits 进行 log_softmax 操作
# #
# #     idx = idx.to(torch.int64)  # 确保 idx 是 int64 类型
# #
# #     # 强制将 idx 的长度限制为 logit 的序列长度 (29)，避免索引超出范围
# #     idx = idx[:, :logit.size(1)]  # 修剪 idx，使其长度与 logit 的序列长度一致
# #
# #     # gather 操作：根据 idx 索引出 logit 中的值，计算损失
# #     nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len]
# #
# #     # 平滑损失，针对每个类别计算其对数概率的和
# #     smooth_loss = -logit.sum(dim=-1)  # [batch_size, seq_len]
# #
# #     # 计算最终损失（包括平滑损失的平衡项）
# #     nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
# #
# #     if weights is None:
# #         # 如果没有提供 weights，则根据 mask 进行填充处理
# #         nll_loss = nll_loss.masked_fill(mask == 0, 0)  # 将无效位置的损失填充为0
# #         nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)  # 对每个样本计算平均损失
# #     else:
# #         # 如果有提供 weights，则按权重加权计算损失
# #         nll_loss = (nll_loss * weights).sum(dim=-1)
# #
# #     return nll_loss.contiguous()
# #
# #
# # def weakly_supervised_loss(props_align, words_logit, words_id, words_mask, rewards,
# #                            weights=None, neg_words_logit=None):
# #     """
# #     计算弱监督损失，包括 NLL 损失和 proposal 损失
# #
# #     Args:
# #         props_align (Tensor): 提案对齐结果，形状为 [batch_size, num_proposals, num_classes]
# #         words_logit (Tensor): 模型对词的预测 logits，形状为 [batch_size, seq_len, num_classes]
# #         words_id (Tensor): 真实的词 ID，形状为 [batch_size, seq_len]
# #         words_mask (Tensor): 词的有效性掩码，形状为 [batch_size, seq_len]
# #         rewards (Tensor): 奖励信号，形状为 [batch_size, num_proposals]
# #         weights (Tensor, optional): 权重，形状为 [batch_size, seq_len]，可选，默认为 None
# #         neg_words_logit (Tensor, optional): 负样本词的 logits，形状为 [batch_size, seq_len, num_classes]，可选
# #
# #     Returns:
# #         tuple: 计算出的最终损失和一个字典，包含各个损失项
# #     """
# #     num_proposals = rewards.size(0)  # 提案的数量
# #     words_logit = words_logit.log_softmax(dim=-1)  # 对 words_logit 进行 log_softmax
# #     nll_loss = cal_nll_loss(words_logit, words_id, words_mask, weights)  # 计算 NLL 损失
# #     nll_loss = nll_loss.view(-1, num_proposals)  # 将损失调整为合适的形状
# #
# #     if neg_words_logit is not None:
# #         # 如果有负样本，计算负样本的 NLL 损失
# #         neg_words_logit = neg_words_logit.log_softmax(dim=-1)  # 对 neg_words_logit 进行 log_softmax
# #         neg_nll_loss = cal_nll_loss(neg_words_logit, words_id, words_mask, weights)  # 计算负样本的 NLL 损失
# #         neg_nll_loss = neg_nll_loss.mean()  # 平均负样本损失
# #
# #     # 排序并计算提案损失
# #     idx = torch.argsort(nll_loss, dim=-1, descending=True)  # 对 NLL 损失进行排序
# #     _, idx = torch.sort(idx, dim=-1)  # 排序后的索引
# #     rewards = rewards[idx]  # 根据排序后的 idx 重排 rewards
# #     prop_loss = -(rewards * props_align.log_softmax(dim=-1))  # 计算提案损失
# #
# #     # 计算最终损失
# #     nll_loss = nll_loss.mean()  # NLL 损失的均值
# #     prop_loss = prop_loss.mean(dim=-1).mean(dim=-1)  # 提案损失的均值
# #     final_loss = nll_loss + 1.0 * prop_loss  # 最终损失
# #
# #     if neg_words_logit is not None:
# #         final_loss += -1e-1 * neg_nll_loss  # 如果有负样本，加入负样本损失
# #
# #     return final_loss, {
# #         'nll': nll_loss.item(),
# #         'neg_nll': neg_nll_loss.item() if neg_words_logit is not None else 0.0,
# #         'prop': prop_loss.item()
# #     }
#
#
#
# ##################v2###########
#
#
# import torch
#
# # def cal_nll_loss(logit, idx, mask, weights=None):
# #     """
# #     计算负对数似然损失 (Negative Log-Likelihood Loss)
# #
# #     Args:
# #         logit (Tensor): 模型预测的logits，形状为 [batch_size, seq_len, num_classes]
# #         idx (Tensor): 真实标签的索引，形状为 [batch_size, seq_len]
# #         mask (Tensor): 形状为 [batch_size, seq_len] 的掩码，标记有效的标签位置
# #         weights (Tensor, optional): 权重，形状为 [batch_size, seq_len]，可选参数，默认值为 None
# #
# #     Returns:
# #         Tensor: 计算出的损失值
# #     """
# #     eps = 0.1
# #     logit = logit.log_softmax(dim=-1)  # 对 logits 进行 log_softmax 操作
# #
# #     idx = idx.to(torch.int64)  # 确保 idx 是 int64 类型
# #
# #     # 动态修剪 idx，使其序列长度与 logit 的长度一致
# #     max_seq_len = logit.size(1)  # 获取 logit 的序列长度
# #     idx = idx[:, :max_seq_len]  # 强制将 idx 的长度调整为 logit 的序列长度
# #
# #     # gather 操作：根据 idx 索引出 logit 中的值，计算损失
# #     nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len]
# #
# #     # 平滑损失，针对每个类别计算其对数概率的和
# #     smooth_loss = -logit.sum(dim=-1)  # [batch_size, seq_len]
# #
# #     # 计算最终损失（包括平滑损失的平衡项）
# #     nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
# #
# #     if weights is None:
# #         # 如果没有提供 weights，则根据 mask 进行填充处理
# #         nll_loss = nll_loss.masked_fill(mask == 0, 0)  # 将无效位置的损失填充为0
# #         nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)  # 对每个样本计算平均损失
# #     else:
# #         # 如果有提供 weights，则按权重加权计算损失
# #         nll_loss = (nll_loss * weights).sum(dim=-1)
# #
# #     return nll_loss.contiguous()
# #
# #
# # def weakly_supervised_loss(props_align, words_logit, words_id, words_mask, rewards,
# #                            weights=None, neg_words_logit=None):
# #     """
# #     计算弱监督损失，包括 NLL 损失和 proposal 损失
# #
# #     Args:
# #         props_align (Tensor): 提案对齐结果，形状为 [batch_size, num_proposals, num_classes]
# #         words_logit (Tensor): 模型对词的预测 logits，形状为 [batch_size, seq_len, num_classes]
# #         words_id (Tensor): 真实的词 ID，形状为 [batch_size, seq_len]
# #         words_mask (Tensor): 词的有效性掩码，形状为 [batch_size, seq_len]
# #         rewards (Tensor): 奖励信号，形状为 [batch_size, num_proposals]
# #         weights (Tensor, optional): 权重，形状为 [batch_size, seq_len]，可选，默认为 None
# #         neg_words_logit (Tensor, optional): 负样本词的 logits，形状为 [batch_size, seq_len, num_classes]，可选
# #
# #     Returns:
# #         tuple: 计算出的最终损失和一个字典，包含各个损失项
# #     """
# #     num_proposals = rewards.size(0)  # 提案的数量
# #     words_logit = words_logit.log_softmax(dim=-1)  # 对 words_logit 进行 log_softmax
# #     nll_loss = cal_nll_loss(words_logit, words_id, words_mask, weights)  # 计算 NLL 损失
# #     nll_loss = nll_loss.view(-1, num_proposals)  # 将损失调整为合适的形状
# #
# #     if neg_words_logit is not None:
# #         # 如果有负样本，计算负样本的 NLL 损失
# #         neg_words_logit = neg_words_logit.log_softmax(dim=-1)  # 对 neg_words_logit 进行 log_softmax
# #         neg_nll_loss = cal_nll_loss(neg_words_logit, words_id, words_mask, weights)  # 计算负样本的 NLL 损失
# #         neg_nll_loss = neg_nll_loss.mean()  # 平均负样本损失
# #
# #     # 排序并计算提案损失
# #     idx = torch.argsort(nll_loss, dim=-1, descending=True)  # 对 NLL 损失进行排序
# #     _, idx = torch.sort(idx, dim=-1)  # 排序后的索引
# #     rewards = rewards[idx]  # 根据排序后的 idx 重排 rewards
# #     prop_loss = -(rewards * props_align.log_softmax(dim=-1))  # 计算提案损失
# #
# #     # 计算最终损失
# #     nll_loss = nll_loss.mean()  # NLL 损失的均值
# #     prop_loss = prop_loss.mean(dim=-1).mean(dim=-1)  # 提案损失的均值
# #     final_loss = nll_loss + 1.0 * prop_loss  # 最终损失
# #
# #     if neg_words_logit is not None:
# #         final_loss += -1e-1 * neg_nll_loss  # 如果有负样本，加入负样本损失
# #
# #     return final_loss, {
# #         'nll': nll_loss.item(),
# #         'neg_nll': neg_nll_loss.item() if neg_words_logit is not None else 0.0,
# #         'prop': prop_loss.item()
# #     }
#
# import torch
# import torch.nn.functional as F
# import pdb
#
#
# def cal_nll_loss(logit, idx, mask, weights=None):
#     eps = 0.1
#     acc = (logit.max(dim=-1)[1] == idx).float()
#     mean_acc = (acc * mask).sum() / mask.sum()
#
#     logit = logit.log_softmax(dim=-1)
#     nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
#     smooth_loss = -logit.sum(dim=-1)
#     nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
#     if weights is None:
#         nll_loss = nll_loss.masked_fill(mask == 0, 0)
#         nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
#     else:
#         nll_loss = (nll_loss * weights).sum(dim=-1)
#
#     return nll_loss.contiguous(), mean_acc
#
#
# def rec_loss(words_logit, words_id, words_mask, num_props, ref_words_logit=None):
#     # "margin_1": 0.1,
#     # "margin_2": 0.15,
#     # "lambda": 0.13,
#     # "alpha_1": 2,
#     # "alpha_2": 0.1
#     bsz = words_logit.size(0) // num_props
#     words_mask1 = words_mask.unsqueeze(1) \
#         .expand(bsz, num_props, -1).contiguous().view(bsz * num_props, -1)
#     words_id1 = words_id.unsqueeze(1) \
#         .expand(bsz, num_props, -1).contiguous().view(bsz * num_props, -1)
#
#     nll_loss, acc = cal_nll_loss(words_logit, words_id1, words_mask1)
#     nll_loss = nll_loss.view(bsz, num_props)
#     min_nll_loss = nll_loss.min(dim=-1)[0]
#
#     final_loss = min_nll_loss.mean()
#
#     if ref_words_logit is not None:
#         ref_nll_loss, ref_acc = cal_nll_loss(ref_words_logit, words_id, words_mask)
#         final_loss = final_loss + ref_nll_loss.mean()
#         final_loss = final_loss / 2
#
#     loss_dict = {
#         'final_loss': final_loss.item(),
#         'nll_loss': min_nll_loss.mean().item(),
#     }
#     if ref_words_logit is not None:
#         loss_dict.update({
#             'ref_nll_loss': ref_nll_loss.mean().item(),
#         })
#
#     return final_loss, loss_dict
#
#
# def ivc_loss(words_logit, words_id, words_mask, num_props, neg_words_logit_1=None, neg_words_logit_2=None,
#              ref_words_logit=None, **kwargs):
#     bsz = words_logit.size(0) // num_props
#     words_mask1 = words_mask.unsqueeze(1) \
#         .expand(bsz, num_props, -1).contiguous().view(bsz * num_props, -1)
#     words_id1 = words_id.unsqueeze(1) \
#         .expand(bsz, num_props, -1).contiguous().view(bsz * num_props, -1)
#
#     nll_loss, acc = cal_nll_loss(words_logit, words_id1, words_mask1)
#     min_nll_loss, idx = nll_loss.view(bsz, num_props).min(dim=-1)
#
#     if ref_words_logit is not None:
#         ref_nll_loss, ref_acc = cal_nll_loss(ref_words_logit, words_id, words_mask)
#         tmp_0 = torch.zeros_like(min_nll_loss).cuda()
#         tmp_0.requires_grad = False
#         ref_loss = torch.max(min_nll_loss - ref_nll_loss + kwargs["margin_1"], tmp_0)
#         rank_loss = ref_loss.mean()
#     else:
#         rank_loss = min_nll_loss.mean()
#
#     if neg_words_logit_1 is not None:
#         neg_nll_loss_1, neg_acc_1 = cal_nll_loss(neg_words_logit_1, words_id1, words_mask1)
#         neg_nll_loss_1 = torch.gather(neg_nll_loss_1.view(bsz, num_props), index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
#         tmp_0 = torch.zeros_like(min_nll_loss).cuda()
#         tmp_0.requires_grad = False
#         neg_loss_1 = torch.max(min_nll_loss - neg_nll_loss_1 + kwargs["margin_2"], tmp_0)
#         rank_loss = rank_loss + neg_loss_1.mean()
#
#     if neg_words_logit_2 is not None:
#         neg_nll_loss_2, neg_acc_2 = cal_nll_loss(neg_words_logit_2, words_id1, words_mask1)
#         neg_nll_loss_2 = torch.gather(neg_nll_loss_2.view(bsz, num_props), index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
#         tmp_0 = torch.zeros_like(min_nll_loss).cuda()
#         tmp_0.requires_grad = False
#         neg_loss_2 = torch.max(min_nll_loss - neg_nll_loss_2 + kwargs["margin_2"], tmp_0)
#         rank_loss = rank_loss + neg_loss_2.mean()
#
#     loss = kwargs['alpha_1'] * rank_loss
#
#     gauss_weight = kwargs['gauss_weight'].view(bsz, num_props, -1)
#     gauss_weight = gauss_weight / gauss_weight.sum(dim=-1, keepdim=True)
#     target = torch.eye(num_props).unsqueeze(0).cuda() * kwargs["lambda"]
#     source = torch.matmul(gauss_weight, gauss_weight.transpose(1, 2))
#     div_loss = torch.norm(target - source, dim=(1, 2)) ** 2
#
#     loss = loss + kwargs['alpha_2'] * div_loss.mean()
#
#     return loss, {
#         'ivc_loss': loss.item(),
#         'neg_loss_1': neg_loss_1.mean().item() if neg_words_logit_1 is not None else 0.0,
#         'neg_loss_2': neg_loss_2.mean().item() if neg_words_logit_2 is not None else 0.0,
#         'ref_loss': ref_loss.mean().item() if ref_words_logit is not None else 0.0,
#         'div_loss': div_loss.mean().item()
#     }


import torch
import torch.nn.functional as F
import pdb


def cal_nll_loss(logit, idx, mask, weights=None):
    # 检查 words_logit 的第二维大小是否为 30，不是就填充
    if logit.shape[1] != 30:
        padding_size = 30 - logit.shape[1]
        logit = F.pad(logit, (0, 0, 0, padding_size), value=0)  # 填充到 30

    # 检查 words_id1 的第二维大小是否为 30，不是就填充
    if idx.shape[1] != 30:
        padding_size = 30 - idx.shape[1]
        idx = F.pad(idx, (0, padding_size), value=0)  # 填充到 30

    # 检查 words_mask1 的第二维大小是否为 30，不是就填充
    if mask.shape[1] != 30:
        padding_size = 30 - mask.shape[1]
        mask = F.pad(mask, (0, padding_size), value=0)  # 填充到 30

    eps = 0.1
    # print(logit.shape)
    # print(idx.shape)torch.Size([720, 30])
    # print(mask.shape)torch.Size([720, 29])
    acc = (logit.max(dim=-1)[1] == idx).float()
    mean_acc = (acc * mask).sum() / mask.sum()

    logit = logit.log_softmax(dim=-1)
    # nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    idx = idx.to(torch.long)  # 将 idx 转换为 int64 类型
    nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)

    smooth_loss = -logit.sum(dim=-1)
    nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
    if weights is None:
        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
    else:
        nll_loss = (nll_loss * weights).sum(dim=-1)

    return nll_loss.contiguous(), mean_acc


def rec_loss(words_logit, words_id, words_mask, num_props, ref_words_logit=None, **kwargs):

    bsz = words_logit.size(0) // num_props
    words_mask1 = words_mask.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz * num_props, -1)
    words_id1 = words_id.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz * num_props, -1)
    # print('a:',words_logit.shape)
    # print('b:', words_id1.shape)
    # print('c:', words_mask1.shape)
    nll_loss, acc = cal_nll_loss(words_logit, words_id1, words_mask1)
    nll_loss = nll_loss.view(bsz, num_props)
    min_nll_loss = nll_loss.min(dim=-1)[0]

    final_loss = min_nll_loss.mean()

    if ref_words_logit is not None:
        # ref_words_logit = ref_words_logit[:, :30, :]
        # print(ref_words_logit.shape)
        ref_nll_loss, ref_acc = cal_nll_loss(ref_words_logit, words_id, words_mask)
        final_loss = final_loss + ref_nll_loss.mean()
        final_loss = final_loss / 2

    loss_dict = {
        'final_loss': final_loss.item(),
        'nll_loss': min_nll_loss.mean().item(),
    }
    if ref_words_logit is not None:
        loss_dict.update({
            'ref_nll_loss': ref_nll_loss.mean().item(),
        })

    return final_loss, loss_dict


def ivc_loss(words_logit, words_id, words_mask, num_props, neg_words_logit_1=None, neg_words_logit_2=None,
             ref_words_logit=None, **kwargs):
    bsz = words_logit.size(0) // num_props
    words_mask1 = words_mask.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz * num_props, -1)
    words_id1 = words_id.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz * num_props, -1)

    nll_loss, acc = cal_nll_loss(words_logit, words_id1, words_mask1)
    min_nll_loss, idx = nll_loss.view(bsz, num_props).min(dim=-1)

    if ref_words_logit is not None:
        ref_words_logit = ref_words_logit[:, :30, :]

        ref_nll_loss, ref_acc = cal_nll_loss(ref_words_logit, words_id, words_mask)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        ref_loss = torch.max(min_nll_loss - ref_nll_loss + kwargs["margin_1"], tmp_0)
        rank_loss = ref_loss.mean()
    else:
        rank_loss = min_nll_loss.mean()

    if neg_words_logit_1 is not None:
        neg_nll_loss_1, neg_acc_1 = cal_nll_loss(neg_words_logit_1, words_id1, words_mask1)
        neg_nll_loss_1 = torch.gather(neg_nll_loss_1.view(bsz, num_props), index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        neg_loss_1 = torch.max(min_nll_loss - neg_nll_loss_1 + kwargs["margin_2"], tmp_0)
        rank_loss = rank_loss + neg_loss_1.mean()

    if neg_words_logit_2 is not None:
        neg_nll_loss_2, neg_acc_2 = cal_nll_loss(neg_words_logit_2, words_id1, words_mask1)
        neg_nll_loss_2 = torch.gather(neg_nll_loss_2.view(bsz, num_props), index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        neg_loss_2 = torch.max(min_nll_loss - neg_nll_loss_2 + kwargs["margin_2"], tmp_0)
        rank_loss = rank_loss + neg_loss_2.mean()

    loss = kwargs['alpha_1'] * rank_loss

    gauss_weight = kwargs['gauss_weight'].view(bsz, num_props, -1)
    gauss_weight = gauss_weight / gauss_weight.sum(dim=-1, keepdim=True)
    target = torch.eye(num_props).unsqueeze(0).cuda() * kwargs["lambda"]
    source = torch.matmul(gauss_weight, gauss_weight.transpose(1, 2))
    div_loss = torch.norm(target - source, dim=(1, 2)) ** 2

    loss = loss + kwargs['alpha_2'] * div_loss.mean()

    return loss, {
        'ivc_loss': loss.item(),
        'neg_loss_1': neg_loss_1.mean().item() if neg_words_logit_1 is not None else 0.0,
        'neg_loss_2': neg_loss_2.mean().item() if neg_words_logit_2 is not None else 0.0,
        'ref_loss': ref_loss.mean().item() if ref_words_logit is not None else 0.0,
        'div_loss': div_loss.mean().item()
    }


import torch


def containment_loss(selected_props, scaled_clip_indices):
    """
    计算 Containment Loss
    :param selected_props: 候选时间段，形状为 (batch_size, num_proposals, 2)，其中最后一个维度是 [start, end]
    :param scaled_clip_indices: 标注帧的时间位置，形状为 (batch_size,)
    :return: 标量损失值
    """
    batch_size, num_proposals, _ = selected_props.shape

    # 扩展 scaled_clip_indices 到 (batch_size, num_proposals)
    labels = scaled_clip_indices.unsqueeze(1).expand(-1, num_proposals)  # (8, 10)

    # 提取 start 和 end
    starts = selected_props[:, :, 0]  # (8, 10)
    ends = selected_props[:, :, 1]  # (8, 10)

    # 计算 Containment Loss
    loss_start = torch.max(starts - labels, torch.tensor(0.0))  # max(start - label, 0)
    loss_end = torch.max(labels - ends, torch.tensor(0.0))  # max(label - end, 0)
    total_loss = loss_start + loss_end  # (8, 10)

    # 对所有样本和候选时间段求平均
    return total_loss.mean()


import torch


def center_loss(selected_props, scaled_clip_indices):
    """
    计算 Center Loss
    :param selected_props: 候选时间段，形状为 (batch_size, num_proposals, 2)，其中最后一个维度是 [start, end]
    :param scaled_clip_indices: 标注帧的时间位置，形状为 (batch_size,)
    :return: 标量损失值
    """
    batch_size, num_proposals, _ = selected_props.shape

    # 扩展 scaled_clip_indices 到 (batch_size, num_proposals)
    labels = scaled_clip_indices.unsqueeze(1).expand(-1, num_proposals)  # (8, 10)

    # 计算每个候选时间段的中心点
    starts = selected_props[:, :, 0]  # (8, 10)
    ends = selected_props[:, :, 1]  # (8, 10)
    centers = (starts + ends) / 2  # (8, 10)

    # 计算 Center Loss（平方误差）
    center_loss = (labels - centers) ** 2  # (8, 10)

    # 对每个样本的所有候选时间段在 dim=1 上求平均
    loss_per_sample = center_loss.mean(dim=1)  # (8,)

    # 对所有样本的损失求平均
    final_loss = loss_per_sample.mean()  # 标量
    return final_loss


import torch
import torch.nn.functional as F

def kl_divergence_loss(retrieval_dist, localization_dist):
    """
    retrieval_dist: 检索模型输出的分布 [batch_size, 32]
    localization_dist: 定位模型输出的分布 [batch_size, 32]
    """
    # 分离定位模型的梯度，防止影响定位模型的训练
    localization_dist = localization_dist.detach()

    # 确保检索模型的输出是概率分布（softmax归一化）
    retrieval_dist = F.softmax(retrieval_dist, dim=-1)

    # # 确保定位模型的输出是概率分布（softmax归一化）
    # localization_dist = F.softmax(localization_dist, dim=-1)

    # 计算KL散度 (KL(localization_dist || retrieval_dist))
    loss = F.kl_div(retrieval_dist.log(), localization_dist, reduction='batchmean')

    return loss*0.05

import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
        初始化组内三元组损失函数
        :param margin: 正负样本对之间的最小间隔
        """
        super().__init__()
        self.margin = margin

    def forward(self, A, B, labels):
        """
        计算组内三元组损失
        :param A: 第一个网络的输出，形状为 (batch_size, 32)
        :param B: 第二个网络的输出，形状为 (batch_size, 32)
        :param labels: 样本的组标签，形状为 (batch_size,)
        :return: 三元组损失值
        """
        batch_size = A.size(0)
        # 对特征进行L2归一化（推荐）
        A = F.normalize(A, p=2, dim=1)
        B = F.normalize(B, p=2, dim=1)
        B =  B.detach()
        # 计算正样本对的距离（相同位置的样本）
        dist_pos = torch.norm(A - B, p=2, dim=1)  # 形状: (batch_size,)

        # 初始化负样本距离
        dist_neg = torch.zeros_like(dist_pos)  # 形状: (batch_size,)

        # 遍历每个组，计算组内负样本距离
        for group in torch.unique(labels):
            # 获取当前组的样本索引
            group_mask = (labels == group)
            group_indices = torch.where(group_mask)[0]

            # 如果组内样本数小于2，跳过（无法采样负样本）
            if len(group_indices) < 2:
                continue

            # 计算当前组内样本的距离矩阵
            group_A = A[group_mask]
            group_B = B[group_mask]
            group_dist = torch.cdist(group_A, group_B, p=2)  # 形状: (group_size, group_size)

            # 将对角线元素（正样本对）设置为一个很大的值，避免被选为负样本
            group_dist.fill_diagonal_(float('inf'))

            # 找到每个样本的最小负样本距离
            group_dist_neg_min, _ = torch.min(group_dist, dim=1)  # 形状: (group_size,)

            # 将当前组的负样本距离赋值到全局变量中
            dist_neg[group_mask] = group_dist_neg_min

        # 计算三元组损失
        losses = F.relu(dist_pos - dist_neg + self.margin)
        return losses.mean()
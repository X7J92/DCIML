import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from Models.gmmformer.model_components import BertAttention, LinearLayer, \
    TrainablePositionalEncoding, GMMBlock
from Models.transformer import DualTransformer
import ipdb
from Models.loss import cal_nll_loss


class GMMFormer_Net(nn.Module):
    def __init__(self, config):
        super(GMMFormer_Net, self).__init__()
        self.config = config
        self.dropout = 0.1
        self.query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                           hidden_size=config.hidden_size, dropout=config.input_drop)
        self.clip_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                          hidden_size=config.hidden_size, dropout=config.input_drop)
        self.frame_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                           hidden_size=config.hidden_size, dropout=config.input_drop)

        self.query_input_proj = LinearLayer(config.query_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.query_encoder = BertAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))

        self.clip_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                           dropout=config.input_drop, relu=True)
        self.clip_encoder = GMMBlock(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                           hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                           attention_probs_dropout_prob=config.drop))
        self.clip_encoder_2 = GMMBlock(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                             hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                             attention_probs_dropout_prob=config.drop))

        self.frame_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.frame_encoder_1 = GMMBlock(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                              hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                              attention_probs_dropout_prob=config.drop))
        self.frame_encoder_2 = GMMBlock(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                              hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                              attention_probs_dropout_prob=config.drop))

        self.modular_vector_mapping = nn.Linear(config.hidden_size, out_features=1, bias=False)
        self.modular_vector_mapping_2 = nn.Linear(config.hidden_size, out_features=1, bias=False)
        self.trans = DualTransformer(256, 4, 3, 3)
        self.word_pos_encoder = SinusoidalPositionalEmbedding(256, 0, 30)
        self.reset_parameters()
        self.frame_fc = nn.Linear(1024, 256)
        self.word_fc = nn.Linear(1024, 256)
        self.mask_vec = nn.Parameter(torch.zeros(256).float(), requires_grad=True)
        self.fc_comp = nn.Linear(256, 9000)
        self.pred_vec = nn.Parameter(torch.zeros(1024).float(), requires_grad=True)
        self.start_vec = nn.Parameter(torch.zeros(1024).float(), requires_grad=True)
        self.fc_gauss = nn.Linear(256, 10 * 2)
        self.momo = nn.Linear(256, 384)
        self.gamma_mlp_a = nn.Sequential(
            nn.Linear(384, 512),
            nn.ReLU(),
            nn.Linear(512, 384)
        )
        # 定义生成β的MLP
        self.beta_mlp_a = nn.Sequential(
            nn.Linear(384, 512),
            nn.ReLU(),
            nn.Linear(512, 384)
        )
        self.gamma_mlp_b = nn.Sequential(
            nn.Linear(384, 512),
            nn.ReLU(),
            nn.Linear(512, 384)
        )
        # 定义生成β的MLP
        self.beta_mlp_b = nn.Sequential(
            nn.Linear(384, 512),
            nn.ReLU(),
            nn.Linear(512, 384)
        )

        self.gamma_mlp_c = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        # 定义生成β的MLP
        self.beta_mlp_c = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.rtg = nn.Linear(384, 256)
        # self.use_negative = config.use_negative

    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size

    def _mask_words(self, words_feat, words_len, weights=None):
        token = self.mask_vec.cuda().unsqueeze(0).unsqueeze(0)
        # token = self.word_fc(token)
        # print(words_len)
        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            num_masked_words = max(l // 3, 1)
            masked_words.append(torch.zeros([words_feat.size(1)]).byte().cuda())

            if l < 1:
                continue
            p = weights[i, :l].cpu().numpy() if weights is not None else None
            # print(p)
            p = p / np.sum(p)  # 归一化 p 数组

            # print(p)  # 输出 p 数组，检查其元素和
            # print(np.sum(p))  # 输出 p 数组的元素和，应该为1

            choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False, p=p)
            masked_words[-1][choices] = 1

        # exit(0)
        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec

        return words_feat1

    def retrieval_task(self, batch, selected_enc_out=None, single=None):

        #############################以上是加载特征在（不需要冻结）############################################
        clip_video_feat = batch['clip_video_features']  # 128，30，1024
        query_feat = batch['text_feat']  # 478，30，1024
        query_mask = batch['text_mask']  # 478，30
        query_labels = batch['text_labels']  # 478
        frame_video_feat = batch['frame_video_features']  # 128，128，1024
        # print(frame_video_feat.shape)
        frame_video_mask = batch['videos_mask']  # 128，128
        bsz = frame_video_feat.size(0)
        # text_labels = batch['text_labels']
        #######################################检索任务##########################################
        encoded_frame_feat, vid_proposal_feat = self.encode_context(
            clip_video_feat, frame_video_feat, frame_video_mask)

        clip_scale_scores, clip_scale_scores_, frame_scale_scores, frame_scale_scores_, selected_indices,selected_indices_second, clip_level_query_context_scores \
            = self.get_pred_from_raw_query(
            query_feat, query_mask, query_labels, vid_proposal_feat, encoded_frame_feat, return_query_feats=True)

        label_dict = {}
        for index, label in enumerate(query_labels):
            if label in label_dict:
                label_dict[label].append(index)
            else:
                label_dict[label] = []
                label_dict[label].append(index)

        lengths = [len(lst) for lst in label_dict.values()]
        lengths_tensor = torch.tensor(lengths)

        cumsum_lengths = torch.cumsum(lengths_tensor, dim=0)
        full_indices = torch.arange(cumsum_lengths[-1], dtype=torch.long)
        segment_indices = torch.searchsorted(cumsum_lengths, full_indices, right=True) - 1
        repeated_encoded_frame_feat = encoded_frame_feat[segment_indices]
        repeated_vid_proposal_feat = vid_proposal_feat[segment_indices]

        ################取出最大帧对应的视频特征#########################

        gathered_features = repeated_vid_proposal_feat[
            torch.arange(repeated_vid_proposal_feat.size(0)), selected_indices]

        video_query = self.encode_query(query_feat, query_mask)

        ######################################将定位模型的结果特征隐式的调制检索特征，实现促进作用############################

        if self.training and single == 1:

            ff_momo = self.momo(selected_enc_out)
            ff_momo_mean = torch.mean(ff_momo, dim=1)

            cumsum_lengths = torch.cumsum(lengths_tensor, dim=0)
            full_indices = torch.arange(cumsum_lengths[-1], dtype=torch.long)
            segment_indices = torch.searchsorted(cumsum_lengths, full_indices, right=True) - 1
            repeated_encoded_frame_feat = encoded_frame_feat[segment_indices]
            repeated_vid_proposal_feat = vid_proposal_feat[segment_indices]

            gamma_a = self.gamma_mlp_a(ff_momo_mean)  #
            beta_a = self.beta_mlp_a(ff_momo_mean)  # 同上


            num_frames = repeated_vid_proposal_feat.shape[1]
            # 将γ和β扩展到帧维度 [batch, 32, 256]
            gamma_a = gamma_a.unsqueeze(1).expand(-1, num_frames, -1)
            beta_a = beta_a.unsqueeze(1).expand(-1, num_frames, -1)
            # 逐帧调制：γ⊙A + β
            vid_proposal_feat_a = gamma_a * repeated_vid_proposal_feat + beta_a

            gamma_b = self.gamma_mlp_b(ff_momo_mean)  #
            beta_b = self.beta_mlp_b(ff_momo_mean)  # 同上
            encoded_frame_feat_b = gamma_b * repeated_encoded_frame_feat + beta_b  # 逐元素调制

            frame_scale_scores1 = torch.matmul(F.normalize(repeated_encoded_frame_feat, dim=-1),
                                               F.normalize(video_query, dim=-1).t()).permute(1, 0)
            frame_scale_scores1 = torch.diag(frame_scale_scores1, 0)

            frame_scale_scores2 = torch.matmul(F.normalize(encoded_frame_feat_b, dim=-1),
                                               F.normalize(video_query, dim=-1).t()).permute(1, 0)
            frame_scale_scores2 = torch.diag(frame_scale_scores2, 0)

            # get clip-level retrieval scores
            clip_scale_scores1, _, _,_ = self.get_clip_scale_scores(video_query, repeated_vid_proposal_feat)
            clip_scale_scores1 = torch.diag(clip_scale_scores1, 0)

            clip_scale_scores2, _, _,_ = self.get_clip_scale_scores(video_query, vid_proposal_feat_a)
            clip_scale_scores2 = torch.diag(clip_scale_scores2, 0)

            clip_loss = torch.clamp(clip_scale_scores1 - clip_scale_scores2 + 0.20, 0).mean()
            frame_loss = torch.clamp(frame_scale_scores1 - frame_scale_scores2 + 0.20, 0).mean()


            encoded_frame_feat_b = encoded_frame_feat_b.detach()
            mse_loss1 = F.mse_loss(F.normalize(encoded_frame_feat_b, dim=-1),
                                  F.normalize(repeated_encoded_frame_feat, dim=-1), reduction='mean')

            vid_proposal_feat_a =vid_proposal_feat_a.detach()
            mse_loss2 = F.mse_loss(F.normalize(vid_proposal_feat_a, dim=-1),
                                  F.normalize(repeated_vid_proposal_feat, dim=-1), reduction='mean')
            mse_loss =   (mse_loss1 + mse_loss2)*0.5
            # clip_loss = None
            # print(mse_loss2)
        else:
            mse_loss = None
            frame_loss = None
            clip_loss = None

        return [clip_scale_scores, clip_scale_scores_, label_dict, frame_scale_scores, frame_scale_scores_, video_query,
                selected_indices,selected_indices_second, clip_level_query_context_scores, mse_loss, frame_loss, clip_loss,gathered_features]

    def grounding_task2(self, batch, eee, num_proposals, random_p, tau, gathered_features=None):
        #############################以上是加载特征在（不需要冻结）############################################
        clip_video_feat = batch['clip_video_features']  # 128，30，1024
        query_feat = batch['text_feat']  # 478，30，1024

        query_mask = batch['text_mask']  # 478，30
        query_labels = batch['text_labels']  # 478
        frame_video_feat = batch['frame_video_features']  # 128，128，1024
        frame_video_mask = batch['videos_mask']  # 128，128

        # 检查 frame_video_feat 的第0位置是否是128维度
        if frame_video_feat.size(1) != 128:
            padding_size = 128 - frame_video_feat.size(1)
            frame_video_feat = torch.nn.functional.pad(frame_video_feat, (0, 0, 0, padding_size), "constant", 0)

        # 检查 frame_video_mask 的1位置是否是128维度
        if frame_video_mask.size(1) != 128:
            padding_size = 128 - frame_video_mask.size(1)
            frame_video_mask = torch.nn.functional.pad(frame_video_mask, (0, padding_size), "constant", 0)



        bsz = frame_video_feat.size(0)
        bszzz=query_mask.shape[0]
        ###################################定位需要的参数不需要冻结）#####################################
        weights = batch['weights']
        word_id = batch['word_id']
        # print(word_id)

        # 使用列表推导式展开所有子子列表
        weights1 = [item for sublist in weights for item in sublist]
        word_id1 = [item for sublist in word_id for item in sublist]
        ##################################定位任务#################################################
        # words_len = torch.tensor([tensor.numel() for tensor in word_id], dtype=torch.int64)
        # 使用列表推导式来获取每个 tensor 的元素数量
        words_len = torch.tensor([t.numel() for t in word_id1], device='cuda:0')

        # 使用 torch.clamp 将大于 30 的元素替换成 30
        words_len_clamped = torch.clamp(words_len, max=29)

        # 使用列表推导式和 torch 进行归一化
        normalized_weights = [torch.tensor(w) / sum(w) for w in weights1]

        # 转换为列表（如果需要）
        normalized_weights = [w.tolist() for w in normalized_weights]

        # 填充每个子列表为长度为30，超过30的截断，填充值为0
        filled_A = []
        for sublist in normalized_weights:
            if len(sublist) > 30:
                # 如果子列表长度大于30，截断前30个元素
                filled_A.append(sublist[:30])
            else:
                # 如果子列表长度小于30，填充0
                fill_length = 30 - len(sublist)
                filled_A.append(sublist + [0] * fill_length)

        # 将填充后的列表转换为Tensor
        weights1 = torch.tensor(filled_A)

        # 创建一个新的列表来存储处理后的 tensor
        processed_tensors = []
        # 遍历每个 tensor
        for t in word_id1:
            t = t.squeeze()  # 确保它是 1D tensor

            if len(t) < 30:
                # 如果元素个数小于 30，则填充 0
                padding = torch.zeros(30 - len(t)).cuda()
                processed_tensors.append(torch.cat((t, padding)))
            else:
                # 如果元素个数超过 30，则截断
                processed_tensors.append(t[:30])

        # 将所有处理后的 tensor 组合成一个 128x30 的 tensor
        word_id1 = torch.stack(processed_tensors)
        pred_vec = self.pred_vec.view(1, 1, -1).expand(bszzz, 1, -1)
        frames_mask = frame_video_mask

        # 将 query_labels 转换为 PyTorch tensor
        query_labels_tensor = torch.tensor(query_labels)
        # 获取 tensor 中的最小值和最大值，用于确定数字的范围
        min_val, max_val = query_labels_tensor.min(), query_labels_tensor.max()
        # 创建一个大小为 max_val - min_val + 1 的 tensor，用于存储每个数字的出现次数
        count_tensor = torch.zeros(max_val - min_val + 1, dtype=torch.int)
        # 遍历 tensor，统计每个数字出现的次数
        for i in range(min_val, max_val + 1):
            count_tensor[i - min_val] = (query_labels_tensor == i).sum()
        copy_num = count_tensor
        # copy_num_sum = torch.sum(count_tensor)
        # 创建空的列表来存储复制后的特征和掩码
        new_feat_list = []
        new_mask_list = []
        # 对 copy_num 的每个位置，按其对应值复制 frame_video_feat1 和 frames_mask
        for i in range(copy_num.size(0)):
            # 对于 copy_num[i] 值，复制 frame_video_feat1[i] 次
            replicated_feat = frame_video_feat[i].unsqueeze(0).repeat(copy_num[i], 1, 1)
            replicated_mask = frames_mask[i].unsqueeze(0).repeat(copy_num[i], 1)

            # 将复制后的特征和掩码分别存入列表
            new_feat_list.append(replicated_feat)
            new_mask_list.append(replicated_mask)

        # 将特征和掩码分别堆叠在一起
        frame_video_feat1 = torch.cat(new_feat_list, dim=0)  # 合并特征
        # ori_frames_feat = frame_video_feat1
        frames_mask = torch.cat(new_mask_list, dim=0)  # 合并掩码
        zeros = torch.zeros(frames_mask.shape[0], 1).cuda()
        frames_mask = torch.cat((frames_mask, zeros), dim=1)  # 在列维度拼接
        frames_mask_t = frames_mask
        # props, props_valid = generate_props(frame_video_feat1)

        frame_video_feat1 = torch.cat([frame_video_feat1, pred_vec], dim=1)
        frame_video_feat1 = F.dropout(frame_video_feat1, self.dropout, self.training)
        frame_video_feat1 = self.frame_fc(frame_video_feat1)
        frame_video_feat2 = frame_video_feat1
        query_feat[:, 0] = self.start_vec.cuda()
        words_pos = self.word_pos_encoder(query_feat)

        query_feat = F.dropout(query_feat, self.dropout, self.training)

        query_feat1 = self.word_fc(query_feat)
        words_mask = query_mask

        # proposals scoring
        enc_out, h = self.trans(frame_video_feat1, frames_mask, query_feat1 + words_pos, words_mask, decoding=1)

########################################################################################################################

        gauss_param = torch.sigmoid(self.fc_gauss(h[:, -1])).view(h.shape[0] * num_proposals, 2)
        gauss_center = gauss_param[:, 0]
        gauss_width = gauss_param[:, 1]

        # downsample for effeciency
        props_len = 128 // 4
        keep_idx = torch.linspace(0, 128 - 1, steps=props_len).long()

        frame_video_feat1 = frame_video_feat1[:, keep_idx]

        bszz = h.shape[0]
        frames_mask = frames_mask[:, keep_idx]
        props_feat = frame_video_feat1.unsqueeze(1) \
            .expand(bszz, num_proposals, -1, -1).contiguous().view(bszz * num_proposals, props_len, -1)
        props_mask = frames_mask.unsqueeze(1) \
            .expand(bszz, num_proposals, -1).contiguous().view(bszz * num_proposals, -1)

        gauss_weight = self.generate_gauss_weight(props_len, gauss_center, gauss_width)

        words_feat = query_feat1
        words_feat = words_feat + words_pos
        words_feat1 = words_feat[:, :-1]
        words_mask1 = words_mask[:, :-1]
        words_id1 = word_id1

        words_mask1 = words_mask1.unsqueeze(1) \
            .expand(bszz, num_proposals, -1).contiguous().view(bszz * num_proposals, -1)
        # words_id1 = words_id1.unsqueeze(1) \
        #     .expand(bszz, 8, -1).contiguous().view(bszz * 8, -1)
        words_feat1 = words_feat1.unsqueeze(1) \
            .expand(bszz, num_proposals, -1, -1).contiguous().view(bszz * num_proposals, words_mask1.size(1), -1)

        pos_weight = gauss_weight / gauss_weight.max(dim=-1, keepdim=True)[0]
        _, h, attn_weight, enc_out_retrieval = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2,
                                                          gauss_weight=pos_weight, need_weight=True)

        words_logit = self.fc_comp(h)

        if self.training:

            if gathered_features is not None:

                gathered_features = self.rtg(gathered_features)
                gamma_c = self.gamma_mlp_c(gathered_features)  #
                beta_c = self.beta_mlp_c(gathered_features)  # 同上
                num_frames = frame_video_feat2.shape[1]
                # 将γ和β扩展到帧维度 [batch, 32, 256]
                gamma = gamma_c.unsqueeze(1).expand(-1, num_frames, -1)
                beta = beta_c.unsqueeze(1).expand(-1, num_frames, -1)

                # 逐帧调制：γ⊙A + β
                modulated_features = gamma * frame_video_feat2 + beta

                # proposals scoring
                enc_out_t, h_t = self.trans(modulated_features, frames_mask_t, query_feat1 + words_pos, words_mask, decoding=1)

                gauss_param_t = torch.sigmoid(self.fc_gauss(h_t[:, -1])).view(h_t.shape[0] * num_proposals, 2)
                gauss_center_t = gauss_param_t[:, 0]
                gauss_width_t = gauss_param_t[:, 1]

                modulated_features = modulated_features[:, keep_idx]

                props_feat_t = modulated_features.unsqueeze(1) \
                    .expand(bszz, num_proposals, -1, -1).contiguous().view(bszz * num_proposals, props_len, -1)

                gauss_weight_t = self.generate_gauss_weight(props_len, gauss_center_t, gauss_width_t)

                pos_weight_t = gauss_weight_t / gauss_weight_t.max(dim=-1, keepdim=True)[0]

                _, h_t, attn_weight_t, enc_out_retrieval_t = self.trans(props_feat_t, props_mask, words_feat1, words_mask1,
                                                                  decoding=2,
                                                                  gauss_weight=pos_weight_t, need_weight=True)
                words_logit_t = self.fc_comp(h_t)

            else:
                words_logit_t =None
            neg_1_weight, neg_2_weight = negative_proposal_mining(props_len, gauss_center, gauss_width, eee)

            _, neg_h_1 = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2,
                                    gauss_weight=neg_1_weight)
            neg_words_logit_1 = self.fc_comp(neg_h_1)

            _, neg_h_2 = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2,
                                    gauss_weight=neg_2_weight)
            neg_words_logit_2 = self.fc_comp(neg_h_2)

            _, ref_h = self.trans(frame_video_feat1, frames_mask, words_feat, words_mask, decoding=2)
            ref_words_logit = self.fc_comp(ref_h)

            ###############################加入损失索引最大的候选时刻#######################

            # words_mask = grounding_s['words_mask'].unsqueeze(1) \
            #     .expand(bsz,num_proposals1, -1).contiguous().view(bsz * num_proposals1, -1)

            words_mask_l = words_mask[:, :30]
            words_mask_l = words_mask_l.unsqueeze(1) \
                .expand(bszz, 10, -1).contiguous().view(bszz * 10, -1)

            # words_id = grounding_s['words_id'].unsqueeze(1) \
            #     .expand(bsz, num_proposals1, -1).contiguous().view(bsz * num_proposals1, -1)

            words_id_l = words_id1.unsqueeze(1) \
                .expand(bszz, 10, -1).contiguous().view(bszz * 10, -1)

            nll_loss, acc = cal_nll_loss(words_logit, words_id_l, words_mask_l)

            #nll_loss_video, acc_video = cal_nll_loss( words_logit_video, words_id_l, words_mask_l)
            if words_logit_t is not None:

                  nll_loss_t, acc_t = cal_nll_loss(words_logit_t, words_id_l, words_mask_l)
                  nll_loss_t = nll_loss_t.mean()
                  nll_loss1 = nll_loss.mean()
                  nll_loss_t_c1 = torch.clamp(nll_loss_t - nll_loss1 + 0.15, 0).mean()
                  #nll_loss_t_c2 = torch.clamp(nll_loss_t - nll_loss_video + 0.20, 0).mean()
                  nll_loss_t_c = nll_loss_t_c1
                  modulated_features_mean =  modulated_features.mean(dim=1)
                  frame_video_feat2_mean = frame_video_feat2.mean(dim=1)
                  modulated_features_mean = modulated_features_mean.detach()
                  mse_loss_t_c = F.mse_loss(F.normalize(modulated_features_mean, dim=-1),
                                          F.normalize(frame_video_feat2_mean, dim=-1), reduction='mean')
            else:
                nll_loss_t_c = None
                mse_loss_t_c = None

            idx_10 = nll_loss.view(bszz, 10).argsort(dim=-1)

            gauss_weight_l = pos_weight.view(bszz, 10, -1)

            enc_out_retrieval = enc_out_retrieval.view(bszz, 10, 32, 256)

            top1_idx = idx_10[:, 0]

            top1_idx_expanded = top1_idx.view(bszz, 1, 1).expand(-1, -1, 32)

            selected_weights = torch.gather(gauss_weight_l, 1, top1_idx_expanded).squeeze(1)

            # 首先，我们需要将 top1_idx_expanded 的维度扩展为 [24, 1, 32, 1]，以便与 enc_out_retrieval 的维度匹配
            top1_idx_expanded = top1_idx_expanded.unsqueeze(-1)  # 现在维度是 [24, 1, 32, 1]

            # 使用 torch.gather 从 enc_out_retrieval 中索引出对应的内容
            selected_enc_out = torch.gather(enc_out_retrieval, 1, top1_idx_expanded.expand(-1, -1, -1, 256))

            # 最后，去掉多余的维度
            selected_enc_out = selected_enc_out.squeeze(1)  # 现在维度是 [24, 32, 256]


        else:
            neg_words_logit_1 = None
            neg_words_logit_2 = None
            ref_words_logit = None
            selected_weights = None
            selected_enc_out = None
            nll_loss_t_c = None
            mse_loss_t_c = None
        return {
            'neg_words_logit_1': neg_words_logit_1,
            'neg_words_logit_2': neg_words_logit_2,
            'ref_words_logit': ref_words_logit,
            'words_logit': words_logit,
            'words_id': words_id1,
            'words_mask': words_mask[:, :30],
            'width': gauss_width,
            'center': gauss_center,
            'gauss_weight': pos_weight,
            'selected_weights': selected_weights,
            'selected_enc_out': selected_enc_out,
            'nll_loss_t_c': nll_loss_t_c,
            'mse_loss_t_c': mse_loss_t_c
        }

    def forward(self, batch, eee, num_proposals, random_p, tau, single=None):
        # 如果 single=1，冻结定位任务的参数，更新检索任务的参数
        if single == 1:
            # 在计算 grounding_task 时，不更新其梯度（冻结）
            with torch.no_grad():
                grounding_results = self.grounding_task2(batch, eee, num_proposals, random_p, tau)  # 不更新定位任务的参数
                selected_enc_out = grounding_results['selected_enc_out']
            retrieval_results = self.retrieval_task(batch, selected_enc_out, single=1)  # 正常更新检索任务的参数
            # print('vvvvvvvv')
        # 如果 single=2，冻结检索任务的参数，更新定位任务的参数
        elif single == 2:
            # 在计算 retrieval_task 时，不更新其梯度（冻结）
            with torch.no_grad():
                selected_enc_out = None
                retrieval_results = self.retrieval_task(batch, selected_enc_out, single=2)  # 不更新检索任务的参数
            gathered_features = retrieval_results[-1]
            grounding_results = self.grounding_task2(batch, eee, num_proposals, random_p, tau,gathered_features
                                                    )  # 正常更新定位任务的参数


        # 如果 single=3，两个任务的参数都正常更新
        elif single == 3:
            grounding_results = self.grounding_task2(batch, eee, num_proposals, random_p, tau)  # 正常更新定位任务的参数
            # grounding_results = self.grounding_task(batch,num_proposals,random_p, tau)  # 正常更新定位任务的参数
            # retrieval_results = self.retrieval_task(batch)  # 正常更新检索任务的参数
            retrieval_results = None
        # 如果 single=4，
        elif single == 4:
            # grounding_results = self.grounding_task2(batch,eee, num_proposals, random_p, tau)  # 正常更新定位任务的参数
            # grounding_results = self.grounding_task(batch,num_proposals,random_p, tau)  # 正常更新定位任务的参数
            retrieval_results = self.retrieval_task(batch)  # 正常更新检索任务的参数
            grounding_results = None
        # 返回检索结果和定位结果
        return retrieval_results, grounding_results

    def encode_query(self, query_feat, query_mask):
        encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder,
                                          self.query_pos_embed)  # (N, Lq, D)
        if query_mask is not None:
            mask = query_mask.unsqueeze(1)

        video_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 1

        return video_query

    def encode_context(self, clip_video_feat, frame_video_feat, video_mask=None):

        encoded_clip_feat = self.encode_input(clip_video_feat, None, self.clip_input_proj, self.clip_encoder,
                                              self.clip_pos_embed)
        encoded_clip_feat = self.clip_encoder_2(encoded_clip_feat, None)  # [bs, 30, 384]

        encoded_frame_feat = self.encode_input(frame_video_feat, video_mask, self.frame_input_proj,
                                               self.frame_encoder_1,
                                               self.frame_pos_embed)  # [bs, N, 384]
        encoded_frame_feat = self.frame_encoder_2(encoded_frame_feat, video_mask.unsqueeze(1))

        encoded_frame_feat = self.get_modularized_frames(encoded_frame_feat, video_mask)

        return encoded_frame_feat, encoded_clip_feat

    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer):
        """
        Args:
            feat: (N, L, D_input), torch.float30
            mask: (N, L), torch.float30, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """

        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        return encoder_layer(feat, mask)  # (N, L, D_hidden)

    def get_modularized_queries(self, encoded_query, query_mask):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        return modular_queries.squeeze()

    def get_modularized_frames(self, encoded_query, query_mask):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping_2(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        return modular_queries.squeeze()

    @staticmethod
    def get_clip_scale_scores(modularied_query, context_feat):

        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)

        clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)

        query_context_scores, indices = torch.max(clip_level_query_context_scores,
                                                  dim=1)
        # 在每一行中找到最大分数的索引
        max_score_indices = torch.argmax(query_context_scores, dim=1)

        # 使用这些索引从 indices 中提取对应的值
        selected_indices = indices[torch.arange(indices.size(0)), max_score_indices]


        top2_scores, top2_indices = torch.topk(query_context_scores, k=2, dim=1)

        # 选择第二个最大值的索引
        second_max_indices = top2_indices[:, 1]

        # 使用这些索引从 indices 中提取对应的值
        selected_indices_second = indices[torch.arange(indices.size(0)), second_max_indices]

        # 使用 torch.stack 在第 0 维度上堆叠
        # selected_indices = torch.stack((selected_indices, selected_indices_second ), dim=0)
        # print('stop')
        return query_context_scores, selected_indices, selected_indices_second,clip_level_query_context_scores

    @staticmethod
    def get_unnormalized_clip_scale_scores(modularied_query, context_feat):

        query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)

        output_query_context_scores, indices = torch.max(query_context_scores, dim=1)

        return output_query_context_scores

    def get_pred_from_raw_query(self, query_feat, query_mask, query_labels=None,
                                video_proposal_feat=None, encoded_frame_feat=None,
                                return_query_feats=False):

        video_query = self.encode_query(query_feat, query_mask)

        # get clip-level retrieval scores
        clip_scale_scores, selected_indices,selected_indices_second, clip_level_query_context_scores = self.get_clip_scale_scores(
            # [640,128], [640,128]
            video_query, video_proposal_feat)

        frame_scale_scores = torch.matmul(F.normalize(encoded_frame_feat, dim=-1),
                                          F.normalize(video_query, dim=-1).t()).permute(1, 0)

        if return_query_feats:
            clip_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query, video_proposal_feat)
            frame_scale_scores_ = torch.matmul(encoded_frame_feat, video_query.t()).permute(1, 0)

            return clip_scale_scores, clip_scale_scores_, frame_scale_scores, frame_scale_scores_, selected_indices,selected_indices_second, clip_level_query_context_scores
        else:

            return clip_scale_scores, frame_scale_scores, selected_indices, selected_indices_second,clip_level_query_context_scores

    def _select_proposals(self, props, props_valid, props_align,
                          random_p, num_proposals, tau):
        bsz = props.size(0)
        props = props.view(bsz, -1, 2)
        props_valid = props_valid.view(bsz, -1).cuda()

        props_chosen = []
        props_idx = []

        def choose(size):
            if np.random.rand() < random_p:
                get_id = np.random.choice(np.arange(0, size), replace=False)
            else:
                get_id = 0
            return get_id

        for i, a in enumerate(props_align):
            a = a.contiguous().view(-1).masked_fill(props_valid[i] == 0, 0)

            # reorder
            idx = torch.argsort(a, descending=True)
            props1 = props[i].index_select(dim=0, index=idx)

            # remove illegal
            kidx = props1[:, 0] >= 0
            idx = idx[kidx]
            props1 = props1[kidx]

            pid = choose(props1.size(0))
            cp, cp_idx = [props1[pid]], [idx[pid]]
            for _ in range(1, num_proposals):
                tmp = cp[-1].unsqueeze(0).expand(props1.size(0), 2)
                iou = calculate_IoU_batch((tmp[:, 0].float(), tmp[:, 1].float()),
                                          (props1[:, 0].float(), props1[:, 1].float()))
                kidx = iou < tau
                if int(kidx.sum()) > 2:
                    idx = idx[kidx]
                    props1 = props1[kidx]
                pid = choose(props1.size(0))
                cp.append(props1[pid])
                cp_idx.append(idx[pid])

            cp, cp_idx = torch.stack(cp, 0), torch.stack(cp_idx, 0)
            # print(cp, cp_idx)
            props_chosen.append(cp)
            props_idx.append(cp_idx)
            # exit(0)
        props_chosen = torch.stack(props_chosen, 0)
        props_idx = torch.stack(props_idx, 0)
        # print(props_chosen)
        return props_chosen, props_idx

    def generate_gauss_weight(self, props_len, center, width):
        # pdb.set_trace()
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / 9

        w = 0.3989422804014327
        weight = w / width * torch.exp(-(weight - center) ** 2 / (2 * width ** 2))

        return weight / weight.max(dim=-1, keepdim=True)[0]


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, **kwargs):
        bsz, seq_len, _ = input.size()
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.cuda(input.device)[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


def _generate_mask(x, x_len):
    if False and int(x_len.min()) == x.size(1):
        mask = None
    else:
        mask = []
        for l in x_len:
            mask.append(torch.zeros([x.size(1)]).byte().cuda())
            mask[-1][:l] = 1
        mask = torch.stack(mask, 0)
    return mask


def calculate_IoU_batch(i0, i1):
    union = (torch.min(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.max(torch.stack([i0[1], i1[1]], 0), 0)[0])
    inter = (torch.max(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.min(torch.stack([i0[1], i1[1]], 0), 0)[0])
    iou = 1.0 * (inter[1] - inter[0] + 1) / (union[1] - union[0] + 1)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


def negative_proposal_mining(props_len, center, width, epoch):
    sigma = 9
    max_epoch = 20
    gamma = 0

    def Gauss(pos, w1, c):
        w1 = w1.unsqueeze(-1).clamp(1e-2) / (sigma / 2)
        c = c.unsqueeze(-1)
        w = 0.3989422804014307
        y1 = w / w1 * torch.exp(-(pos - c) ** 2 / (2 * w1 ** 2))
        return y1 / y1.max(dim=-1, keepdim=True)[0]

    weight = torch.linspace(0, 1, props_len)
    weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)

    left_width = torch.clamp(center - width / 2, min=0)
    left_center = left_width * min(epoch / 20, 1) ** gamma * 0.5
    right_width = torch.clamp(1 - center - width / 2, min=0)
    right_center = 1 - right_width * min(epoch / max_epoch, 1) ** gamma * 0.5

    left_neg_weight = Gauss(weight, left_center, left_center)
    right_neg_weight = Gauss(weight, 1 - right_center, right_center)

    return left_neg_weight, right_neg_weight
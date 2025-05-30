# import json
# import torch
# import torch.utils.data as data
# import numpy as np
# import re
# import h5py
# import os
# import pickle
# import nltk
#
# import re
#
# def generate_props(frames_feat):
#
#     num_props = 8
#
#     # 直接创建 prop_width 数组
#     prop_width = np.asarray([1.0 / num_props * i for i in range(1, num_props + 1)])
#
#
#     num_clips = 128
#     prop_width = (prop_width * len(frames_feat)).astype(np.int64)
#     prop_width[prop_width == 0] = 1
#
#     props = []
#     valid = []
#     end = np.arange(1, num_clips + 1).astype(np.int64)
#     for w in prop_width:
#             start = end - w
#             props.append(np.stack([start, end], -1))  # [nc, 2]
#             valid.append(np.logical_and(props[-1][:, 0] >= 0, props[-1][:, 1] <= len(frames_feat)))  # [nc]
#     props = np.stack(props, 1).astype(np.int64)  # [nc, np, 2]
#     # print(props[len(frames_feat)-1], len(frames_feat))
#     valid = np.stack(valid, 1).astype(np.uint8)  # [nc, np]
#     valid_torch = torch.from_numpy(valid)
#     props_torch = torch.from_numpy(props)
#     return props_torch, valid_torch
#
#
#
# def process_captions(captions_text):
#     captions_text1 = []
#     timestamps = []
#     durations = []
#
#     # 正则表达式模式
#     pattern = r'^(.*?)(\[\d+(\.\d+)?,\s*\d+(\.\d+)?\])(.*?)(\d+(\.\d+)?)$'
#
#     # 遍历 captions_text 列表
#     for caption in captions_text:
#         # 使用正则表达式匹配并提取句子、时间戳和时长
#         match = re.match(pattern, caption.strip())
#
#         if match:
#             sentence_part = match.group(1).strip()  # 句子
#             timestamp_str = match.group(2).strip()  # 时间戳部分
#             duration_str = match.group(6).strip()  # 时长部分
#
#             # 解析时间戳
#             timestamp_values = list(map(float, timestamp_str[1:-1].split(',')))
#
#             # 存储结果
#             captions_text1.append(sentence_part)
#             timestamps.append(timestamp_values)
#             durations.append(float(duration_str))
#
#     return captions_text1, timestamps, durations
#
#
#
#
# def getVideoId(cap_id):
#     vid_id = cap_id.split('#')[0]
#     return vid_id
#
# def clean_str(string):
#     string = re.sub(r"[^A-Za-z0-9]", " ", string)
#     return string.strip().lower().split()
#
# def read_video_ids(cap_file):
#     video_ids_list = []
#     with open(cap_file, 'r') as cap_reader:
#         for line in cap_reader.readlines():
#             cap_id, caption = line.strip().split(' ', 1)
#             video_id = getVideoId(cap_id)
#             if video_id not in video_ids_list:
#                 video_ids_list.append(video_id)
#     return video_ids_list
#
# def average_to_fixed_length(visual_input, map_size):
#     visual_input = torch.from_numpy(visual_input)
#     num_sample_clips = map_size
#     num_clips = visual_input.shape[0]
#     idxs = torch.arange(0, num_sample_clips + 1, 1.0) / num_sample_clips * num_clips
#
#     idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips - 1))
#
#     new_visual_input = []
#
#     for i in range(num_sample_clips):
#
#         s_idx, e_idx = idxs[i].item(), idxs[i + 1].item()
#         if s_idx < e_idx:
#             new_visual_input.append(torch.mean(visual_input[s_idx:e_idx], dim=0))
#         else:
#             new_visual_input.append(visual_input[s_idx])
#     new_visual_input = torch.stack(new_visual_input, dim=0).numpy()
#
#
#     return new_visual_input
#
# def uniform_feature_sampling(features, max_len):
#     num_clips = features.shape[0]
#     if max_len is None or num_clips <= max_len:
#         return features
#     idxs = np.arange(0, max_len + 1, 1.0) / max_len * num_clips
#     idxs = np.round(idxs).astype(np.int32)
#     idxs[idxs > num_clips - 1] = num_clips - 1
#     new_features = []
#     for i in range(max_len):
#         s_idx, e_idx = idxs[i], idxs[i + 1]
#         if s_idx < e_idx:
#             new_features.append(np.mean(features[s_idx:e_idx], axis=0))
#         else:
#             new_features.append(features[s_idx])
#     new_features = np.asarray(new_features)
#     return new_features
#
#
# def l2_normalize_np_array(np_array, eps=1e-5):
#     """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
#     return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)
#
#
#
# def collate_train(data):
#     """
#     Build mini-batch tensors from a list of (video, caption) tuples.
#     """
#     # Sort a data list by caption length
#     if data[0][1] is not None:
#         data.sort(key=lambda x: len(x[1]), reverse=True)
#     clip_video_features, frame_video_features, captions, idxs, cap_ids, video_ids, weights_list,word_id_list,timestamps,durations= zip(*data)
#     #videos
#     clip_videos = torch.cat(clip_video_features, dim=0).float()
#
#     video_lengths = [len(frame) for frame in frame_video_features]
#     frame_vec_len = len(frame_video_features[0][0])
#     frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
#     videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
#     for i, frames in enumerate(frame_video_features):
#         end = video_lengths[i]
#         frame_videos[i, :end, :] = frames[:end, :]
#         videos_mask[i, :end] = 1.0
#
#     #captions
#     feat_dim = captions[0][0].shape[-1]
#
#     merge_captions = []
#     all_lengths = []
#     labels = []
#
#     for index, caps in enumerate(captions):
#         labels.extend(index for i in range(len(caps)))
#         all_lengths.extend(len(cap) for cap in caps)
#         merge_captions.extend(cap for cap in caps)
#
#     target = torch.zeros(len(all_lengths), max(all_lengths), feat_dim)
#     words_mask = torch.zeros(len(all_lengths), max(all_lengths))
#
#     for index, cap in enumerate(merge_captions):
#         end = all_lengths[index]
#         target[index, :end, :] = cap[:end, :]
#         words_mask[index, :end] = 1.0
#
#     # 假设 weights_list 和 word_id_list 是 tuple 类型，labels_list 是 list 类型
#
#     # 合并为一个大的 tuple
#     # combined_tuple = weights_list + word_id_list + tuple(labels)
#
#     # 返回合并后的 tuple
#     return dict(
#         clip_video_features=clip_videos,
#         frame_video_features=frame_videos,
#         videos_mask=videos_mask,
#         text_feat=target,
#         text_mask=words_mask,
#         text_labels=labels,
#         weights=weights_list,
#         word_id=word_id_list,
#         timestamps=timestamps,
#         durations=durations
#     )
#
#     # return dict(clip_video_features=clip_videos,
#     #             frame_video_features=frame_videos,
#     #             videos_mask=videos_mask,
#     #             text_feat=target,
#     #             text_mask=words_mask,
#     #             combined_list=combined_list
#     #             )
#
#
# def collate_frame_val(data):
#     clip_video_features, frame_video_features, idxs, video_ids = zip(*data)
#
#     # Merge videos (convert tuple of 1D tensor to 4D tensor)
#     # videos
#     clip_videos = torch.cat(clip_video_features, dim=0).float()
#
#     video_lengths = [len(frame) for frame in frame_video_features]
#     frame_vec_len = len(frame_video_features[0][0])
#     frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
#     videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
#     for i, frames in enumerate(frame_video_features):
#         end = video_lengths[i]
#         frame_videos[i, :end, :] = frames[:end, :]
#         videos_mask[i, :end] = 1.0
#
#     return clip_videos, frame_videos, videos_mask, idxs, video_ids
#
#
# def collate_text_val(data):
#     if data[0][0] is not None:
#         data.sort(key=lambda x: len(x[0]), reverse=True)
#     captions,idxs, cap_ids = zip(*data)
#
#     if captions[0] is not None:
#         # Merge captions (convert tuple of 1D tensor to 2D tensor)
#         lengths = [len(cap) for cap in captions]
#         target = torch.zeros(len(captions), max(lengths), captions[0].shape[-1])
#         words_mask = torch.zeros(len(captions), max(lengths))
#         for i, cap in enumerate(captions):
#             end = lengths[i]
#             target[i, :end] = cap[:end]
#             words_mask[i, :end] = 1.0
#     else:
#         target = None
#         lengths = None
#         words_mask = None
#
#
#     return target, words_mask, idxs, cap_ids
# ######################注意使用不同的词汇表，在不同数据集上训练的时候#####################
# with open('/home/l/data_1/wmz4/cnm-main/data/activitynet/glove.pkl', 'rb') as fp:
#     vocabs=pickle.load(fp)
# ##############################################################################
# class Dataset4PRVR(data.Dataset):
#     """
#     Load captions and video frame features by pre-trained CNN model.
#     """
#
#     def __init__(self, cap_file, visual_feat, text_feat_path, cfg, video2frames=None):
#         # Captions
#         self.captions = {}
#         self.cap_ids = []
#         self.video_ids = []
#         self.vid_caps = {}
#         self.video2frames = video2frames
#         self.vocabs = vocabs
#
#         self.keep_vocab = dict()
#         indexs = 0
#         for w, _ in vocabs['counter'].most_common(8000):
#             self.keep_vocab[w] = indexs
#             indexs += 1
#
#         with open(cap_file, 'r') as cap_reader:
#             for line in cap_reader.readlines():
#                 cap_id, caption = line.strip().split(' ', 1)
#                 video_id = getVideoId(cap_id)
#                 self.captions[cap_id] = caption
#                 self.cap_ids.append(cap_id)
#                 if video_id not in self.video_ids:
#                     self.video_ids.append(video_id)
#                 if video_id in self.vid_caps:
#                     self.vid_caps[video_id].append(cap_id)
#                 else:
#                     self.vid_caps[video_id] = []
#                     self.vid_caps[video_id].append(cap_id)
#         self.visual_feat = visual_feat
#         self.text_feat_path = text_feat_path
#
#         self.map_size = cfg['map_size']
#         self.max_ctx_len = cfg['max_ctx_l']
#         self.max_desc_len = cfg['max_desc_l']
#
#         self.open_file = False
#         self.length = len(self.vid_caps)
#
#
#     def __getitem__(self, index):
#         if self.open_file:
#             self.open_file = True
#         else:
#             self.text_feat = h5py.File(self.text_feat_path, 'r')
#             self.open_file = True
#
#         video_id = self.video_ids[index]
#         cap_ids = self.vid_caps[video_id]
#
#         # 获取对应的视频帧ID列表
#         frame_list = self.video2frames[video_id]
#
#         # 处理视频帧特征
#         frame_vecs = []
#         for frame_id in frame_list:
#             frame_vecs.append(self.visual_feat.read_one(frame_id))
#
#         clip_video_feature = average_to_fixed_length(np.array(frame_vecs), self.map_size)
#         clip_video_feature = l2_normalize_np_array(clip_video_feature)
#         clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)
#
#         frame_video_feature = uniform_feature_sampling(np.array(frame_vecs), self.max_ctx_len)
#         frame_video_feature = l2_normalize_np_array(frame_video_feature)
#         frame_video_feature = torch.from_numpy(frame_video_feature)
#
#         props, props_valid = generate_props(frame_video_feature)
#
#         # 处理文本描述特征
#         cap_tensors = []
#         captions_text = []  # 用于存储文本描述
#         for cap_id in cap_ids:
#             cap_feat = self.text_feat[cap_id][...]
#             cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
#             cap_tensors.append(cap_tensor)
#
#             # 将对应的文本描述添加到 captions 列表中
#             captions_text.append(self.captions[cap_id])
#
#
#
#         captions_text1, timestamps, durations = process_captions(captions_text)
#
#
#
#         weights_list = []
#         # for sentence in self.annotations[index]['sentences']:
#         # print(sentence_sample)
#
#         word_id_list= []
#
#
#         for sentence in captions_text1:
#             words = []
#             sentence_weights = []
#             for word, tag in nltk.pos_tag(nltk.tokenize.word_tokenize(sentence)):
#                 word = word.lower()
#                 if word in self.keep_vocab:
#                     if 'NN' in tag:
#                         sentence_weights.append(2)
#                     elif 'VB' in tag:
#                         sentence_weights.append(2)
#                     elif 'JJ' in tag or 'RB' in tag:
#                         sentence_weights.append(2)
#                     else:
#                         sentence_weights.append(1)
#                     words.append(word)
#             weights_list.append(sentence_weights)
#             words_id = [self.keep_vocab[w] for w in words]
#             words_id =  torch.tensor(words_id)
#             word_idxs = words_id
#             word_id_list.append(word_idxs)
#
#
#
#
#         # 返回：增加 captions
#         return clip_video_feature, frame_video_feature, cap_tensors, index, cap_ids, video_id,weights_list,word_id_list,timestamps, durations
#
#     def __len__(self):
#         return self.length
#
#
# class VisDataSet4PRVR(data.Dataset):
#
#     def __init__(self, visual_feat, video2frames, cfg, video_ids=None):
#         self.visual_feat = visual_feat
#         self.video2frames = video2frames
#         if video_ids is not None:
#             self.video_ids = video_ids
#         else:
#             self.video_ids = video2frames.keys()
#         self.length = len(self.video_ids)
#         self.map_size = cfg['map_size']
#         self.max_ctx_len = cfg['max_ctx_l']
#     def __getitem__(self, index):
#         video_id = self.video_ids[index]
#         # frame_list = self.video2frames[video_id]
#
#
#         try:
#             # 尝试访问 video2frames 字典
#             frame_list = self.video2frames[video_id]
#         except KeyError:
#             # 如果 video_id 不在 video2frames 中，跳过（pass）
#             print(f"Warning: {video_id} not found in video2frames, skipping.")
#             pass
#
#         frame_vecs = []
#         for frame_id in frame_list:
#             frame_vecs.append(self.visual_feat.read_one(frame_id))
#         clip_video_feature = average_to_fixed_length(np.array(frame_vecs), self.map_size)
#         clip_video_feature = l2_normalize_np_array(clip_video_feature)
#         clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)
#
#         frame_video_feature = uniform_feature_sampling(np.array(frame_vecs), self.max_ctx_len)
#         frame_video_feature = l2_normalize_np_array(frame_video_feature)
#         frame_video_feature = torch.from_numpy(frame_video_feature)
#
#         return clip_video_feature, frame_video_feature, index, video_id
#
#     def __len__(self):
#         return self.length
#
#
# class TxtDataSet4PRVR(data.Dataset):
#     """
#     Load captions
#     """
#
#     def __init__(self, cap_file, text_feat_path, cfg):
#         # Captions
#         self.captions = {}
#         self.cap_ids = []
#         with open(cap_file, 'r') as cap_reader:
#             for line in cap_reader.readlines():
#                 cap_id, caption = line.strip().split(' ', 1)
#                 self.captions[cap_id] = caption
#                 self.cap_ids.append(cap_id)
#         self.text_feat_path = text_feat_path
#         self.max_desc_len = cfg['max_desc_l']
#         self.open_file = False
#         self.length = len(self.cap_ids)
#
#     def __getitem__(self, index):
#         cap_id = self.cap_ids[index]
#         if self.open_file:
#             self.open_file = True
#         else:
#             self.text_feat = h5py.File(self.text_feat_path, 'r')
#
#             self.open_file = True
#
#
#         cap_feat = self.text_feat[cap_id][...]
#
#         cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
#
#         return cap_tensor, index, cap_id
#
#     def __len__(self):
#         return self.length
#
#
# if __name__ == '__main__':
#     pass
#
#

import json
import torch
import torch.utils.data as data
import numpy as np
import re
import h5py
import os
import pickle
import nltk

import re

def generate_props(frames_feat):

    num_props = 8

    # 直接创建 prop_width 数组
    prop_width = np.asarray([1.0 / num_props * i for i in range(1, num_props + 1)])


    num_clips = 128
    prop_width = (prop_width * len(frames_feat)).astype(np.int64)
    prop_width[prop_width == 0] = 1

    props = []
    valid = []
    end = np.arange(1, num_clips + 1).astype(np.int64)
    for w in prop_width:
            start = end - w
            props.append(np.stack([start, end], -1))  # [nc, 2]
            valid.append(np.logical_and(props[-1][:, 0] >= 0, props[-1][:, 1] <= len(frames_feat)))  # [nc]
    props = np.stack(props, 1).astype(np.int64)  # [nc, np, 2]
    # print(props[len(frames_feat)-1], len(frames_feat))
    valid = np.stack(valid, 1).astype(np.uint8)  # [nc, np]
    valid_torch = torch.from_numpy(valid)
    props_torch = torch.from_numpy(props)
    return props_torch, valid_torch



def process_captions(captions_text):
    captions_text1 = []
    timestamps = []
    durations = []

    # 正则表达式模式
    pattern = r'^(.*?)(\[\d+(\.\d+)?,\s*\d+(\.\d+)?\])(.*?)(\d+(\.\d+)?)$'

    # 遍历 captions_text 列表
    for caption in captions_text:
        # 使用正则表达式匹配并提取句子、时间戳和时长
        match = re.match(pattern, caption.strip())

        if match:
            sentence_part = match.group(1).strip()  # 句子
            timestamp_str = match.group(2).strip()  # 时间戳部分
            duration_str = match.group(6).strip()  # 时长部分

            # 解析时间戳
            timestamp_values = list(map(float, timestamp_str[1:-1].split(',')))

            # 存储结果
            captions_text1.append(sentence_part)
            timestamps.append(timestamp_values)
            durations.append(float(duration_str))

    return captions_text1, timestamps, durations




def getVideoId(cap_id):
    vid_id = cap_id.split('#')[0]
    return vid_id

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower().split()

def read_video_ids(cap_file):
    video_ids_list = []
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader.readlines():
            cap_id, caption = line.strip().split(' ', 1)
            video_id = getVideoId(cap_id)
            if video_id not in video_ids_list:
                video_ids_list.append(video_id)
    return video_ids_list

def average_to_fixed_length(visual_input, map_size):
    visual_input = torch.from_numpy(visual_input)
    num_sample_clips = map_size
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips + 1, 1.0) / num_sample_clips * num_clips

    idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips - 1))

    new_visual_input = []

    for i in range(num_sample_clips):

        s_idx, e_idx = idxs[i].item(), idxs[i + 1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx], dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0).numpy()


    return new_visual_input

def uniform_feature_sampling(features, max_len):
    num_clips = features.shape[0]
    if max_len is None or num_clips <= max_len:
        return features
    idxs = np.arange(0, max_len + 1, 1.0) / max_len * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_features = []
    for i in range(max_len):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_features.append(np.mean(features[s_idx:e_idx], axis=0))
        else:
            new_features.append(features[s_idx])
    new_features = np.asarray(new_features)
    return new_features


def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)




def collate_test(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    clip_video_features, frame_video_features, captions, idxs, cap_ids, video_ids, weights_list,word_id_list,timestamps,durations,frames_len= zip(*data)



    #videos
    clip_videos = torch.cat(clip_video_features, dim=0).float()

    video_lengths = [len(frame) for frame in frame_video_features]
    frame_vec_len = len(frame_video_features[0][0])
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    #captions
    feat_dim = captions[0][0].shape[-1]

    merge_captions = []
    all_lengths = []
    labels = []

    for index, caps in enumerate(captions):
        labels.extend(index for i in range(len(caps)))
        all_lengths.extend(len(cap) for cap in caps)
        merge_captions.extend(cap for cap in caps)

    target = torch.zeros(len(all_lengths), max(all_lengths), feat_dim)
    words_mask = torch.zeros(len(all_lengths), max(all_lengths))

    for index, cap in enumerate(merge_captions):
        end = all_lengths[index]
        target[index, :end, :] = cap[:end, :]
        words_mask[index, :end] = 1.0

    # 假设 weights_list 和 word_id_list 是 tuple 类型，labels_list 是 list 类型

    # 合并为一个大的 tuple
    # combined_tuple = weights_list + word_id_list + tuple(labels)

    # 返回合并后的 tuple
    return dict(
        clip_video_features=clip_videos,
        frame_video_features=frame_videos,
        videos_mask=videos_mask,
        text_feat=target,
        text_mask=words_mask,
        text_labels=labels,
        weights=weights_list,
        word_id=word_id_list,
        timestamps=timestamps,
        durations=durations,
        frames_len=frames_len
    )


def collate_train(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    clip_video_features, frame_video_features, captions, idxs, cap_ids, video_ids, weights_list,word_id_list,timestamps,durations,frames_len= zip(*data)

    #videos
    clip_videos = torch.cat(clip_video_features, dim=0).float()

    video_lengths = [len(frame) for frame in frame_video_features]
    frame_vec_len = len(frame_video_features[0][0])
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    #captions
    feat_dim = captions[0][0].shape[-1]

    merge_captions = []
    all_lengths = []
    labels = []

    for index, caps in enumerate(captions):
        labels.extend(index for i in range(len(caps)))
        all_lengths.extend(len(cap) for cap in caps)
        merge_captions.extend(cap for cap in caps)

    target = torch.zeros(len(all_lengths), max(all_lengths), feat_dim)
    words_mask = torch.zeros(len(all_lengths), max(all_lengths))

    for index, cap in enumerate(merge_captions):
        end = all_lengths[index]
        target[index, :end, :] = cap[:end, :]
        words_mask[index, :end] = 1.0

    # 假设 weights_list 和 word_id_list 是 tuple 类型，labels_list 是 list 类型

    # 合并为一个大的 tuple
    # combined_tuple = weights_list + word_id_list + tuple(labels)

    # 返回合并后的 tuple
    return dict(
        clip_video_features=clip_videos,
        frame_video_features=frame_videos,
        videos_mask=videos_mask,
        text_feat=target,
        text_mask=words_mask,
        text_labels=labels,
        weights=weights_list,
        word_id=word_id_list,
        timestamps=timestamps,
        durations=durations,
        frames_len=frames_len
    )

    # return dict(clip_video_features=clip_videos,
    #             frame_video_features=frame_videos,
    #             videos_mask=videos_mask,
    #             text_feat=target,
    #             text_mask=words_mask,
    #             combined_list=combined_list
    #             )


def collate_frame_val(data):
    clip_video_features, frame_video_features, idxs, video_ids = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    # videos
    clip_videos = torch.cat(clip_video_features, dim=0).float()

    video_lengths = [len(frame) for frame in frame_video_features]
    frame_vec_len = len(frame_video_features[0][0])
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    return clip_videos, frame_videos, videos_mask, idxs, video_ids


def collate_text_val(data):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    captions,idxs, cap_ids = zip(*data)

    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = torch.zeros(len(captions), max(lengths), captions[0].shape[-1])
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        target = None
        lengths = None
        words_mask = None


    return target, words_mask, idxs, cap_ids
######################注意使用不同的词汇表，在不同数据集上训练的时候#####################
with open('/mnt/data/wmz/4/1/GMMFormer-main/src/Datasets/PRVR/activitynet/glove.pkl', 'rb') as fp:
    vocabs=pickle.load(fp)
##############################################################################
class Dataset4PRVR(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, cap_file, visual_feat, text_feat_path, cfg, video2frames=None):
        # Captions
        self.captions = {}
        self.cap_ids = []
        self.video_ids = []
        self.vid_caps = {}
        self.video2frames = video2frames
        self.vocabs = vocabs

        self.keep_vocab = dict()
        indexs = 0
        for w, _ in vocabs['counter'].most_common(8000):
            self.keep_vocab[w] = indexs
            indexs += 1

        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                video_id = getVideoId(cap_id)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                if video_id not in self.video_ids:
                    self.video_ids.append(video_id)
                if video_id in self.vid_caps:
                    self.vid_caps[video_id].append(cap_id)
                else:
                    self.vid_caps[video_id] = []
                    self.vid_caps[video_id].append(cap_id)
        self.visual_feat = visual_feat
        self.text_feat_path = text_feat_path

        self.map_size = cfg['map_size']
        self.max_ctx_len = cfg['max_ctx_l']
        self.max_desc_len = cfg['max_desc_l']

        self.open_file = False
        self.length = len(self.vid_caps)


    def __getitem__(self, index):
        if self.open_file:
            self.open_file = True
        else:
            self.text_feat = h5py.File(self.text_feat_path, 'r')
            self.open_file = True

        video_id = self.video_ids[index]
        cap_ids = self.vid_caps[video_id]

        # 获取对应的视频帧ID列表
        frame_list = self.video2frames[video_id]

        # 处理视频帧特征
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))

        clip_video_feature = average_to_fixed_length(np.array(frame_vecs), self.map_size)
        clip_video_feature = l2_normalize_np_array(clip_video_feature)
        clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)

        frame_video_feature = uniform_feature_sampling(np.array(frame_vecs), self.max_ctx_len)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)

        frames_len=[]
        number=frame_video_feature.size(0)
        frames_len.append(number)

        props, props_valid = generate_props(frame_video_feature)

        # 处理文本描述特征
        cap_tensors = []
        captions_text = []  # 用于存储文本描述
        for cap_id in cap_ids:
            cap_feat = self.text_feat[cap_id][...]
            cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
            cap_tensors.append(cap_tensor)

            # 将对应的文本描述添加到 captions 列表中
            captions_text.append(self.captions[cap_id])



        captions_text1, timestamps, durations = process_captions(captions_text)



        weights_list = []
        # for sentence in self.annotations[index]['sentences']:
        # print(sentence_sample)

        word_id_list= []


        for sentence in captions_text1:
            words = []
            sentence_weights = []
            for word, tag in nltk.pos_tag(nltk.tokenize.word_tokenize(sentence)):
                word = word.lower()
                if word in self.keep_vocab:
                    if 'NN' in tag:
                        sentence_weights.append(2)
                    elif 'VB' in tag:
                        sentence_weights.append(2)
                    elif 'JJ' in tag or 'RB' in tag:
                        sentence_weights.append(2)
                    else:
                        sentence_weights.append(1)
                    words.append(word)
            weights_list.append(sentence_weights)
            words_id = [self.keep_vocab[w] for w in words]
            words_id =  torch.tensor(words_id)
            word_idxs = words_id
            word_id_list.append(word_idxs)




        # 返回：增加 captions
        return clip_video_feature, frame_video_feature, cap_tensors, index, cap_ids, video_id,weights_list,word_id_list,timestamps, durations,frames_len

    def __len__(self):
        return self.length


class VisDataSet4PRVR(data.Dataset):

    def __init__(self, visual_feat, video2frames, cfg, video_ids=None):
        self.visual_feat = visual_feat
        self.video2frames = video2frames
        if video_ids is not None:
            self.video_ids = video_ids
        else:
            self.video_ids = video2frames.keys()
        self.length = len(self.video_ids)
        self.map_size = cfg['map_size']
        self.max_ctx_len = cfg['max_ctx_l']
    def __getitem__(self, index):
        video_id = self.video_ids[index]
        # frame_list = self.video2frames[video_id]


        try:
            # 尝试访问 video2frames 字典
            frame_list = self.video2frames[video_id]
        except KeyError:
            # 如果 video_id 不在 video2frames 中，跳过（pass）
            print(f"Warning: {video_id} not found in video2frames, skipping.")
            pass

        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))
        clip_video_feature = average_to_fixed_length(np.array(frame_vecs), self.map_size)
        clip_video_feature = l2_normalize_np_array(clip_video_feature)
        clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)

        frame_video_feature = uniform_feature_sampling(np.array(frame_vecs), self.max_ctx_len)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)

        return clip_video_feature, frame_video_feature, index, video_id

    def __len__(self):
        return self.length


class TxtDataSet4PRVR(data.Dataset):
    """
    Load captions
    """

    def __init__(self, cap_file, text_feat_path, cfg):
        # Captions
        self.captions = {}
        self.cap_ids = []
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
        self.text_feat_path = text_feat_path
        self.max_desc_len = cfg['max_desc_l']
        self.open_file = False
        self.length = len(self.cap_ids)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        if self.open_file:
            self.open_file = True
        else:
            self.text_feat = h5py.File(self.text_feat_path, 'r')

            self.open_file = True


        cap_feat = self.text_feat[cap_id][...]

        cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]

        return cap_tensor, index, cap_id

    def __len__(self):
        return self.length


if __name__ == '__main__':
    pass




# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import Parameter
#
#
# class MultiheadAttention(nn.Module):
#     """Multi-headed attention.
#
#     See "Attention Is All You Need" for more details.
#     """
#
#     def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
#         self.scaling = self.head_dim ** -0.5
#
#         self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
#         if bias:
#             self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
#         else:
#             self.register_parameter('in_proj_bias', None)
#         self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#
#         if add_bias_kv:
#             self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
#             self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
#         else:
#             self.bias_k = self.bias_v = None
#
#         self.add_zero_attn = add_zero_attn
#
#         self.reset_parameters()
#
#         self.onnx_trace = False
#
#     def prepare_for_onnx_export_(self):
#         self.onnx_trace = True
#
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.in_proj_weight)
#         nn.init.xavier_uniform_(self.out_proj.weight)
#         if self.in_proj_bias is not None:
#             nn.init.constant_(self.in_proj_bias, 0.)
#             nn.init.constant_(self.out_proj.bias, 0.)
#         if self.bias_k is not None:
#             nn.init.xavier_normal_(self.bias_k)
#         if self.bias_v is not None:
#             nn.init.xavier_normal_(self.bias_v)
#
#     def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
#                 need_weights=True, static_kv=False, attn_mask=None, fast_weights=None,
#                 gauss_weight=None):
#         """Input shape: Time x Batch x Channel
#
#         Self-attention can be implemented by passing in the same arguments for
#         query, key and value. Timesteps can be masked by supplying a T x T mask in the
#         `attn_mask` argument. Padding elements can be excluded from
#         the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
#         batch x src_len, where padding elements are indicated by 1s.
#         """
#
#         qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
#         kv_same = key.data_ptr() == value.data_ptr()
#
#         tgt_len, bsz, embed_dim = query.size()
#         assert embed_dim == self.embed_dim
#         assert list(query.size()) == [tgt_len, bsz, embed_dim]
#         assert key.size() == value.size()
#
#         if incremental_state is not None:
#             saved_state = self._get_input_buffer(incremental_state)
#             if 'prev_key' in saved_state:
#                 # previous time steps are cached - no need to recompute
#                 # key and value if they are static
#                 if static_kv:
#                     assert kv_same and not qkv_same
#                     key = value = None
#         else:
#             saved_state = None
#
#         if qkv_same:
#             # self-attention
#             q, k, v = self.in_proj_qkv(query)
#         elif kv_same:
#             # encoder-decoder attention
#             q = self.in_proj_q(query)
#             if key is None:
#                 assert value is None
#                 k = v = None
#             else:
#                 k, v = self.in_proj_kv(key)
#         else:
#             q = self.in_proj_q(query)
#             k = self.in_proj_k(key)
#             v = self.in_proj_v(value)
#         q = q * self.scaling
#
#         if self.bias_k is not None:
#             assert self.bias_v is not None
#             k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
#             v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
#             if attn_mask is not None:
#                 attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
#             if key_padding_mask is not None:
#                 key_padding_mask = torch.cat(
#                     [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)
#
#         q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
#         if k is not None:
#             k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
#         if v is not None:
#             v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
#
#         if saved_state is not None:
#             # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
#             if 'prev_key' in saved_state:
#                 prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
#                 if static_kv:
#                     k = prev_key
#                 else:
#                     k = torch.cat((prev_key, k), dim=1)
#             if 'prev_value' in saved_state:
#                 prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
#                 if static_kv:
#                     v = prev_value
#                 else:
#                     v = torch.cat((prev_value, v), dim=1)
#             saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
#             saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)
#
#             self._set_input_buffer(incremental_state, saved_state)
#
#         src_len = k.size(1)
#
#         # This is part of a workaround to get around fork/join parallelism
#         # not supporting Optional types.
#         if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
#             key_padding_mask = None
#
#         if key_padding_mask is not None:
#             assert key_padding_mask.size(0) == bsz
#             assert key_padding_mask.size(1) == src_len
#
#         if self.add_zero_attn:
#             src_len += 1
#             k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
#             v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
#             if attn_mask is not None:
#                 attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
#             if key_padding_mask is not None:
#                 key_padding_mask = torch.cat(
#                     [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)
#
#         attn_weights = torch.bmm(q, k.transpose(1, 2))
#         assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
#
#         if attn_mask is not None:
#             attn_mask = attn_mask.unsqueeze(0)
#             if self.onnx_trace:
#                 attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
#             attn_weights += attn_mask
#
#         if key_padding_mask is not None:
#             # don't attend to padding symbols
#             attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
#             if self.onnx_trace:
#                 attn_weights = torch.where(
#                     key_padding_mask.unsqueeze(1).unsqueeze(2),
#                     torch.Tensor([float("-Inf")]),
#                     attn_weights.float()
#                 ).type_as(attn_weights)
#             else:
#                 attn_weights = attn_weights.float().masked_fill(
#                     key_padding_mask.unsqueeze(1).unsqueeze(2) == 1,
#                     float('-1e30'),
#                 ).type_as(attn_weights)  # FP16 support: cast to float and back
#             attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
#
#         from fairseq import utils
#         attn_weights = utils.softmax(
#             attn_weights, dim=-1, onnx_trace=self.onnx_trace,
#         ).type_as(attn_weights)
#
#         if gauss_weight is not None:
#             # gauss_weight = gauss_weight.unsqueeze(1).repeat(self.num_heads, tgt_len, 1)
#             gauss_weight = gauss_weight.unsqueeze(1).unsqueeze(1) \
#                 .expand(-1, self.num_heads, tgt_len, -1).reshape(*attn_weights.shape)
#             attn_weights = attn_weights * (gauss_weight + 1e-10)
#             attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
#
#         attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
#
#         attn = torch.bmm(attn_weights, v)
#         assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
#         if (self.onnx_trace and attn.size(1) == 1):
#             # when ONNX tracing a single decoder step (sequence length == 1)
#             # the transpose is a no-op copy before view, thus unnecessary
#             attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
#         else:
#             attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
#         attn = self.out_proj(attn)
#
#         if need_weights:
#             # average attention weights over heads
#             attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
#             attn_weights = attn_weights.sum(dim=1) / self.num_heads
#         else:
#             attn_weights = None
#
#         return attn, attn_weights
#
#     def in_proj_qkv(self, query):
#         return self._in_proj(query).chunk(3, dim=-1)
#
#     def in_proj_kv(self, key):
#         return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)
#
#     def in_proj_q(self, query):
#         return self._in_proj(query, end=self.embed_dim)
#
#     def in_proj_k(self, key):
#         return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
#
#     def in_proj_v(self, value):
#         return self._in_proj(value, start=2 * self.embed_dim)
#
#     def _in_proj(self, input, start=0, end=None):
#         weight = self.in_proj_weight
#         bias = self.in_proj_bias
#         weight = weight[start:end, :]
#         if bias is not None:
#             bias = bias[start:end]
#         return F.linear(input, weight, bias)
#
#     def reorder_incremental_state(self, incremental_state, new_order):
#         """Reorder buffered internal state (for incremental generation)."""
#         input_buffer = self._get_input_buffer(incremental_state)
#         if input_buffer is not None:
#             for k in input_buffer.keys():
#                 input_buffer[k] = input_buffer[k].index_select(0, new_order)
#             self._set_input_buffer(incremental_state, input_buffer)
#
#
# def fill_with_neg_inf(t):
#     """FP16-compatible function that fills a tensor with -inf."""
#     return t.float().fill_(float('-inf')).type_as(t)
#
#
# # TransformerEncoder
# class TransformerEncoder(nn.Module):
#     def __init__(self, num_layers, d_model, num_heads, dropout=0.0):
#         super().__init__()
#         self.encoder_layers = nn.ModuleList([
#             TransformerEncoderLayer(d_model, num_heads, dropout)
#             for _ in range(num_layers)
#         ])
#
#     def forward(self, x, mask=None):
#         non_padding_mask = None if mask is None else 1 - mask
#         x = x.transpose(0, 1)
#         for layer in self.encoder_layers:
#             x = layer(x, non_padding_mask)
#         x = x.transpose(0, 1)
#         return x
#
#
# class TransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, dropout=0.0):
#         super().__init__()
#         d_model = d_model
#         num_heads = num_heads
#         self.dropout = dropout
#
#         self.self_attn = MultiheadAttention(d_model, num_heads)
#         self.self_attn_layer_norm = nn.LayerNorm(d_model)
#         self.fc1 = nn.Linear(d_model, d_model << 1)
#         self.fc2 = nn.Linear(d_model << 1, d_model)
#         self.final_layer_norm = nn.LayerNorm(d_model)
#
#     def forward(self, x, mask):
#         dim = x.size(0)
#
#         attn_mask = None if self.attn_mask is None else self.attn_mask.cuda()[:dim, :dim]
#         res = x
#         x, weight = self.self_attn(x, x, x, mask, attn_mask=attn_mask)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = res + x
#         x = self.self_attn_layer_norm(x)
#
#         res = x
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = res + x
#         x = self.final_layer_norm(x)
#         return x
import torch.nn as nn
import torch.nn.functional as F

from Models.modules import MultiheadAttention


# TransformerEncoder
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        non_padding_mask = None if mask is None else 1 - mask
        x = x.transpose(0, 1)
        for layer in self.encoder_layers:
            x = layer(x, non_padding_mask)
        x = x.transpose(0, 1)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        d_model = d_model
        num_heads = num_heads
        self.dropout = dropout

        self.self_attn = MultiheadAttention(d_model, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model << 1)
        self.fc2 = nn.Linear(d_model << 1, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        dim = x.size(0)

        attn_mask = None if self.attn_mask is None else self.attn_mask.cuda()[:dim, :dim]
        res = x
        x, weight = self.self_attn(x, x, x, mask, attn_mask=attn_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        x = self.self_attn_layer_norm(x)

        res = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        x = self.final_layer_norm(x)
        return x

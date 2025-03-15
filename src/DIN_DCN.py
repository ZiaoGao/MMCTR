import torch
from torch import nn
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, DIN_Attention, Dice, CrossNetV2, CrossNetMix, MultiHeadTargetAttention
from fuxictr.utils import not_in_whitelist

class DCN_v2(nn.Module):
    """
    DCNv2 模型，支持四种结构模式：
      1) crossnet_only: 仅 CrossNet
      2) stacked: CrossNet 输出后接一层 DNN
      3) parallel: CrossNet 和 DNN 并行输出后再拼接
      4) stacked_parallel: CrossNet + DNN（串行） 与 原始特征 + DNN（并行）后再拼接

    同时支持通过 CrossNetMix 来替换 CrossNetV2，从而使用低秩分解。
    """
    def __init__(self, 
                 input_dim,                      # 整个模型输入的维度 (batch_size, input_dim)
                 model_structure="parallel",     # ["crossnet_only", "stacked", "parallel", "stacked_parallel"]
                 use_low_rank_mixture=True,     
                 low_rank=32,
                 num_experts=4,
                 cross_layers=3,                 # CrossNet 或 CrossNetMix 的层数
                 stacked_dnn_hidden_units=[],    # stacked DNN 的多层隐藏单元
                 parallel_dnn_hidden_units=[],   # parallel DNN 的多层隐藏单元
                 hidden_activations="ReLU",      # DNN 隐层激活函数
                 net_dropout=0, 
                 batch_norm=True,
                 output_activation=None          # 最后输出层激活，可为 "sigmoid" 或自定义 nn.Module
                 ):
        super(DCN_v2, self).__init__()

        # 1) CrossNet or CrossNetMix
        if use_low_rank_mixture:
            # 使用低秩分解版本的 CrossNetMix
            self.crossnet = CrossNetMix(
                input_dim, 
                cross_layers, 
                low_rank=low_rank, 
                num_experts=num_experts
            )
        else:
            self.crossnet = CrossNetV2(
                input_dim, 
                cross_layers
            )

        # 2) 模型结构设置：crossnet_only、stacked、parallel、stacked_parallel
        self.model_structure = model_structure
        assert self.model_structure in ["crossnet_only", "stacked", "parallel", "stacked_parallel"], \
               f"model_structure={self.model_structure} not supported!"

        # 3.1) 如果需要在 CrossNet 输出后再串行堆叠 DNN
        self.stacked_dnn = None
        stacked_output_dim = 0
        if self.model_structure in ["stacked", "stacked_parallel"]:
            self.stacked_dnn = MLP_Block(
                input_dim=input_dim,
                output_dim=None,  # 输出隐层
                hidden_units=stacked_dnn_hidden_units,
                hidden_activations=hidden_activations,
                output_activation=None, 
                dropout_rates=net_dropout,
                batch_norm=batch_norm
            )
            # 如果隐藏层不为空，则最后一层隐单元数就是 stacked_output_dim
            if len(stacked_dnn_hidden_units) > 0:
                stacked_output_dim = stacked_dnn_hidden_units[-1]
            else:
                stacked_output_dim = input_dim

        # 3.2) 如果要并行一条 DNN
        self.parallel_dnn = None
        parallel_output_dim = 0
        if self.model_structure in ["parallel", "stacked_parallel"]:
            self.parallel_dnn = MLP_Block(
                input_dim=input_dim,
                output_dim=None,  # 输出隐层
                hidden_units=parallel_dnn_hidden_units,
                hidden_activations=hidden_activations,
                output_activation=None, 
                dropout_rates=net_dropout,
                batch_norm=batch_norm
            )
            if len(parallel_dnn_hidden_units) > 0:
                parallel_output_dim = parallel_dnn_hidden_units[-1]
            else:
                parallel_output_dim = input_dim

        # 4) 最终拼接维度
        if self.model_structure == "crossnet_only":
            final_dim = input_dim
        elif self.model_structure == "stacked":
            final_dim = stacked_output_dim
        elif self.model_structure == "parallel":
            final_dim = input_dim + parallel_output_dim
        else:  # "stacked_parallel"
            final_dim = stacked_output_dim + parallel_output_dim

        # 5) 最后一层线性
        self.fc = nn.Linear(final_dim, 1)

        # 6) 最后的输出激活函数
        self.output_activation = None
        if output_activation is not None:
            if isinstance(output_activation, str) and output_activation.lower() == "sigmoid":
                self.output_activation = nn.Sigmoid()
            elif isinstance(output_activation, nn.Module):
                self.output_activation = output_activation
            # 其他字符串如 "relu"、"tanh" 需要自行扩展

    def forward(self, X):
        """
        X: [batch_size, input_dim]
        """
        # 1) CrossNet 或 CrossNetMix 输出
        cross_out = self.crossnet(X)

        # 2) 根据不同的结构进行拼接
        if self.model_structure == "crossnet_only":
            final_out = cross_out

        elif self.model_structure == "stacked":
            final_out = self.stacked_dnn(cross_out)

        elif self.model_structure == "parallel":
            dnn_out = self.parallel_dnn(X)  # 并行 DNN 直接对 X 做处理
            final_out = torch.cat([cross_out, dnn_out], dim=-1)

        else:  # self.model_structure == "stacked_parallel"
            final_out = torch.cat([
                self.stacked_dnn(cross_out),
                self.parallel_dnn(X)
            ], dim=-1)

        # 3) 最终线性层 + 可选激活
        y_pred = self.fc(final_out)
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)

        return y_pred


class DIN_DCN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DIN_DCN", 
                 gpu=-1, 
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_hidden_units=[64],
                 attention_hidden_activations="Dice",
                 attention_output_activation=None,
                 attention_dropout=0,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 net_dropout=0, 
                 batch_norm=False, 
                 din_use_softmax=False,
                 accumulation_steps=1,
                 cross_layers=6,
                 cross_dnn_hidden_units=[512, 128, 64],  # 替代 dnn_hidden_units
                 cross_dnn_activations="ReLU",
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(DIN_DCN, self).__init__(feature_map,
                                        model_id=model_id, 
                                        gpu=gpu, 
                                        embedding_regularizer=embedding_regularizer, 
                                        net_regularizer=net_regularizer,
                                        **kwargs)

        if isinstance(dnn_activations, str) and dnn_activations.lower() == "dice":
            dnn_activations = [Dice(units) for units in dnn_hidden_units]

        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.item_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim) 

        self.accumulation_steps = accumulation_steps
        
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.attention_layers = DIN_Attention(
            self.item_info_dim,
            attention_units=attention_hidden_units,
            hidden_activations=attention_hidden_activations,
            output_activation=attention_output_activation,
            dropout_rate=attention_dropout,
            use_softmax=din_use_softmax
        )
        self.attention_layers_multi = MultiHeadTargetAttention(
            input_dim = 256,
            num_heads = 4,
            dropout_rate=attention_dropout,
        )
        input_dim = feature_map.sum_emb_out_dim() + self.item_info_dim

        self.dcn_v2 = DCN_v2(
            input_dim=input_dim,
            cross_layers=cross_layers,
            model_structure="stacked",
            stacked_dnn_hidden_units=cross_dnn_hidden_units,
            hidden_activations=cross_dnn_activations,
            net_dropout=net_dropout,
            batch_norm=batch_norm,
            output_activation="Sigmoid"
        )
        
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        emb_list = []

        if batch_dict: 
            feature_emb = self.embedding_layer(batch_dict, flatten_emb=True)
            emb_list.append(feature_emb)

        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        batch_size = mask.shape[0]
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)
        target_emb = item_feat_emb[:, -1, :]
        sequence_emb = item_feat_emb[:, 0:-1, :]

        # pooling_emb = self.attention_layers(target_emb, sequence_emb, mask)
        pooling_emb = self.attention_layers_multi(target_emb, sequence_emb, mask)

        emb_list += [target_emb, pooling_emb]
        feature_emb = torch.cat(emb_list, dim=-1)

        y_pred = self.dcn_v2(feature_emb)
        y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def get_inputs(self, inputs, feature_source=None):
        batch_dict, item_dict, mask = inputs
        X_dict = dict()
        for feature, value in batch_dict.items():
            if feature in self.feature_map.labels:
                continue
            feature_spec = self.feature_map.features[feature]
            if feature_spec["type"] == "meta":
                continue
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            X_dict[feature] = value.to(self.device)
        for item, value in item_dict.items():
            item_dict[item] = value.to(self.device)
        return X_dict, item_dict, mask.to(self.device)

    def get_labels(self, inputs):
        labels = self.feature_map.labels
        batch_dict = inputs[0]
        y = batch_dict[labels[0]].to(self.device)
        return y.float().view(-1, 1)

    def get_group_id(self, inputs):
        return inputs[0][self.feature_map.group_id]

    def train_step(self, batch_data):
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        loss = loss / self.accumulation_steps
        loss.backward()
        if (self._batch_index + 1) % self.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

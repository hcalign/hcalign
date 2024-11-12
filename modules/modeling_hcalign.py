from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging


import torch
from torch import nn
import torch.nn.functional as F

from modules.until_module import PreTrainedModel, AllGather, CrossEn, MILNCELoss, MILNCELoss_BoF, KLdivergence
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip import CLIP, convert_weights
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from .guided_attention_pooling import GuidedAttentionPool
from .prototype_generator import PrototypeGenerator
from modules.module_clip import CLIP, convert_weights


logger = logging.getLogger(__name__)
allgather = AllGather.apply

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config=None, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        # import pdb; pdb.set_trace()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        # import pdb; pdb.set_trace()

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs) # -----------
        

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        if model.sim_header == 'tightTransf':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue

        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


class HcAlign(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super().__init__(cross_config, )
        self.task_config = task_config  

        # self._stage_one = True
        # self._stage_two = False
        # show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        # self.loose_type = False
        # if self._stage_one and check_attr('loose_type', self.task_config):
        #     self.loose_type = True
        #     show_log(task_config, "Test retrieval by loose type.")

        self.loose_type = True

        self.max_words = task_config.max_words
        self.max_frames = task_config.max_frames


        self._init_clip_model(cross_config, clip_state_dict, task_config)
        self.build_temproal_model(task_config, cross_config)
        self.build_bifurcate_tokenwise_module(task_config)
        self.use_original_clip_for_frame_features = True    
        self.loss_fct = CrossEn()
        self.build_maximize_semantic_iou_task(task_config)
        
    def _init_clip_model(self, cross_config, clip_state_dict, task_config):
        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        self.context_length = context_length
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.embed_dim = embed_dim

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

    def build_temproal_model(self, task_config, cross_config):
        context_length = self.context_length
        transformer_width = self.transformer_width
        transformer_heads = self.transformer_heads
        embed_dim = self.embed_dim

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
        if self.loose_type is False:
            # Cross Encoder ===>
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)

    def build_bifurcate_tokenwise_module(self, config):
        self.btm_mode = config.btm_mode
        self.guided_video_pool = GuidedAttentionPool(embed_dim=self.embed_dim)
        self.guided_text_pool = GuidedAttentionPool(embed_dim=self.embed_dim)

    def build_maximize_semantic_iou_task(self, config):
        self.msl_weight = config.cc_msl_weight  
        self.cc_neg_mine_ratio = config.cc_neg_mine_ratio 
        if self.msl_weight <= 0:
            logger.info("skip maximize semantic IoU loss task.")    
            return
        self.prototype_generator = PrototypeGenerator(num_prototypes=config.cc_num_prototypes, 
                                                        embed_dim=self.embed_dim,
                                                        selection_thresh=config.cc_selection_thresh,
                                                        cluster_iteration=config.cc_cluster_iteration,)
                                                                                    
        self.proj_visual = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, config.cc_num_prototypes),
        )

        self.proj_text = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, config.cc_num_prototypes),

        )
        self.proj_text.apply(self._init_module_weights)
        self.proj_visual.apply(self._init_module_weights)
        
    def _init_module_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        elif isinstance(module, nn.BatchNorm1d):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
    
    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None,
                       start_siou=False, mode='train_val', dataloader=None, update=False):
        assert mode in ['train_val', 'extract_features', 'gen_prototype']
        if mode == 'gen_prototype':
            self.generate_prototype(dataloader, update=update)
            return None
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        # [bs, 1, dim], [bs, num_words, dim], [bs, num_frames, dim]
        (sequence_output, seq_features), (visual_output, visual_spatial_output) = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask, 
                                                                video, video_mask, shaped=True, video_frame=video_frame, return_hidden=True)
        
        if mode == 'extract_features':
            outputs = self.get_features(sequence_output, seq_features, visual_output, attention_mask, video_mask)
            video_feats, frame_feats, sentence_feats, word_feats, logit_scale, video_global_feature, video_frame_feature = outputs
            return sentence_feats, word_feats, video_feats, frame_feats 
        
        if self.training:
            loss = 0.
            sim_matrix,  outputs = self.get_similarity_logits(sequence_output, seq_features, visual_output, attention_mask, 
                                        video_mask, shaped=True, loose_type=self.loose_type)
            sim_loss1 = self.loss_fct(sim_matrix)
            sim_loss2 = self.loss_fct(sim_matrix.T)
            sim_loss = (sim_loss1 + sim_loss2) / 2
            loss += sim_loss
            loss_msg = {"sim_loss": sim_loss.detach().item(), }

            if start_siou and self.msl_weight > 0:
                msl_loss = self.get_semantic_iou_loss(outputs)
                loss += msl_loss * self.msl_weight
                loss_msg['siou_loss'] = msl_loss.detach().item()
                
            loss_msg['loss'] = loss.detach().item()
            return loss, loss_msg
        else:
            return None
        
    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden, seq_features = self.clip.encode_text(input_ids, return_hidden=True)
        sequence_hidden, seq_features = sequence_hidden.float(), seq_features.float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden, seq_features

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1, return_hidden=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        visual_hidden = self.clip.encode_image(video, video_frame=video_frame, return_hidden=return_hidden)
        if return_hidden:
            visual_hidden, visual_spatial = visual_hidden
            visual_hidden = visual_hidden.float()
            video_spatial_feat = visual_spatial.float()
            visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))
            video_spatial_feat = video_spatial_feat.float().view((bs_pair, -1)+ video_spatial_feat.shape[1:])
            return visual_hidden, video_spatial_feat
        else:
            visual_hidden = visual_hidden.float() 
            visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))
            return visual_hidden

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1, return_hidden=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output, seq_features = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True) # [bs, 1, dim], [bs, num_words, dim]
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame, return_hidden=return_hidden)                  # [bs, num_frames, dim]

        return (sequence_output, seq_features), visual_output

    def get_similarity_logits(self, sequence_output, seq_features, visual_output, text_mask, video_mask, shaped=False, loose_type=True):
        
        if shaped is False:
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        assert self.sim_header in ["meanP", "seqLSTM", "seqTransf"]
        assert self.loose_type is True
        retrieve_logits, outputs = self._get_btm_similarity(sequence_output, seq_features, visual_output, text_mask, video_mask,
                                                            return_features=True)


        return retrieve_logits, outputs


    def _get_btm_similarity(self, sequence_output, seq_features, visual_output, text_mask, video_mask, 
                             return_features=False, ):
        """
            sequence_output: CLS token of text       # [bs, 1, dim]
            seq_features: all tokens of text         # [bs, num_words, dim]
            visual_output: all frames of video       # [bs, num_frames, dim]
        """
        outputs = self.get_features(sequence_output, seq_features, visual_output, text_mask, video_mask, sim_header=self.sim_header)
        video_feats, frame_feats, sentence_feats, word_feats, logit_scale, video_pre_feature, frame_pre_feature = outputs
        self.tau = 100
 
        if self.training:
            text_mask = allgather(text_mask, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
        else:
            text_mask = text_mask
            video_mask = video_mask

        fill = torch.HalfTensor([float("-inf")]).type_as(word_feats)

        # query-specific video feature
        guided_video_feat = self.guided_video_pool(sentence_feats, frame_feats,) # (bs_video, bs_text, dim)
        # query-specific text feature
        guided_sentence_feat = self.guided_text_pool(video_feats, word_feats)  # (bs_text, bs_video, dim)

        # normliaze the features and transpose dimension to get (bs_text, bs_video, dim)
        guided_video_feat = guided_video_feat / guided_video_feat.norm(dim=-1, keepdim=True)
        guided_sentence_feat = guided_sentence_feat / guided_sentence_feat.norm(dim=-1, keepdim=True)
        guided_video_feat = guided_video_feat.permute(1, 0, 2)  # (bs_text, bs_video, dim)

        # aggregate the word-level features to get query-specific word-to-video similarity
        t2v_logits = torch.einsum('amd,abd->abm', [word_feats, guided_video_feat]) # (bs_text, bs_video, num_words)
        t2v_logits = torch.einsum('abm,am->abm', [t2v_logits, text_mask]) 
        text_weights = torch.softmax(torch.where(t2v_logits==0, fill, t2v_logits)*self.tau, dim=-1)  # abt  
        t2v_logits = torch.sum(t2v_logits * text_weights, dim=2)  # ab
        A_qs = t2v_logits

        # aggregate the frame-level features to get query-specific frame-to-sentence similarity
        v2t_logits = torch.einsum('bnd,abd->abn', [frame_feats, guided_sentence_feat]) # (bs_video, bs_text, num_frames)
        v2t_logits = torch.einsum('abn,bn->abn', [v2t_logits, video_mask])
        video_weights = torch.softmax(torch.where(v2t_logits==0, fill, v2t_logits)*self.tau, dim=-1)  # abt
        v2t_logits = torch.sum(v2t_logits * video_weights, dim=2)
        B_qs = v2t_logits
        retrieve_logits_dual = (t2v_logits + v2t_logits) / 2.0

        # aggregate the word-level features to get query-free word-to-video similarity
        t2v_logits = torch.einsum('amd,bd->abm', [word_feats, video_feats]) # (bs_text, bs_video, num_words)
        t2v_logits = torch.einsum('abm,am->abm', [t2v_logits, text_mask]) 
        text_weights = torch.softmax(torch.where(t2v_logits==0, fill, t2v_logits)*self.tau, dim=-1)  # abt  
        t2v_logits = torch.sum(t2v_logits * text_weights, dim=2)  # ab
        A_qf = t2v_logits

        # aggregate the frame-level features to get query-free frame-to-sentence similarity
        v2t_logits = torch.einsum('bnd,ad->abn', [frame_feats, sentence_feats]) # (bs_video, bs_text, num_frames)
        v2t_logits = torch.einsum('abn,bn->abn', [v2t_logits, video_mask])
        video_weights = torch.softmax(torch.where(v2t_logits==0, fill, v2t_logits)*self.tau, dim=-1)  # abt
        v2t_logits = torch.sum(v2t_logits * video_weights, dim=2)
        B_qf = v2t_logits

        # for reproduce the Table 6 results in the paper, we use the following code
        if self.btm_mode == 'all':
            logits = (A_qs + B_qs + A_qf + B_qf) / 4.0
        elif self.btm_mode == 'query_free':
            logits = (A_qf + B_qf) / 2.0
        elif self.btm_mode == 'query_spec':
            logits = (A_qs + B_qs) /2.0
        elif self.btm_mode == 'word-video':
            logits = (A_qf + A_qs) / 2.0
        elif self.btm_mode == 'frame-sentence':
            logits = (B_qf + B_qs) / 2.0
        else:
            raise NotImplementedError
        
        if self.training:  
            logits = logits * logit_scale
        
        if return_features:
            return logits, outputs
        else:
            return logits

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)


    def generate_prototype(self, data_loader, update=False):
        # run generate process every epoch
        self.prototype_generator.generate_prototype(data_loader, self, update=update)

    def get_features(self, sequence_output, seq_features, visual_output, attention_mask, video_mask, sim_header="meanP",):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        if sim_header == "meanP":
            # Default: Parameter-free type
            visual_output_original = visual_output
            pass
        elif sim_header == "seqLSTM":
            # Sequential type: LSTM
            visual_output_original = visual_output
            visual_output = pack_padded_sequence(visual_output, torch.sum(video_mask, dim=-1).cpu(),
                                                 batch_first=True, enforce_sorted=False)
            visual_output, _ = self.lstm_visual(visual_output)
            if self.training: self.lstm_visual.flatten_parameters()
            visual_output, _ = pad_packed_sequence(visual_output, batch_first=True)
            visual_output = torch.cat((visual_output, visual_output_original[:, visual_output.size(1):, ...].contiguous()), dim=1)
            visual_output = visual_output + visual_output_original
        elif sim_header == "seqTransf":
            # Sequential type: Transformer Encoder
            visual_output_original = visual_output
            seq_length = visual_output.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            visual_output = visual_output + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
            visual_output = visual_output + visual_output_original

        # video-level visual feature 
        
        video_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        video_output = self._mean_pooling_for_similarity_visual(video_output, video_mask)
        video_output = video_output / video_output.norm(dim=-1, keepdim=True)   
        video_global_feature = video_output                 # [bs, dim]

        # frame-level visual features       
        if self.use_original_clip_for_frame_features:
            video_frame_feature = visual_output_original                 # [bs, num_frames, dim]
            frame_features = visual_output_original / visual_output_original.norm(dim=-1, keepdim=True)                # [bs, num_frames, dim]
        else:
            video_frame_feature = video_output                 # [bs, num_frames, dim]
            frame_features = visual_output  / visual_output.norm(dim=-1, keepdim=True)                                  # [bs, num_frames, dim]

        # sentence-level textual feature
        sentence_output = sequence_output.squeeze(1)
        sentence_output  = sentence_output / sentence_output.norm(dim=-1, keepdim=True)          # [bs, dim]
        
        # word-level textual features
        word_features = seq_features / seq_features.norm(dim=-1, keepdim=True)                   # [bs, num_words, dim]

        logit_scale = self.clip.logit_scale.exp()

        if self.training:
            video_output = allgather(video_output, self.task_config)
            frame_features = allgather(frame_features, self.task_config)
            sentence_output = allgather(sentence_output, self.task_config)
            word_features = allgather(word_features, self.task_config)
            torch.distributed.barrier()
        return video_output, frame_features, sentence_output, word_features, logit_scale, video_global_feature, video_frame_feature

    def get_semantic_iou_loss(self, features_list):
        """
            sentence_feats: (a, d)
            word_feats: (a, m, d)
            video_feats: (b, d)
            frame_feats: (b, n, d)
        """
        # the features have been normalized
        video_feats, frame_feats, sentence_feats, word_feats, logit_scale, video_global_feature, video_frame_feature = features_list
        
        visual_proj_vecs = self.proj_visual(video_feats)
        text_proj_vecs = self.proj_text(sentence_feats)
        visual_cross_gt = self.prototype_generator.assign_labels(frame_feats, is_norm=True)
        text_cross_gt = self.prototype_generator.assign_labels(word_feats, is_norm=True) 
        #text_cross_gt = self.prototype_generator.assign_labels(sentence_feats, is_norm=True) 
        
        visual_proj_vecs = F.sigmoid(visual_proj_vecs)
        text_proj_vecs = F.sigmoid(text_proj_vecs)

        # balance the positive and negative samples
        visual_proj_vecs, text_cross_gt = self.filter_neg_random(visual_proj_vecs, text_cross_gt,  ratio=self.cc_neg_mine_ratio)
        text_proj_vecs, visual_cross_gt = self.filter_neg_random(text_proj_vecs, visual_cross_gt, ratio=self.cc_neg_mine_ratio)
        
        visual_cross_loss = self.compute_dice_loss(visual_proj_vecs, text_cross_gt)
        text_cross_loss = self.compute_dice_loss(text_proj_vecs, visual_cross_gt)
        siou_loss = (visual_cross_loss + text_cross_loss) / 2
        return siou_loss

    def filter_neg_random(self, preds, gts, ratio=3):
        bs = preds.size(0)
        preds_list = []
        gts_list = []

        for i in range(bs):
            pred = preds[i].reshape(-1)
            gt = gts[i].reshape(-1)
            pos_index = torch.argwhere(gt.long() == 1)
            neg_index = torch.argwhere(gt.long() == 0)
            rand_index = torch.randperm(len(neg_index))[:len(pos_index)*ratio]
            selected_index = torch.cat([pos_index, neg_index[rand_index]])
            selected_index = selected_index.reshape(-1)
            preds_list.append(pred[selected_index])
            gts_list.append(gt[selected_index])
        
        preds_new = torch.cat(preds_list, dim=0)
        gts_new = torch.cat(gts_list, dim=0)
        return preds_new, gts_new

    def filter_neg_random22(self, pred, gt, ratio=3):
        gt = gt.reshape(-1)
        pred = pred.reshape(-1)
        pos_index = torch.argwhere(gt.long() == 1)
        neg_index = torch.argwhere(gt.long() == 0)
        rand_index = torch.randperm(len(neg_index))[:len(pos_index)*ratio]
        selected_index = torch.cat([pos_index, neg_index[rand_index]])
        selected_index = selected_index.reshape(-1)
        return pred[selected_index], gt[selected_index]
    
    def compute_dice_loss(self, pred, gt, return_mean=True):
        eps = 1
        intersection = torch.sum(pred * gt, dim=-1)
        union = torch.sum(pred, dim=-1) + torch.sum(gt, dim=-1)
        loss = 1. - (2 * intersection + eps) / (union + eps)
        if return_mean:
            loss = loss.mean()
        return loss


"""
    This code is referred from https://github.com/facebookresearch/swav
"""


from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F



from modules.until_module import AllGather
allgather = AllGather.apply


import logging
logger = logging.getLogger(__name__)


class PrototypeGenerator(nn.Module):
    def __init__(self, num_prototypes=1000, 
                        embed_dim=1024,
                        selection_thresh=0.5,
                        cluster_iteration=10,) -> None:
        super(PrototypeGenerator, self).__init__()
        self.prototype = nn.Linear(embed_dim, num_prototypes, bias=False)
        nn.init.normal_(self.prototype.weight, std=0.01)
        self.selection_thresh = selection_thresh
        self.cluster_iteration = cluster_iteration
        self.embed_dim = embed_dim
        self.nmb_prototypes = [num_prototypes]
        self.is_prototype_ready = False

        for param in self.prototype.parameters():
            param.requires_grad = False
        for mod in self.prototype.modules():
            mod.eval()

        self.label_generator = MaxminGenerator( thresh=selection_thresh)

    def init_memory(self, dataloader, model):
        feat_dim = self.embed_dim
        local_memory_embeddings = []
        model.eval()
        with torch.no_grad():
            if dist.get_rank() == 0:
                pbar = tqdm(total=len(dataloader))

            # use multiple captions for each video
            for idx, batch in enumerate(dataloader):
                batch = tuple(t.cuda() for t in batch)
                # text_ids: (bs_text, n_sentence, n_words), text_mask: (bs_text, n_sentence, n_words)
                # video: (bs_video, n_frame, h, w, c), video_mask: (bs_video, n_frame)
                text_ids, text_mask, segment_ids, video, video_mask = batch
                
                # extract text feats and video feats
                sentence_feats, word_feats, video_feats, frame_feats = model(text_ids, segment_ids, text_mask, video, video_mask, 
                                                                             mode='extract_features')
                
                # recommend to use frame_feats and sentence_feats
                # the word_feats is not sutible because it contains too many unmeaningful words(e.g, of, the;)
                # the video_feats is not sutible because it is too compact to represent the video
                local_memory_embeddings.append(frame_feats.reshape(-1, feat_dim))
                local_memory_embeddings.append(sentence_feats.reshape(-1, feat_dim))
                
                if dist.get_rank() == 0: pbar.update(1)

        logger.info('Initializion of the memory banks done.')
        local_memory_embeddings = torch.cat(local_memory_embeddings, dim=0)
        return local_memory_embeddings

    @torch.no_grad()
    def generate_prototype(self, data_loader, model, update=False, momentum=0.5):
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # initialize memory
        local_memory_embeddings = self.init_memory(data_loader, model)  # (dsize, dim)
        local_memory_embeddings = allgather(local_memory_embeddings, model.task_config)
        dist.barrier()

        # do cluster
        logger.info('Start to do clustering...')
        logger.info(f"local_memory_embeddings: {local_memory_embeddings.shape}, {local_memory_embeddings.max()}")
        with torch.no_grad():
            for i_K, K in enumerate(self.nmb_prototypes):
                # run distributed k-means
                # init centroids with elements from memory bank of rank 0
                centroids = torch.empty(K, self.embed_dim).cuda(non_blocking=True)  # (k, dim)
                if local_rank == 0:
                    if self.is_prototype_ready:
                        centroids = self.prototype.weight
                    else:
                        # random select K points
                        random_idx = torch.randperm(len(local_memory_embeddings))[:K]
                        assert len(random_idx) >= K, "please reduce the number of centroids"
                        centroids = local_memory_embeddings[random_idx]
                dist.barrier()
                dist.broadcast(centroids, 0)
                dist.barrier()

                for n_iter in tqdm(range(self.cluster_iteration+1)):
                    # E step
                    if local_rank<=1: logger.info(f"[rank={local_rank}] centroids: {centroids.max()}")

                    bs = 128
                    local_assignments = []
                    for i in range(0, len(local_memory_embeddings), bs):
                        cur_local_embeddings = local_memory_embeddings[i:i+bs]
                        cur_dot_products = torch.mm(cur_local_embeddings, centroids.t())   # (dsize, k)
                        _, cur_local_assignments = cur_dot_products.max(dim=1)
                        local_assignments.append(cur_local_assignments)
                    local_assignments = torch.cat(local_assignments, dim=0)
                    # dot_products = torch.mm(local_memory_embeddings, centroids.t())   # (dsize, k)
                    # _, local_assignments = dot_products.max(dim=1)

                    # finish
                    if n_iter == self.cluster_iteration:
                        break

                    # M step, update centroids
                    where_helper = self.get_indices_sparse(local_assignments.cpu().numpy())
                    counts = torch.zeros(K).cuda(non_blocking=True).int()
                    emb_sums = torch.zeros(K, self.embed_dim).cuda(non_blocking=True)
                    for k in range(len(where_helper)):
                        if len(where_helper[k][0]) > 0:
                            emb_sums[k] += torch.sum(
                                local_memory_embeddings[where_helper[k][0]],
                                dim=0,
                            )
                            counts[k] += len(where_helper[k][0])
                    dist.all_reduce(counts)
                    mask = counts > 0
                    dist.all_reduce(emb_sums)
                    centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)+1e-5
                    centroids = centroids / centroids.norm(dim=-1, keepdim=True)

                # update prototype
                if len(self.nmb_prototypes) == 1:
                    if update:
                        weight_old = self.prototype.weight
                        weight_new = (1 - momentum) * centroids + momentum * weight_old
                    else:
                        weight_new = centroids
                    self.prototype.weight.copy_(weight_new)
                else:
                    raise NotImplementedError("multi-prototype is not supported yet")
        logger.info('Clustering done.')
        self.is_prototype_ready = True

    def get_indices_sparse(self, data):
        cols = np.arange(data.size)
        M = csr_matrix((cols, (data.ravel(), cols)), shape=(int(data.max()) + 1, data.size))
        return [np.unravel_index(row.data, data.shape) for row in M]
    
    @torch.no_grad()
    def assign_labels(self, features, is_norm=True):
        #assert self.is_prototype_ready, "prototype is not ready"
        if not is_norm:
            features = nn.functional.normalize(features, dim=-1, p=2)
        assign_labels = self.label_generator(features, self.prototype.weight, )
        return assign_labels
    
        
class MaxminGenerator(nn.Module):
    def __init__(self, thresh=0.9) -> None:
        super().__init__()
        self.thresh = thresh

    def forward(self, features, prototype, ):
        dim = features.shape[-1]
        if len(features.shape) == 2:
            scores = torch.mm(features, prototype.t()) / dim  # (bs, k)            
        else: #3
            scores = torch.einsum('bmd, kd->bmk', features, prototype) / dim
            scores_weight = F.softmax(scores, dim=1)
            scores = torch.einsum('bmk, bmk->bk', scores, scores_weight)
        scores = self._max_min_norm(scores)
        labels = (scores > self.thresh).float()
        return labels
    
    def _max_min_norm(self, x):
        x_max = x.max(dim=-1, keepdim=True)[0]
        x_min = x.min(dim=-1, keepdim=True)[0]
        x = (x - x_min) / (x_max - x_min)
        return x


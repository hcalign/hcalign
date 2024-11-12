from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
import sys
sys.path.append('..')
# from dataloaders.rawvideo_util import RawVideoExtractor
from dataloaders.rawvideo_util_tvr import RawVideoExtractor
from decord import VideoReader, cpu


import torch
from PIL import Image
from decord import VideoReader, cpu
from dataloaders.img_transform import get_transform


class WEBVID_DataLoader(Dataset):
    """WEBVID dataset loader."""
    def __init__(
            self,
            subset,
            data_path,
            features_path,
            ann_file,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            strategy=1
            
    ):
        self.subset = subset
        assert self.subset in ["train", "val", "test"]
        assert self.subset in ["train", ]


        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.image_resolution = image_resolution
        self.video_framerate = feature_framerate
        self.ann_file = ann_file
        is_training = subset == "train"
        self.transform = get_transform(image_resolution, is_training)
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]

        self.slice_framepos = slice_framepos
        self.strategy = strategy
        assert self.slice_framepos in [0, 1, 2]



        ann_file = os.path.join(self.data_path, ann_file)
        
        data = json.load(open(ann_file, 'r'))
        video_ids = list(data.keys())


        # video_dict = {}
        # for root, dub_dir, video_files in os.walk(self.features_path):

        #     for video_file in dub_dir: # frames----------
        #         video_id_ = video_file # frames----------


        #         if video_id_ not in video_ids:
        #             continue
        #         file_path_ = os.path.join(root, video_file)
        #         video_dict[video_id_] = file_path_
        video_dict = {}
        video_files = os.listdir(self.features_path)

        for video_id_ in video_ids:
            file_path_ = os.path.join(self.features_path, video_id_)
            if not os.path.exists(file_path_):
                continue
            video_dict[video_id_] = file_path_

        self.video_dict = video_dict

        # 去除不存在的video
        print(f"video_ids: from {len(video_ids)} to {len(video_dict)}")
        video_ids = list(video_dict.keys())

        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []
        for video_id in video_ids:
            for cap_txt in data[video_id]:
                self.sentences_dict[len(self.sentences_dict)] = (video_id, cap_txt)
            self.cut_off_points.append(len(self.sentences_dict))

        ## below variables are used to multi-sentences retrieval
        # self.cut_off_points: used to tag the label when calculate the metric
        # self.sentence_num: used to cut the sentence representation
        # self.video_num: used to cut the video representation
        self.multi_sentence_per_video = False    # !!! important tag for eval

        print("Video number: {}".format(len(self.video_dict)))
        print("Total Paire: {}".format(len(self.sentences_dict)))
        
        self.sample_len = len(self.sentences_dict)


        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution,)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    
    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.int32)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int32)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int32)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(caption)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
        return pairs_text, pairs_mask, pairs_segment


    def _get_rawvideo(self, choice_video_ids):

        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int32)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float32)

        for i, video_id in enumerate(choice_video_ids):
            video_path = self.video_dict[video_id]

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']

            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def _get_rawvideo_dec(self, video_path, s=None, e=None):
        # speed up video decode via decord.
        video_mask = np.zeros(self.max_frames, dtype=np.int32)
        max_video_length = 0

        # T x 3 x H x W
        video = np.zeros((self.max_frames, 3, self.image_resolution, self.image_resolution), dtype=np.float32)

        if s is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = start_time if start_time >= 0. else 0.
            end_time = end_time if end_time >= 0. else 0.
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = start_time + 1
    

        if os.path.exists(video_path):
            vreader = VideoReader(video_path, ctx=cpu(0))
        else:
            print(video_path)
            raise FileNotFoundError

        fps = vreader.get_avg_fps()
        f_start = 0 if start_time is None else int(start_time * fps)
        f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
        num_frames = f_end - f_start + 1
        if num_frames > 0:
            # T x 3 x H x W
            sample_fps = int(self.video_framerate)
            t_stride = int(round(float(fps) / sample_fps))

            all_pos = list(range(f_start, f_end + 1, t_stride))
            if len(all_pos) > self.max_frames:
                sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=self.max_frames, dtype=int)]
            else:
                sample_pos = all_pos

            patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
            patch_images = torch.stack([self.transform(img) for img in patch_images])
            slice_len = patch_images.shape[0]
            max_video_length = max_video_length if max_video_length > slice_len else slice_len
            if slice_len < 1:
                pass
            else:
                video[:slice_len, ...] = patch_images
        else:
            print("video path: {} error. ".format(video_path, ))

        video_mask[:max_video_length] = [1] * max_video_length

        return video, video_mask

   
    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]
        video_path = self.video_dict[video_id]
        # pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        pairs_text, pairs_mask, pairs_segment = self._get_text(video_id, caption)

        video, video_mask = self._get_rawvideo_dec(video_path, None, None)
            
        video = torch.from_numpy(video).unsqueeze(1).unsqueeze(0)
        video_mask = torch.from_numpy(video_mask).unsqueeze(0)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask
        



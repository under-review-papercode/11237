import os
import torch
from torch import Tensor
from typing import Any, List, Optional, Sequence, Union
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob
import pandas as pd
import numpy as np
import string
from utils import svg2paths2, disvg, raster, get_single_paths, get_similar_length_paths, check_for_continouity, get_rasterized_segments, all_paths_to_max_diff, Path, svg_string_to_tensor, drawing_to_tensor
import copy
import random
import math
from tokenizer import VQTokenizer


class VQDataset(Dataset):
    """
    main input here is the csv_path to a split.csv
    new addition: the csv needs a column called "index_in_numpy_array", which points each sample to an index in the vq_token numpy array

    - fraction_of_class_only_inputs: float (default 0.2), fraction of samples where the text input is only the "class" entry of the dataframe
    - fraction_of_blank_inputs: float (default 0.1), fraction of samples where the text input is empty
    - use_given_text_tokens_only: bool (default False), if True, the text input will always be the already tokenized file
    - shuffle_vq_order: bool (default False), if True, `<SOS>, <CLS>, t_1, ..., t_n, <SEP>, <BOS>, v_1, p_1, v_2, p_2, ... v_m, p_m, <EOS>` will become `<SOS>, <CLS>, t_1, ..., t_n, <SEP>, <BOS>, v_i, p_i, v_i+1, p_i+1, ..., v_m, p_m, v_1, p_1, ..., v_i-1, p_i-1, <EOS>` for random index i
    its not really "shuffling", but more cutting the sequence into two parts and switching their order
    """
    def __init__(self,
                 csv_path:str,
                 vq_token_npy_path:str,
                 tokenizer: VQTokenizer,
                 context_length: int,
                 dataset:str,
                 min_context_length: int = 10,
                fraction_of_strokenuwa_inputs:float= 0.0,
                fraction_of_class_only_inputs:float= 0.9,
                fraction_of_blank_inputs:float= 0.1,
                fraction_of_iconshop_chatgpt_inputs:float= 0.0,
                 shuffle_vq_order:bool=True,
                 use_pre_computed_text_tokens_only: bool=False,
                 train:bool = True,
                 subset:str=None,):
        super(VQDataset, self).__init__()

        self.split = pd.read_csv(csv_path)
        self.train = train
        self.subset = subset
        self.context_length = context_length
        self.min_context_length = min_context_length

        sum_of_fractions = fraction_of_class_only_inputs + fraction_of_blank_inputs + fraction_of_strokenuwa_inputs + fraction_of_iconshop_chatgpt_inputs if train else 0.0
        if subset is not None:
            assert subset in self.split["class"].unique(), f"Subset {subset} not found in the dataset."
            self.split = self.split[self.split["class"] == subset].reset_index(drop=True)
        assert dataset in ["figr8", "fonts"], f"Dataset must be either 'figr8' or 'fonts', got {dataset}."
        assert sum_of_fractions <= 1, f"All fractions must be less or equal to 1, got {sum_of_fractions}."



        self.fraction_of_class_only_inputs = fraction_of_class_only_inputs if train else 0.0
        self.fraction_of_blank_inputs = fraction_of_blank_inputs if train else 0.0
        self.fraction_of_strokenuwa_inputs = fraction_of_strokenuwa_inputs if train else 0.0
        self.fraction_of_iconshop_chatgpt_inputs = fraction_of_iconshop_chatgpt_inputs if train else 0.0
        self.dataset = dataset

        if sum_of_fractions < 1:
            self.fraction_of_full_description_inputs = 1 - sum_of_fractions
        else:
            self.fraction_of_full_description_inputs = 0.

        if not train or train is None:
            self.fraction_of_full_description_inputs = 1.0

        if dataset == "figr8":
            assert "class" in self.split.columns if self.fraction_of_class_only_inputs > 0 else True, "Column 'class' is required for figr8 dataset."
            assert "strokenuwa_prompt" in self.split.columns if self.fraction_of_strokenuwa_inputs > 0 else True, "Column 'strokenuwa_prompt' is required for figr8 dataset."
            assert "iconshop_sentence_prompt" in self.split.columns if self.fraction_of_iconshop_chatgpt_inputs > 0 else True, "Column 'iconshop_sentence_prompt' is required for figr8 dataset."
            assert "description" in self.split.columns if self.fraction_of_full_description_inputs > 0 else True, "Column 'description' is required for figr8 dataset."

        self.use_pre_computed_text_tokens_only = use_pre_computed_text_tokens_only
        self.shuffle_vq_order = shuffle_vq_order

        self.tokenizer = tokenizer
        self.tokenizer.use_text_encoder_only = True

        self.bert_cls_token = self.tokenizer.text_tokenizer.get_vocab().get("[CLS]")
        self.bert_sep_token = self.tokenizer.text_tokenizer.get_vocab().get("[SEP]")
        self.bert_pad_token = self.tokenizer.text_tokenizer.get_vocab().get("[PAD]")
        self.sos_token = self.tokenizer.special_token_mapping.get("<SOS>")
        self.bos_token = self.tokenizer.special_token_mapping.get("<BOS>")
        self.eos_token = self.tokenizer.special_token_mapping.get("<EOS>")
        self.pad_token = self.tokenizer.special_token_mapping.get("<PAD>")

        # load pre-computed vq tokens
        self.vq_token_npy_path = vq_token_npy_path
        numpy_array = np.load(vq_token_npy_path)
        self.vq_numpy_array = np.split(numpy_array, np.where(numpy_array == self.bos_token)[0])[1:]

        if not len(self.vq_numpy_array) == len(self.split) and self.subset is None:
            print(f"[WARNING] Number of samples in the numpy array and the csv file do not match. Numpy array has {len(self.vq_numpy_array)} samples, csv has {len(self.split)} samples.")
        
        if train is None:
            self.split = self.split[self.split["split"] == "test"].reset_index(drop=True)
            self.split = self.split.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            if train:
                self.split = self.split[self.split["split"] == "train"].reset_index(drop=True)
            else:
                self.split = self.split[self.split["split"] == "val"].reset_index(drop=True)

        self.split["index_in_numpy_array"] = self.split["index_in_numpy_array"].astype(int)
        samples_before_filtering = len(self.split)

        self.split = self.split[self.split["text_token_length"] < 16]
        self.max_text_length = self.split["text_token_length"].max()
        # TODO add font blacklisting here
        self.split = self.split[self.split["vq_token_length"] + self.max_text_length + 2 <= self.context_length]
        self.split = self.split[self.split["vq_token_length"] >= self.min_context_length]

        samples_after_filtering = len(self.split)
        if samples_before_filtering > 0:
            print(f"[INFO] Filtered {samples_before_filtering - samples_after_filtering} samples because they were too long or too short. That is {np.round((samples_before_filtering - samples_after_filtering) / samples_before_filtering * 100, decimals=2)}% of the dataset.")
        else:
            print(f"[WARNING] No samples found for {'train' if train else 'test'} split.")

    def _get_padded_text_tokens(self, text_tokens: np.ndarray):
        padded_text = np.append(text_tokens, np.zeros(self.max_text_length - len(text_tokens), dtype=np.ushort) + self.bert_pad_token)
        return padded_text
    
    def _get_padded_vq_tokens(self, vq_tokens: np.ndarray):
        if vq_tokens[0] != self.bos_token:
            vq_tokens = np.concatenate([np.array([self.bos_token]), vq_tokens])
        vq_with_eos = np.append(vq_tokens, np.zeros(1, dtype=np.ushort) + self.eos_token)
        final_padded_vq = np.append(vq_with_eos, np.zeros(self.context_length - self.max_text_length - len(vq_with_eos) - 1, dtype=np.ushort) + self.pad_token)  # -1 because SOS token is prefixed to the sequence later
        return final_padded_vq

        # assert len(self.text_tokens) == len(self.vq_tokens), "Text and VQ tokens should have the same shape."
        # assert self.text_tokens[0,0] == bert_cls_token, "First token in text tokens should be the BERT CLS token."
        # assert self.vq_tokens[0,0] == bos_token, "First token in VQ tokens should be the BOS token."
        # assert self.text_attention_masks[0,0] == 1, "First token in text attention masks should be 1."

    def __len__(self):
        return len(self.split)
    
    def _get_tokenized_text(self, row):
        if self.dataset == "fonts":
            text_to_tokenize = np.random.choice([row["class"], 
                                                 row["description"], 
                                                 ""],
                                             p=[self.fraction_of_class_only_inputs, 
                                                self.fraction_of_full_description_inputs, 
                                                self.fraction_of_blank_inputs])
            if text_to_tokenize in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and len(text_to_tokenize) == 1:
                text_to_tokenize = f"capital {text_to_tokenize}"
            text_tokens = self.tokenizer.tokenize_text(text_to_tokenize)
            return text_tokens
        elif self.dataset == "figr8":
            text_to_tokenize = np.random.choice([row.get("class"), 
                                                 row.get("strokenuwa_prompt"), 
                                                 row.get("iconshop_sentence_prompt"), 
                                                 row.get("description"),
                                                 ""],
                                                p=[self.fraction_of_class_only_inputs, 
                                                   self.fraction_of_strokenuwa_inputs, 
                                                   self.fraction_of_iconshop_chatgpt_inputs, 
                                                   self.fraction_of_full_description_inputs,
                                                   self.fraction_of_blank_inputs])
            if text_to_tokenize is None:
                text_to_tokenize = "None"
            text_tokens = self.tokenizer.tokenize_text(text_to_tokenize)
            return text_tokens

    def __getitem__(self, idx:int):
        """
        IMPORTANT
        text tokens have their special tokens and padding already included.
        vq tokens have their special tokens (BOS and EOS) and padding already included.
        only SOS needs to be prefixed after the data is loaded.
        """
        row = self.split.iloc[idx]

        if self.use_pre_computed_text_tokens_only:
            text_tokens = np.load(row["text_token_path"])
        else:
            text_tokens = self._get_tokenized_text(row)
        text_tokens = self._get_padded_text_tokens(text_tokens)

        # vq_tokens = np.load(row["vq_token_path"])
        vq_tokens = self.vq_numpy_array[row["index_in_numpy_array"]]

        if self.shuffle_vq_order:
            try:
                i = np.random.randint(5, len(vq_tokens) - 5)
            except:
                # if something goes wrong, just take the middle of the min-sequence
                i = self.min_context_length//2
            if self.tokenizer._is_position(vq_tokens[i]):
                i -= 1  # position is guaranteed to be preceded by a patch in a single code setup
            vq_tokens = np.concatenate([np.array([self.bos_token]),vq_tokens[i:], vq_tokens[1:i]])
        vq_tokens = self._get_padded_vq_tokens(vq_tokens)
        text_attention_mask = (text_tokens != self.bert_pad_token).astype(np.int64)

        text_tokens = torch.from_numpy(text_tokens.astype(np.int32)).long()
        vq_tokens = torch.from_numpy(vq_tokens.astype(np.int32)).long()
        vq_targets = torch.roll(vq_tokens, -1)
        vq_targets[-1] = self.pad_token
        attention_mask = torch.from_numpy(text_attention_mask.astype(np.int32)).long()

        return text_tokens, attention_mask, vq_tokens, vq_targets, torch.ones(1).to(text_tokens.device)*self.pad_token

class VQDataModule(LightningDataModule):

    def __init__(
        self,
        csv_path: str,
        dataset:str,
        vq_token_npy_path: str,
        tokenizer: VQTokenizer,
        context_length: int,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size:int = 32,
        num_workers: int = 0,
        min_context_length: int = 10,
        fraction_of_class_only_inputs: float = 0.2,
        fraction_of_blank_inputs: float = 0.1,
        fraction_of_strokenuwa_inputs: float = 0.0,
        fraction_of_iconshop_chatgpt_inputs: float = 0.0,
        shuffle_vq_order:bool=False,
        use_pre_computed_text_tokens_only: bool=False,
        subset:str=None,
        **kwargs,
    ):
        super().__init__()

        self.csv_path = csv_path
        self.dataset = dataset
        self.vq_token_npy_path= vq_token_npy_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.min_context_length = min_context_length
        self.fraction_of_class_only_inputs = fraction_of_class_only_inputs
        self.fraction_of_blank_inputs = fraction_of_blank_inputs
        self.fraction_of_strokenuwa_inputs = fraction_of_strokenuwa_inputs
        self.fraction_of_iconshop_chatgpt_inputs = fraction_of_iconshop_chatgpt_inputs
        self.shuffle_vq_order = shuffle_vq_order
        self.use_pre_computed_text_tokens_only = use_pre_computed_text_tokens_only
        self.subset = subset


    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in ["train", "test", "val"]:
            stage = None

        if stage is None or stage == "train":
            self.train_dataset = VQDataset(
                self.csv_path,
                self.vq_token_npy_path,
                tokenizer=self.tokenizer,
                context_length=self.context_length,
                dataset=self.dataset,
                train=True,
                min_context_length=self.min_context_length,
                fraction_of_class_only_inputs = self.fraction_of_class_only_inputs,
                fraction_of_blank_inputs = self.fraction_of_blank_inputs,
                fraction_of_iconshop_chatgpt_inputs=self.fraction_of_iconshop_chatgpt_inputs,
                fraction_of_strokenuwa_inputs=self.fraction_of_strokenuwa_inputs,
                shuffle_vq_order = self.shuffle_vq_order,
                use_pre_computed_text_tokens_only = self.use_pre_computed_text_tokens_only,
                subset=self.subset,
            )

        if stage is None or stage == "val":
            self.val_dataset = VQDataset(
                self.csv_path,
                self.vq_token_npy_path,
                tokenizer=self.tokenizer,
                context_length=self.context_length,
                dataset=self.dataset,
                train=False,
                min_context_length=self.min_context_length,
                fraction_of_class_only_inputs = self.fraction_of_class_only_inputs,
                fraction_of_blank_inputs = self.fraction_of_blank_inputs,
                fraction_of_iconshop_chatgpt_inputs=self.fraction_of_iconshop_chatgpt_inputs,
                fraction_of_strokenuwa_inputs=self.fraction_of_strokenuwa_inputs,
                shuffle_vq_order = self.shuffle_vq_order,
                use_pre_computed_text_tokens_only = self.use_pre_computed_text_tokens_only,
                subset=self.subset,
            )
        if stage is None or stage == "test":
            self.test_dataset = VQDataset(
                self.csv_path,
                self.vq_token_npy_path,
                tokenizer=self.tokenizer,
                context_length=self.context_length,
                dataset=self.dataset,
                train=None,
                min_context_length=self.min_context_length,
                fraction_of_class_only_inputs = self.fraction_of_class_only_inputs,
                fraction_of_blank_inputs = self.fraction_of_blank_inputs,
                fraction_of_iconshop_chatgpt_inputs=self.fraction_of_iconshop_chatgpt_inputs,
                fraction_of_strokenuwa_inputs=self.fraction_of_strokenuwa_inputs,
                shuffle_vq_order = self.shuffle_vq_order,
                use_pre_computed_text_tokens_only = self.use_pre_computed_text_tokens_only,
                subset=self.subset,
            )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )
    
class GlyphazznStage1Dataset(Dataset):
    """
    Glyphazzn dataset that requires already normalized SVGs. Yields patches, positions, and labels. The label is the index of string.printable -> label = string.printable.index(label)

    Requires the following structure:
    top_level_dir (simplified svgs)
    |________________________________________
    |                   |                   |
    train               test                split.csv (with columns: file_path, class, split, description)
    |                   |
    0-9, a-z, A-Z       0-9, a-z, A-Z
    |                   |
    *.svg               *.svg

    Args:
        - top_level_dirs: paths to the top level directory of all simplified SVGs
        - channels: number of channels for the rasterized images
        - width: width/height of the rasterized images
        - train: whether to use the train or test split
        - subset:
        - individual_min_length: minimum length of a path segment to qualify for being a single shape layer
        - individual_max_length: maximum length of a path segment, everything longer than this will be cropped into multiple segments
        - stroke_width: stroke width for rasterization
        - max_shapes_per_svg: maximum number of shape layers per svg file, can be tuned for VRAM savings
    """

    def __init__(self,
                 image_root_dir: str,
                 csv_path: List[str],
                 channels: int,
                 width: int,
                 train: bool = True,
                 individual_min_length: float = 1.,
                 individual_max_length: float = 10.,
                 stroke_width: float = 0.3,
                 max_shapes_per_svg: int = 64,
                 use_single_paths:bool = False,
                 return_index = False,
                 subset_class:str=None,
                 **kwargs):
        super(GlyphazznStage1Dataset, self)
        print(f"[INFO] These keywords were provided in GlyphazznStage1Dataset but are not used: {kwargs.keys()}")
        self.csv_path = csv_path
        self.individual_min_length = individual_min_length
        self.individual_max_length = individual_max_length
        self.stroke_width = stroke_width
        self.max_shapes_per_svg = max_shapes_per_svg
        self.channels = channels
        self.width = width
        self.train = train
        self.use_single_paths = use_single_paths
        self.return_index = return_index
        self.subset_class = subset_class
        self.image_root_dir = image_root_dir
        print("[GlyphazznStage1Dataset] loading df...")

        self.df = pd.read_csv(csv_path)
        self.class2id = {id_name: class_name for class_name, id_name in enumerate(self.df["class"].unique())}
        if train is None:
            self.df = self.df[self.df["split"] == "test"].reset_index(drop=True)
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            if train:
                self.df = self.df[self.df["split"] == "train"].reset_index(drop=True)
            else:
                self.df = self.df[self.df["split"] == "val"].reset_index(drop=True)

        if subset_class is not None and subset_class in self.df["class"].unique():
            self.df = self.df[self.df["class"] == subset_class].reset_index(drop=True)

    def crop_path_into_segments(self, path:Path, length:float = 5.):
        """
        a single input path is cropped into segments of approx length `length`. I say "approx" because we divide the path into same length segments, which will not be exactly `length` long.
        """
        segments = []
        try:
            num_iters = math.ceil(path.length() / length)
            for i in range(num_iters):
                cropped_segment = path.cropped(i/num_iters, (i+1)/num_iters)
                segments.append(cropped_segment)
        except Exception as e:
            pass
        return segments

    def get_similar_length_paths(self, single_paths, max_length: float = 5., filter_min_length:bool = False):
        """
        splits all the paths into similar length segments if they're too long
        """
        similar_length_paths = []
        if filter_min_length:
            prev_len = len(single_paths)
            single_paths = [x for x in single_paths if x.length() >= self.individual_min_length]
            after_len = len(single_paths)
            if after_len >= 0.8 * len(prev_len):
                print("More than 80% of paths were removed because they were too short. This is likely an error.")
        for path in single_paths:
            if path.length() < self.individual_min_length:
                similar_length_paths.append(path)
                continue
            try:
                segments = self.crop_path_into_segments(path, length=max_length)
                similar_length_paths.extend(segments)
            except AssertionError:
                print("Error while cropping path into segments, skipping...")
                continue
        return similar_length_paths
    
    def get_similar_length_paths_from_index(self, index, max_length: float = 5.):
        svg_path = self.df.iloc[index].simplified_svg_file_path
        paths, attributes, svg_attributes = svg2paths2(svg_path)
        single_paths = get_single_paths(paths)
        sim_length_paths = self.get_similar_length_paths(single_paths, max_length=max_length)
        return sim_length_paths

    def __getitem__(self, index) -> tuple:
        svg_path = self.df.iloc[index]["simplified_svg_file_path"]
        svg_path = os.path.join(self.image_root_dir, svg_path)
        label = self.df.iloc[index]["class"]
        label = self.class2id[label]
        description = self.df.iloc[index]["description"]
        try:
            paths, attributes, svg_attributes = svg2paths2(svg_path)
        except Exception as e:
            print(f"[ERROR] Could not load {svg_path}. Exception: {e}")
            return torch.ones(2,3,128,128), torch.ones(2).int(), torch.ones(2,2), "EMPTY"
        if self.use_single_paths:
            single_paths = get_single_paths(paths)
            single_paths = self.get_similar_length_paths(single_paths, self.individual_max_length, filter_min_length=False)
        else:
            single_paths = self.get_similar_length_paths(paths, self.individual_max_length)
        
        assert check_for_continouity(single_paths), "paths are not continous"
        # select a random slice of the paths of length max_shapes_per_svg
        single_paths = [path for path in single_paths if path.length() > 0.]
        if len(single_paths) > self.max_shapes_per_svg:
            start_idx = random.randint(0, len(single_paths) - self.max_shapes_per_svg)
            single_paths = single_paths[start_idx:start_idx+self.max_shapes_per_svg]
        rasterized_segments, centers = get_rasterized_segments(single_paths, self.stroke_width, self.individual_max_length, svg_attributes, centered=True, height=self.width, width=self.width)
        imgs = torch.stack(rasterized_segments)  # (n_shapes, channels, width, width)
        centers = torch.tensor(centers)  # (n_shapes, 2)
        labels = torch.ones(imgs.size(0)) * label
        if self.return_index:
            return imgs, labels.int(), centers, description, index
        else:
            return imgs, labels.int(), centers, description
    
    def _get_full_item(self, index:int) -> List[Tensor]:
        """
        This function is intended to be used by the tokenization process.
        """
        svg_path = self.df.iloc[index]["simplified_svg_file_path"]
        label = self.df.iloc[index]["class"]
        label = self.class2id[label]
        description = self.df.iloc[index]["description"]

        paths, attributes, svg_attributes = svg2paths2(svg_path)
        if self.use_single_paths:
            single_paths = get_single_paths(paths, self.individual_max_length)
        else:
            single_paths = self.get_similar_length_paths(paths, self.individual_max_length)
        assert check_for_continouity(single_paths), "paths are not continous"
        single_paths = [path for path in single_paths if path.length() > 0.]
        rasterized_segments, centers = get_rasterized_segments(single_paths, self.stroke_width, self.individual_max_length, svg_attributes, centered=True, height=self.width, width=self.width)
        imgs = torch.stack(rasterized_segments)  # (n_shapes, channels, width, width)
        centers = torch.tensor(centers)  # (n_shapes, 2)
        labels = torch.ones(imgs.size(0)) * label
        return imgs, labels.int(), centers, description
    
    def _get_full_svg_drawing(self, index, width:int = 720, as_tensor:bool = False):
        svg_path = self.df.iloc[index].simplified_svg_file_path
        paths, attributes, svg_attributes = svg2paths2(svg_path)
        if self.use_single_paths:
            single_paths = get_single_paths(paths)
        else:
            single_paths = self.get_similar_length_paths(paths, self.individual_max_length)
        # single_paths = get_single_paths(paths)
        single_paths = [path for path in single_paths if path.length() > 0.]
        drawing = disvg(single_paths, paths2Drawing=True, stroke_widths=[self.stroke_width]*len(single_paths), viewbox = svg_attributes["viewBox"],dimensions=(width, width))
        if as_tensor:
            return svg_string_to_tensor(drawing.tostring())
        else:
            return drawing

    def __len__(self):
        return len(self.df)

class GlyphazznStage1Datamodule(LightningDataModule):
    def __init__(
        self,
        image_root_dir: str,
        csv_path: str,
        train_batch_size: int,
        val_batch_size: int,
        channels: int,
        width: int,
        test_batch_size:int = 32,
        individual_max_length: float = 10.,
        max_shapes_per_svg:int=64,
        num_workers: int = 0,
        stroke_width: float = 0.3,
        subset:str = None,
        use_single_paths:bool = False,
        return_index = False,
        **kwargs,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.csv_path = csv_path
        self.channels = channels
        self.width = width
        self.num_workers = num_workers
        self.stroke_width = stroke_width
        self.individual_max_length = individual_max_length
        self.subset = subset
        self.max_shapes_per_svg = max_shapes_per_svg
        self.use_single_paths = use_single_paths
        self.return_index = return_index
        self.image_root_dir = image_root_dir
    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in ["train", "test", "val"]:
            stage = None

        if stage is None or stage == "train":
            self.train_dataset = GlyphazznStage1Dataset(
                self.image_root_dir,
                self.csv_path,
                self.channels,
                self.width,
                train=True,
                individual_max_length=self.individual_max_length,
                stroke_width=self.stroke_width,
                max_shapes_per_svg=self.max_shapes_per_svg,
                use_single_paths=self.use_single_paths,
                return_index = self.return_index,
                subset_class=self.subset
            )

        if stage is None or stage == "val":
            self.val_dataset = GlyphazznStage1Dataset(
                self.image_root_dir,
                self.csv_path,
                self.channels,
                self.width,
                train=False,
                individual_max_length=self.individual_max_length,
                stroke_width=self.stroke_width,
                max_shapes_per_svg=self.max_shapes_per_svg,
                use_single_paths=self.use_single_paths,
                return_index = self.return_index,
                subset_class=self.subset
            )

        if stage is None or stage == "test":
            self.test_dataset = GlyphazznStage1Dataset(
                self.image_root_dir,
                self.csv_path,
                self.channels,
                self.width,
                train=None,
                individual_max_length=self.individual_max_length,
                stroke_width=self.stroke_width,
                max_shapes_per_svg=self.max_shapes_per_svg,
                use_single_paths=self.use_single_paths,
                return_index = self.return_index,
                subset_class=self.subset
            )

    #       ===============================================================

    def collate_fn(self, batch):
        if self.return_index:
            imgs, labels, centers, descriptions, idxs = zip(*batch)
        else:
            imgs, labels, centers, descriptions = zip(*batch)
        imgs = torch.concat(imgs)
        labels = torch.concat(labels)
        centers = torch.concat(centers)
        if self.return_index:
            return imgs, labels, centers, descriptions, idxs
        else:
            return imgs, labels, centers, descriptions

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            collate_fn=self.collate_fn
        )


class GenericRasterizedSVGDataset(Dataset):
    def __init__(self,
                 csv_path:str,
                 train:bool,
                 img_size:int = 128,
                 channels:int = 3,
                 fill:bool = True,
                 stroke_width:float = 0.4,
                 subset:str=None,
                 **kwargs) -> None:
        super(GenericRasterizedSVGDataset).__init__()

        self.csv_path = csv_path
        self.train = train
        self.img_size = img_size
        self.channels = channels
        self.fill = fill
        self.stroke_width = stroke_width
        if "subset" in kwargs:
            self.subset = kwargs["subset"]
        else:
            self.subset = subset

        self.df = pd.read_csv(self.csv_path)
        self.class2idx = {class_name: idx for idx, class_name in enumerate(self.df["class"].unique())}
        self.idx2class = {idx: class_name for class_name, idx in self.class2idx.items()}

        if self.train is None:
            self.df = self.df[self.df["split"] == "test"].reset_index(drop=True)
        else:
            if self.train:
                self.df = self.df[self.df["split"] == "train"].reset_index(drop=True)
            else:
                self.df = self.df[self.df["split"] == "val"].reset_index(drop=True)

        if self.subset is not None:
            print(f"Using subset {self.subset}")
            if isinstance(self.subset, list):
                self.df = self.df[self.df["class"].isin(self.subset)].reset_index(drop=True)
            else:
                self.df = self.df[self.df["class"] == self.subset].reset_index(drop=True)

    def _rasterize_svg(self, svg_path, img_size, fill):
        paths, attributes, svg_attributes = svg2paths2(svg_path)
        for i in range(len(attributes)):
            if "fill" in attributes[i]:
                attributes[i]["fill"] = "black" if fill else "none"
            attributes[i]["stroke-width"] = f"{self.stroke_width}"
        rasterized = disvg(paths, viewbox = svg_attributes["viewBox"], dimensions = (img_size, img_size), attributes = attributes, paths2Drawing=True)
        return drawing_to_tensor(rasterized)
    
    def __getitem__(self, index) -> Tensor:
        img = self._rasterize_svg(self.df.iloc[index]["simplified_svg_file_path"], self.img_size, self.fill)
        label = self.class2idx[self.df.iloc[index]["class"]]
        return img, label

    def __len__(self) -> int:
        return len(self.df)

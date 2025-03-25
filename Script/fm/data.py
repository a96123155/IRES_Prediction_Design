# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Sequence, Tuple, List, Union
import pickle
import re
import shutil
import torch
from pathlib import Path
from .constants import proteinseq_toks, rnaseq_toks
import math
import random
import itertools
from copy import deepcopy

RawMSA = Sequence[Tuple[str, str]]

class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"), # 
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
        prepend_bos: bool = True,
        append_eos: bool = False,
        use_msa: bool = False,
    mask_prob: float = 0.15, ###---
    ):
        self.mask_prob = mask_prob ###---
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_msa = use_msa

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i  + 1}>")
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}
#         print(self.tok_to_idx)
        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ['<eos>', '<pad>', '<mask>'] # , '<unk>', '<cls>'
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy() # return {"toks": self.toks}

    def get_batch_converter(self):
        if self.use_msa:
            return MSABatchConverter(self)
        else:
            return BatchConverter(self)

    @classmethod
    def from_dict(cls, d, **kwargs):
        return cls(standard_toks=d["toks"], **kwargs)

    @classmethod
    def from_architecture(cls, name: str, theme="protein") -> "Alphabet":
        if name in ("ESM-1", "protein_bert_base"):
            standard_toks = proteinseq_toks["toks"] if theme == "protein" else rnaseq_toks["toks"]
            prepend_toks: Tuple[str, ...] = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks: Tuple[str, ...] = ("<cls>", "<mask>", "<sep>")
            prepend_bos = True
            append_eos = False
            use_msa = False
        elif name in ("ESM-1b", "roberta_large"):
            standard_toks = proteinseq_toks["toks"] if theme == "protein" else rnaseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = True
            use_msa = False
        elif name in ("MSA Transformer", "msa_transformer"):
            standard_toks = proteinseq_toks["toks"] if theme == "protein" else rnaseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = False
            use_msa = True
        else:
            raise ValueError("Unknown architecture selected")
        
        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa)
    
    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # We strip left and right by default
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]

    
class FastaBatchedDataset(object):
    def __init__(self, sequence_labels, sequence_strs, mask_prob = 0.15):
        self.sequence_labels = list(sequence_labels)
        self.sequence_strs = list(sequence_strs)
        self.mask_prob = mask_prob

    @classmethod
    def from_file(cls, fasta_file, mask_prob = 0.15):
        sequence_labels, sequence_strs = [], []
        cur_seq_label = None
        buf = []

        def _flush_current_seq():
            nonlocal cur_seq_label, buf
            if cur_seq_label is None:
                return
            sequence_labels.append(cur_seq_label)
            sequence_strs.append("".join(buf))
            cur_seq_label = None
            buf = []

        with open(fasta_file, "r") as infile:
            for line_idx, line in enumerate(infile):
                if line.startswith(">"):  # label line
                    _flush_current_seq()
                    line = line[1:].strip()
                    if len(line) > 0:
                        cur_seq_label = line
                    else:
                        cur_seq_label = f"seqnum{line_idx:09d}"
                else:  # sequence line
                    buf.append(line.strip())

        _flush_current_seq()

        assert len(set(sequence_labels)) == len(sequence_labels)

        return cls(sequence_labels, sequence_strs, mask_prob)

    def __len__(self):
        return len(self.sequence_labels)

    def mask_sequence(self, seq): ###---
        length = len(seq)
#         print(self.mask_prob)
        max_length = math.ceil(length * self.mask_prob)
        rand = random.sample(range(0, length), max_length)
        res = ''.join(['<mask>' if idx in rand else ele for idx, ele in enumerate(seq)])
        #print(seq, rand, res)
        return rand, res
    def __getitem__(self, idx):
        sequence_str = self.sequence_strs[idx]
        sequence_label = self.sequence_labels[idx]
        masked_indices, masked_sequence_str = self.mask_sequence(sequence_str)
        return sequence_label, sequence_str, masked_sequence_str, masked_indices
    #return self.sequence_labels[idx], self.sequence_strs[idx]

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = [(len(s), i) for i, s in enumerate(self.sequence_strs)]
        sizes.sort()
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0

        for sz, i in sizes:
            sz += extra_toks_per_seq
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()
        return batches




class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list, masked_seq_str_list, masked_indices_list = zip(*raw_batch)

        masked_seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in masked_seq_str_list] ###---
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list] ###---
#         print('====', seq_str_list)
#         print('----', masked_seq_str_list)
#         print('++++', masked_seq_encoded_list)
#         print('****', seq_encoded_list)
        max_len = max(len(seq_encoded) for seq_encoded in masked_seq_encoded_list)
        tokens = torch.empty(
            (
                batch_size,
                
        max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        masked_tokens = deepcopy(tokens)
        labels = []
        strs, masked_strs = [], []
        masked_indices = []
#         print('=================')
        for i, (label, seq_str, masked_seq_str, seq_encoded, masked_seq_encoded, indices_mask) in enumerate(
            zip(batch_labels, seq_str_list, masked_seq_str_list, seq_encoded_list, masked_seq_encoded_list, masked_indices_list) ###---
        ):
            labels.append(label)
            strs.append(seq_str)
            masked_strs.append(masked_seq_str)
            masked_indices.append(indices_mask)

            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
                masked_tokens[i, 0] = self.alphabet.cls_idx

            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            masked_seq = torch.tensor(masked_seq_encoded, dtype=torch.int64)
#             print(tokens, masked_tokens)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = seq

            masked_tokens[
                i,
                int(self.alphabet.prepend_bos) : len(masked_seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = masked_seq
#             print(tokens, masked_tokens)
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
                masked_tokens[i, len(masked_seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
#             print(tokens, masked_tokens)
        return labels, strs, masked_strs, tokens, masked_tokens, masked_indices
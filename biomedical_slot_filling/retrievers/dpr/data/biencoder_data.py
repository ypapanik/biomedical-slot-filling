import collections
import glob
import logging
import os
from typing import List


import torch
from torch import Tensor as T

from biomedical_slot_filling.retrievers.dpr.utils.data_utils import Tensorizer

logger = logging.getLogger(__name__)
BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])


class BiEncoderSample(object):
    query: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]


class RepTokenSelector(object):
    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):
        raise NotImplementedError


class RepStaticPosTokenSelector(RepTokenSelector):
    def __init__(self, static_position: int = 0):
        self.static_position = static_position

    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):
        return self.static_position


class RepSpecificTokenSelector(RepTokenSelector):
    def __init__(self, token: str = "[CLS]"):
        self.token = token
        self.token_id = None

    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):
        if not self.token_id:
            self.token_id = tenzorizer.get_token_id(self.token)
        token_indexes = (input_ids == self.token_id).nonzero()
        # check if all samples in input_ids has index presence and out a default value otherwise
        bsz = input_ids.size(0)
        if bsz == token_indexes.size(0):
            return token_indexes

        token_indexes_result = []
        found_idx_cnt = 0
        for i in range(bsz):
            if (
                found_idx_cnt < token_indexes.size(0)
                and token_indexes[found_idx_cnt][0] == i
            ):
                # this samples has the special token
                token_indexes_result.append(token_indexes[found_idx_cnt])
                found_idx_cnt += 1
            else:
                logger.warning("missing special token %s", input_ids[i])

                token_indexes_result.append(
                    torch.tensor([i, 0]).to(input_ids.device)
                )  # setting 0-th token, i.e. CLS for BERT as the special one
        token_indexes_result = torch.stack(token_indexes_result, dim=0)
        return token_indexes_result


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        special_token: str = None,
        shuffle_positives: bool = False,
        query_special_suffix: str = None,
        encoder_type: str = None,
    ):
        self.selector = RepStaticPosTokenSelector()
        self.special_token = special_token
        self.encoder_type = encoder_type
        self.shuffle_positives = shuffle_positives
        self.query_special_suffix = query_special_suffix

    def load_data(self):
        raise NotImplementedError

    def __getitem__(self, index) -> BiEncoderSample:
        raise NotImplementedError

    def _process_query(self, query: str):
        # as of now, always normalize query
        query = normalize_question(query)
        if self.query_special_suffix and not query.endswith(self.query_special_suffix):
            query += self.query_special_suffix

        return query


def get_dpr_files(source_name) -> List[str]:
    print(source_name, os.listdir())
    if os.path.exists(source_name) or glob.glob(source_name):
        return glob.glob(source_name)
    else:
        # try to use data downloader
        from biomedical_slot_filling.retrievers.dpr.data.download_data import download

        return download(source_name)


def normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    return ctx_text


def normalize_question(question: str) -> str:
    question = question.replace("’", "'")
    return question


def load_data(
    ctx_src,
    normalize: bool = True,
    return_as_list_of_tuples=True,
):
    ctxs = {}
    with open(ctx_src, 'r') as f:
        for i, line in enumerate(f):
            if i ==0:
                continue
            try:
                pmid, text, title = line.strip().split('\t')
            except:
                pmid, text = line.strip().split('\t')
                title = ' '
            if normalize:
                text = normalize_passage(text)
            ctxs[pmid] = BiEncoderPassage(text=text, title=title)
    if return_as_list_of_tuples:
        return [(k, v) for k, v in ctxs.items()]
    else:
        return ctxs

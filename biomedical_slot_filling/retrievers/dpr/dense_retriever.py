import logging
from typing import List

import torch
from torch import nn, Tensor as T

from biomedical_slot_filling.retrievers.dpr.data.biencoder_data import RepTokenSelector
from biomedical_slot_filling.retrievers.dpr.models.biencoder import (
    _select_span_with_token,
    BiEncoder,
)
from biomedical_slot_filling.retrievers.dpr.options import setup_logger
from biomedical_slot_filling.retrievers.dpr.utils.data_utils import Tensorizer

logger = logging.getLogger()
setup_logger(logger)


def generate_question_vectors(
    question_encoder: torch.nn.Module,
    tensorizer: Tensorizer,
    questions: List[str],
    bsz: int,
    query_token: str = None,
    selector: RepTokenSelector = None,
) -> T:
    n = len(questions)
    query_vectors = []

    with torch.no_grad():
        for j, batch_start in enumerate(range(0, n, bsz)):
            batch_questions = questions[batch_start : batch_start + bsz]

            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    batch_token_tensors = [
                        _select_span_with_token(q, tensorizer, token_str=query_token)
                        for q in batch_questions
                    ]
                else:
                    batch_token_tensors = [
                        tensorizer.text_to_tensor(" ".join([query_token, q]))
                        for q in batch_questions
                    ]
            else:
                batch_token_tensors = [
                    tensorizer.text_to_tensor(q) for q in batch_questions
                ]

            q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
            q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
            q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)

            if selector:
                rep_positions = selector.get_positions(q_ids_batch, tensorizer)

                _, out, _ = BiEncoder.get_representation(
                    question_encoder,
                    q_ids_batch,
                    q_seg_batch,
                    q_attn_mask,
                    representation_token_pos=rep_positions,
                )
            else:
                _, out, _ = question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

            query_vectors.extend(out.cpu().split(1, dim=0))

            if len(query_vectors) % 100 == 0:
                logger.info("Encoded queries %d", len(query_vectors))

    query_tensor = torch.cat(query_vectors, dim=0)
    logger.info("Total encoded queries tensor %s", query_tensor.size())
    assert query_tensor.size(0) == len(questions)
    return query_tensor


class DenseRetriever(object):
    def __init__(
        self, question_encoder: nn.Module, batch_size: int, tensorizer: Tensorizer
    ):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.selector = None

    def encode_queries(self, questions: List[str], query_token: str = None) -> T:

        bsz = self.batch_size
        self.question_encoder.eval()
        return generate_question_vectors(
            self.question_encoder,
            self.tensorizer,
            questions,
            bsz,
            query_token=query_token,
            selector=self.selector,
        )

import logging
from typing import Tuple, List

from biomedical_slot_filling.retrievers.dpr.data.biencoder_data import (
    Dataset,
    BiEncoderSample,
    BiEncoderPassage,
    normalize_passage,
)
from biomedical_slot_filling.retrievers.dpr.utils.data_utils import read_data_from_json_files

logger = logging.getLogger(__name__)


class JsonQADataset(Dataset):
    def __init__(
        self,
        file: str,
        special_token: str = None,
        encoder_type: str = None,
        shuffle_positives: bool = False,
        normalize: bool = False,
        query_special_suffix: str = None,
    ):
        super().__init__(
            special_token=special_token,
            encoder_type=encoder_type,
            shuffle_positives=shuffle_positives,
            query_special_suffix=query_special_suffix,
        )
        self.file = file
        self.data = []
        self.normalize = normalize

    def load_data(self):
        data = read_data_from_json_files([self.file])
        # filter those without positive ctx
        self.data = [r for r in data if len(r["positive_ctxs"]) > 0]
        logger.info("Total cleaned data size: {}".format(len(self.data)))

    def __getitem__(self, index) -> BiEncoderSample:
        json_sample = self.data[index]
        r = BiEncoderSample()
        r.query = self._process_query(json_sample["question"])

        positive_ctxs = json_sample["positive_ctxs"]
        negative_ctxs = (
            json_sample["negative_ctxs"] if "negative_ctxs" in json_sample else []
        )
        hard_negative_ctxs = (
            json_sample["hard_negative_ctxs"]
            if "hard_negative_ctxs" in json_sample
            else []
        )

        for ctx in positive_ctxs + negative_ctxs + hard_negative_ctxs:
            if "title" not in ctx:
                ctx["title"] = None

        def create_passage(ctx: dict):
            return BiEncoderPassage(
                normalize_passage(ctx["text"]) if self.normalize else ctx["text"],
                ctx["title"],
            )

        r.positive_passages = [create_passage(ctx) for ctx in positive_ctxs]
        r.negative_passages = [create_passage(ctx) for ctx in negative_ctxs]
        r.hard_negative_passages = [create_passage(ctx) for ctx in hard_negative_ctxs]
        return r

    def __len__(self):
        return len(self.data)

    def get_qas(self) -> Tuple[List[str], List[str]]:
        return [s["question"] for s in self.data], [s["answers"] for s in self.data]

    def get_qas_range(
        self, start_idx: int, end_idx: int
    ) -> Tuple[List[str], List[str]]:
        return (
            [s["question"] for s in self.data[start_idx:end_idx]],
            [s["answers"] for s in self.data[start_idx:end_idx]],
        )

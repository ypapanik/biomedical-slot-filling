from typing import List
from transformers import PreTrainedTokenizerFast
import torch
import logging
import random

from biomedical_slot_filling.retrievers.dpr.biencoder_hypers import BiEncoderHypers
from biomedical_slot_filling.retrievers.dpr.distloader_base import DistBatchesBase, MultiFileLoader

logger = logging.getLogger(__name__)


class BiEncoderInst:
    __slots__ = 'qry', 'pos_ctx', 'neg_ctx'

    def __init__(self, qry, pos_ctx, neg_ctx):
        self.qry = qry
        self.pos_ctx = pos_ctx
        self.neg_ctx = neg_ctx


class BiEncoderBatches(DistBatchesBase):
    def __init__(self, insts: List[BiEncoderInst], hypers: BiEncoderHypers,
                 qry_tokenizer: PreTrainedTokenizerFast, ctx_tokenizer: PreTrainedTokenizerFast):
        super().__init__(insts, hypers)
        self.qry_tokenizer = qry_tokenizer
        self.ctx_tokenizer = ctx_tokenizer

    def make_batch(self, index, insts):
        ctx_titles = [title for i in insts for title in [i.pos_ctx[0], i.neg_ctx[0]]]
        ctx_texts = [text for i in insts for text in [i.pos_ctx[1], i.neg_ctx[1]]]
        # if index == 0:
        #     logger.info(f'titles = {ctx_titles}\ntexts = {ctx_texts}')
        qrys = [i.qry for i in insts]
        ctxs_tensors = self.ctx_tokenizer(ctx_titles, ctx_texts, max_length=self.hypers.seq_len_c,
                                          truncation=True, padding="longest", return_tensors="pt")
        qrys_tensors = self.qry_tokenizer(qrys, max_length=self.hypers.seq_len_q,
                                          truncation=True, padding="longest", return_tensors="pt")
        positive_indices = torch.arange(len(insts), dtype=torch.long) * 2
        assert qrys_tensors['input_ids'].shape[0] * 2 == ctxs_tensors['input_ids'].shape[0]
        return qrys_tensors['input_ids'], qrys_tensors['attention_mask'], \
               ctxs_tensors['input_ids'], ctxs_tensors['attention_mask'], \
               positive_indices


class BiEncoderLoader(MultiFileLoader):
    def __init__(self, hypers: BiEncoderHypers, per_gpu_batch_size: int, qry_tokenizer, ctx_tokenizer, data_dir, *,
                 files_per_dataloader=1, checkpoint_info=None):
        super().__init__(hypers=hypers, per_gpu_batch_size=per_gpu_batch_size, train_dir=data_dir,
                         checkpoint_info=checkpoint_info, files_per_dataloader=files_per_dataloader)
        self.qry_tokenizer = qry_tokenizer
        self.ctx_tokenizer = ctx_tokenizer

    def batch_dict(self, batch):
        """
        :param batch: input_ids_q, attention_mask_q, input_ids_c, attention_mask_c, positive_indices
        :return:
        """
        batch = tuple(t.to(self.hypers.device) for t in batch)
        return {'input_ids_q': batch[0], 'attention_mask_q': batch[1],
                'input_ids_c': batch[2], 'attention_mask_c': batch[3],
                'positive_indices': batch[4]}

    def display_batch(self, batch):
        input_ids_q = batch[0]
        input_ids_c = batch[2]
        positive_indices = batch[4]
        logger.info(f'{input_ids_q.shape} queries and {input_ids_c.shape} contexts\n{positive_indices}')
        qndx = random.randint(0, input_ids_q.shape[0]-1)
        logger.info(f'   query: {self.qry_tokenizer.decode(input_ids_q[qndx])}')
        logger.info(f' positve: {self.ctx_tokenizer.decode(input_ids_c[positive_indices[qndx]])}')
        logger.info(f'negative: {self.ctx_tokenizer.decode(input_ids_c[1+positive_indices[qndx]])}')

    def _one_load(self, instances) -> DistBatchesBase:
        insts = []
        for instance in instances:
            qry = instance['question']
            positives = []
            for pos in instance['positive_ctxs']:
                positives.append((pos['title'], pos['text']))
            negatives = []
            for neg in instance['negative_ctxs']:
                negatives.append((neg['title'], neg['text']))
            for neg in instance['hard_negative_ctxs']:
                negatives.append((neg['title'], neg['text']))


            if len(negatives) == 0 or len(positives) == 0:
                logger.warning(f'bad instance! {len(negatives)} negatives, {len(positives)} positives ')
                continue
            for positive in positives:
                for negative in negatives:
                    insts.append(BiEncoderInst(qry, positive, negative))
        return BiEncoderBatches(insts, self.hypers, self.qry_tokenizer, self.ctx_tokenizer)

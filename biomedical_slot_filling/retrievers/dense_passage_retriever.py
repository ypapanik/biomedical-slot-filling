import json
import logging
import os
import pickle
from datetime import datetime
from typing import List

import numpy as np
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, DPRQuestionEncoderTokenizer, \
    DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast

from biomedical_slot_filling.retrievers.dpr.biencoder_gcp import BiEncoder
from biomedical_slot_filling.retrievers.dpr.biencoder_hypers import BiEncoderHypers
from biomedical_slot_filling.retrievers.dpr.dataloader_biencoder import BiEncoderLoader

from biomedical_slot_filling.retrievers.dpr.faiss_retriever import LocalFaissRetriever
from biomedical_slot_filling.retrievers.dpr.indexer.faiss_indexers import (
    DenseFlatIndexer,
    DenseHNSWFlatIndexer,
    DenseHNSWSQIndexer,
)

from biomedical_slot_filling.retrievers.dpr.options import (
    setup_logger,
)
from biomedical_slot_filling.retrievers.dpr.reporting import Reporting

from biomedical_slot_filling.retrievers.dpr.transformer_optimize import TransformerOptimize

logger = logging.getLogger()
setup_logger(logger)


indexers = {
    "flat": DenseFlatIndexer(),
    "hnsw": DenseHNSWFlatIndexer(),
    "hnsw_sq": DenseHNSWSQIndexer(),
}

def encode(doc_batch: List, ctx_encoder: DPRContextEncoder,
           ctx_tokenizer: DPRContextEncoderTokenizerFast,
           device: str) -> np.ndarray:
    documents = {"title": [doci['title'] for doci in doc_batch], 'text': [doci['text'] for doci in doc_batch]}
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(
        documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=device), return_dict=True).pooler_output
    return embeddings.detach().cpu().to(dtype=torch.float16).numpy()


class DPR:
    def __init__(self):
        self.device = str(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.n_gpu = torch.cuda.device_count()
        self.vector_size = 768

    def load_context_model(self, context_encoder):
        self.ctx_encoder = DPRContextEncoder.from_pretrained(context_encoder, cache_dir='data/cache_dir')
        self.ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base',
                                                                            cache_dir='data/cache_dir')
        self.ctx_encoder.to(device=self.device)
        if self.n_gpu >1:
            self.ctx_encoder = torch.nn.DataParallel(self.ctx_encoder)
        self.ctx_encoder.eval()
    def load_query_model(self, query_encoder):
        self.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base',
                                                                           cache_dir='data/cache_dir')
        self.query_model = DPRQuestionEncoder.from_pretrained(query_encoder,
                                                              cache_dir='data/cache_dir')
        self.query_model.to(device=self.device)
        if self.n_gpu >1:
            self.query_model = torch.nn.DataParallel(self.query_model)
        self.query_model.eval()

    def load_index(self, index_path, index):
        index.init_index(self.vector_size)
        self.retriever = LocalFaissRetriever(index)
        self.retriever.index.deserialize(index_path)

        # self.index = ANNIndex(index_path+'.index')
        # with open(index_path+'.meta') as f:
        #     self.pmids = [line.strip() for line in f]

    def query(self, questions, n, batch_size=32, del_index_after_use=False):
        encoded_questions = []
        for j, batch_start in enumerate(range(0, len(questions), batch_size)):
            print(str(j*batch_size), '/', len(questions))
            batch_questions = questions[batch_start : batch_start + batch_size]
            tokenized_questions = self.query_tokenizer(batch_questions, return_tensors='pt',
                                                       truncation=True,
                                                       max_length=16,
                                                       padding=True)["input_ids"]
            questions_tensor = self.query_model(tokenized_questions.to(device=self.device)).pooler_output.cpu().detach().numpy()
            encoded_questions.extend(questions_tensor)
        encoded_questions = np.array(encoded_questions)

        results = self.retriever.get_top_docs(encoded_questions, n, del_index_after_use=del_index_after_use)
        return results
        # results = self.index.search(query_vectors=encoded_questions, k=n)
        # retrieved_indices = results[1]
        # retrieved_scores = results[0]
        # retrieved_pmids = [[self.pmids[i] for i in doc_indices] for doc_indices in retrieved_indices]
        # print(retrieved_pmids, retrieved_indices, retrieved_scores)
        # return retrieved_pmids, retrieved_scores

    def encode_context(self, ctx_src, out_dir, batch_size=128):
        report = Reporting()
        doc_batch = []
        encoded_passages = {}
        os.mkdir(out_dir)
        enc_passages_f = open(os.path.join(out_dir, 'encoded_0'), 'wb')
        with open(ctx_src, 'r') as f:
            for line_nr,line in enumerate(f):
                if line_nr == 0:
                    continue
                if line_nr %1000000 == 0:
                    print(datetime.now(), line_nr)
                    pickle.dump(encoded_passages, enc_passages_f)
                    enc_passages_f.close()
                    encoded_passages = {}
                    enc_passages_f = open(os.path.join(out_dir, 'encoded_'+str(line_nr)), 'wb')
                if report.is_time():
                    print(
                        f'{datetime.now()} On instance {report.check_count}, {report.check_count / report.elapsed_seconds()} instances per second')
                pmid, text, title = line.strip().split('\t')

                doc_batch.append({
                    'pmid': pmid,
                    'text':text,
                    'title':title
                })
                batch_pmids = [d['pmid'] for d in doc_batch]
                if len(doc_batch) == batch_size:
                    embeddings = encode(doc_batch, self.ctx_encoder, self.ctx_tokenizer, device=self.device)
                    for pmid, vector in zip(batch_pmids, embeddings):
                        encoded_passages[pmid] = vector
                    doc_batch = []

            if len(doc_batch) > 0:
                embeddings = encode(doc_batch, self.ctx_encoder, self.ctx_tokenizer, device=self.device)
                for pmid, vector in zip(batch_pmids, embeddings):
                    encoded_passages[pmid] = vector
                pickle.dump(encoded_passages, enc_passages_f)
                enc_passages_f.close()

    def build_index(
            self,
            index,
            encoded_files,
            index_path,
            qa_selector=None,
    ):
        index_buffer_sz = index.buffer_size
        index.init_index(self.vector_size)
        self.retriever = LocalFaissRetriever(index)
        if qa_selector:
            logger.info("Using custom representation token selector")
            self.retriever.selector = qa_selector

        self.retriever.index_encoded_data(
            vector_files=encoded_files,
            buffer_size=index_buffer_sz,
        )
        self.retriever.index.serialize(index_path)

        # build_index(corpus_file=encoded_file, output_file=index_path,
        #             ef_search=128,
        #             ef_construction=200,
        #             d=768,
        #             m=128,
        #             index_batch_size=batch_size,
        #             scalar_quantizer=-1)

    def train(
        self,
        output_dir,
        training_file,
        validation_file,
        # model_file,
        gradient_accumulation_steps,
        batch_size,
        num_train_epochs,
            lr,
        # dev_batch_size,
    ):
        num_instances = len(json.load(open(training_file, 'r')))

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        hypers = BiEncoderHypers()
        hypers.output_dir = output_dir
        hypers.gradient_accumulation_steps = gradient_accumulation_steps
        hypers.num_train_epochs = num_train_epochs
        hypers.per_gpu_train_batch_size = batch_size
        hypers.full_train_batch_size = batch_size
        save_every_steps = num_instances // (batch_size*gradient_accumulation_steps)
        hypers.save_steps = save_every_steps
        hypers.learning_rate = lr
        logger.info(f'hypers:\n{hypers}')



        # hypers.model_type = ''
        # hypers.model_name_or_path = ''
        qry_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained('facebook/dpr-question_encoder-multiset-base',
                                                                        cache_dir='data/cache')
        ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained('facebook/dpr-ctx_encoder-multiset-base',
                                                                       cache_dir='data/cache')
        model = BiEncoder(qry_encoder_name_or_path='facebook/dpr-question_encoder-multiset-base',
                          ctx_encoder_name_or_path='facebook/dpr-ctx_encoder-multiset-base',
                          hypers=hypers
        )
        model.to(self.device)
        model.train()
        train_loader = BiEncoderLoader(per_gpu_batch_size=batch_size,
                                 qry_tokenizer=qry_tokenizer,
                                 ctx_tokenizer=ctx_tokenizer, data_dir=training_file,
                                 hypers=hypers)
        if validation_file:
            val_loader = BiEncoderLoader(per_gpu_batch_size=batch_size,
                                       qry_tokenizer=qry_tokenizer,
                                       ctx_tokenizer=ctx_tokenizer, data_dir=validation_file,
                                       hypers=hypers)
        optimizer = TransformerOptimize(
            hypers=hypers,
            num_instances_to_train_over=num_train_epochs * num_instances,
            model=model)

        logger.info(f'save every {save_every_steps} steps, total steps: {optimizer.t_total}, '
              f'batch size={batch_size*gradient_accumulation_steps}')
        best_val_loss = float('inf')
        while True:
            batches = train_loader.get_dataloader()
            if optimizer.global_step >= optimizer.t_total:
                break
            for batch in batches:
                loss, accuracy = optimizer.model(**train_loader.batch_dict(batch))
                loss_val = optimizer.step_loss(loss, accuracy=accuracy)
                if optimizer.global_step >= optimizer.t_total:
                    break
                if optimizer.global_step % (save_every_steps*gradient_accumulation_steps) == 0 :
                    model_to_save = (optimizer.model.module
                                     if hasattr(optimizer.model, "module")
                                     else optimizer.model)
                    if validation_file:
                        val_loss, vall_acc = self.calculate_val_metrics(val_dataloader=val_loader,
                                                                    optimizer=optimizer)
                        logger.info(f'step={optimizer.global_step}, '
                              f'training loss={loss},{loss_val} , training accuracy={accuracy},'
                          f'val_loss={vall_acc}, val_acc={vall_acc}')
                        if val_loss <best_val_loss:
                            logger.info(f'previous loss: {best_val_loss}, current: {val_loss}, saving to {output_dir}')
                            best_val_loss = val_loss
                            model_to_save.save(output_dir)
                            qry_tokenizer.save_pretrained(output_dir)
                    else:
                        out_dir = os.path.join(output_dir,str(optimizer.global_step))
                        if not os.path.exists(out_dir):
                            os.mkdir(out_dir)
                        logger.info(f'step={optimizer.global_step}, '
                                    f'training loss={loss}, training accuracy={accuracy},'
                                    f'saving to {out_dir}')
                        model_to_save.save(out_dir)

    def calculate_val_metrics(self, val_dataloader, optimizer):
        loss = 0
        acc = 0
        for batch in val_dataloader.get_dataloader():
            loss_per_batch, accuracy_per_batch = optimizer.model(**val_dataloader.batch_dict(batch))
            loss += loss_per_batch
            acc += accuracy_per_batch
        return loss, acc

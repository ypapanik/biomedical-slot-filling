from __future__ import absolute_import, division, print_function

import logging
import os
import random
import timeit
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler


from tqdm import tqdm, trange
from transformers import (
    squad_convert_examples_to_features,
    AutoConfig,
    AutoTokenizer,
    AutoModelForQuestionAnswering
)

from transformers import AdamW, get_linear_schedule_with_warmup


from biomedical_slot_filling.data_processing.utils import (read_squad_examples, RawResult, write_predictions)
from biomedical_slot_filling.data_processing.utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad


def set_seed(seed=42, n_gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()


class QuestionAnsweringModel:
    def __init__(self,
                 pretrained_model_name_or_path,
                 model_type,
                 qa_model_path,
                 local_rank=-1,
                 fp16=False,
                 no_cuda=False,
                 cache_dir='data/pretrained/',
                 do_lower_case=True,
                 max_query_length=64,
                 max_seq_length=384,
                 doc_stride=128,
                 threads=4):
        self.model_type = model_type.lower()
        self.logger = logging.getLogger(__name__)
        self.local_rank = local_rank
        self.fp16 = fp16
        self.do_lower_case = do_lower_case
        self.max_query_length = max_query_length
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.output_dir = qa_model_path
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.cache_dir = cache_dir
        self.threads = threads

        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                                                 cache_dir=self.cache_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                                                       do_lower_case=do_lower_case,
                                                       cache_dir=self.cache_dir,
                                                       use_fast=False)
        self.model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                                                      from_tf=bool('.ckpt' in self.pretrained_model_name_or_path),
                                                      config=self.config,
                                                      cache_dir=self.cache_dir)

        # Setup CUDA, GPU & distributed training
        if local_rank == -1 or no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            torch.distributed.init_process_group(backend='nccl')
            self.n_gpu = 1
        self.device = device

        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO if local_rank in [-1, 0] else logging.WARN)
        self.logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                       local_rank, device, self.n_gpu, bool(local_rank != -1), fp16)

        set_seed(n_gpu=self.n_gpu)

        # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
        # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
        # remove the need for this code, but it is still valid.
        if fp16:
            try:
                import apex
                apex.amp.register_half_function(torch, 'einsum')
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")



    def train(self, train_file,
              per_gpu_train_batch_size=8,
              max_steps=-1,
              gradient_accumulation_steps=8,
              num_train_epochs=3,
              weight_decay=0.0,
              learning_rate=5e-5,
              warmup_steps=0,
              adam_epsilon=1e-8,
              fp16_opt_level='O1',
              max_grad_norm=1.0,
              overwrite_output_dir=True,
              eval_file=None):

        if os.path.exists(self.output_dir) and os.listdir(self.output_dir) and not overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(self.output_dir))

        if not os.path.exists(self.output_dir) and self.local_rank in [-1, 0]:
            os.makedirs(self.output_dir)


        # Load pretrained model and tokenizer
        if self.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        self.model.to(self.device)

        train_dataset = self.load_and_cache_examples(input_file=train_file,
                                                     evaluate=False,
                                                     output_examples=False,
                                                     tokenizer=self.tokenizer)

        save_steps = len(train_dataset) // (per_gpu_train_batch_size*gradient_accumulation_steps)
        logging.info('Saving every'+str(save_steps)+'steps')
        """ Train the model """

        train_batch_size = per_gpu_train_batch_size * max(1, self.n_gpu)
        train_sampler = RandomSampler(train_dataset) if self.local_rank == -1 else DistributedSampler(train_dataset)

        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
        logging.info(f'total steps {t_total}')
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Distributed training (should be after apex fp16 initialization)
        if self.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                              output_device=self.local_rank,
                                                              find_unused_parameters=True)

        global_step = 1
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        print(f"Number of epochs: {num_train_epochs}")
        train_iterator = trange(int(num_train_epochs), desc="Epoch", disable=self.local_rank not in [-1, 0])
        curr_f1  = 0.0
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=self.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):

                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          "token_type_ids": batch[2],
                          'start_positions': batch[3],
                          'end_positions': batch[4]}
                outputs = self.model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                if self.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    if self.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                    if self.local_rank in [-1, 0] and global_step % save_steps == 0:
                        # Save model checkpoint
                        if self.local_rank == -1 and eval_file:  # Only evaluate when single GPU otherwise metrics may not average well
                            # Log metrics
                            results = self.predict(predict_file=eval_file)
                            print(results)
                            if results['f1'] > curr_f1:
                                curr_f1 = results['f1']
                                model_to_save = self.model.module if hasattr(self.model,
                                                                             'module') else self.model  # Take care of distributed/parallel training
                                model_to_save.save_pretrained(self.output_dir)
                                self.tokenizer.save_pretrained(self.output_dir)
                                self.logger.info("f1 improved: %s Saving model checkpoint to %s",
                                                 str(results['f1']),
                                                 self.output_dir)

                if max_steps > 0 and global_step > max_steps:
                    epoch_iterator.close()
                    break
            if max_steps > 0 and global_step > max_steps:
                train_iterator.close()
                break

        self.logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        return global_step, tr_loss / global_step

    def predict(self,
                predict_file,
                prefix="",
                per_gpu_eval_batch_size=8,
                version_2_with_negative=False,
                n_best_size=20,
                max_answer_length=30,
                null_score_diff_threshold=0.0,
                verbose_logging=False,
                output_prediction_file=None,
                output_nbest_file=None):

        dataset, examples, features = self.load_and_cache_examples(input_file=predict_file,
                                                                   evaluate=True,
                                                                   output_examples=True,
                                                                   version_2_with_negative=version_2_with_negative,
                                                                   overwrite_cache=False,
                                                                   tokenizer=self.tokenizer)

        eval_batch_size = per_gpu_eval_batch_size * max(1, self.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset) if self.local_rank == -1 else DistributedSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        # multi-gpu evaluate
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)
        all_results = []
        start_time = timeit.default_timer()
        for batch in tqdm(eval_dataloader, desc="Predicting"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                example_indices = batch[3]
                outputs = self.model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                result = RawResult(unique_id=unique_id,
                                   start_logits=to_list(outputs[0][i]),
                                   end_logits=to_list(outputs[1][i]))
                all_results.append(result)

        evalTime = timeit.default_timer() - start_time
        self.logger.info("Prediction done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

        # Compute predictions
        if not output_prediction_file:
            output_prediction_file = os.path.join(self.output_dir, "predictions_{}.json".format(prefix))
        if not  output_nbest_file:
            output_nbest_file = os.path.join(self.output_dir, "nbest_predictions_{}.json".format(prefix))
        if version_2_with_negative:
            output_null_log_odds_file = os.path.join(self.output_dir, "null_odds_{}.json".format(prefix))
        else:
            output_null_log_odds_file = None

        write_predictions(all_examples=examples,
                              all_features=features,
                              all_results=all_results,
                              n_best_size=n_best_size,
                              max_answer_length=max_answer_length,
                              do_lower_case=self.do_lower_case,
                              output_prediction_file=output_prediction_file,
                              output_nbest_file=output_nbest_file,
                              output_null_log_odds_file=output_null_log_odds_file,
                              verbose_logging=verbose_logging,
                              version_2_with_negative=version_2_with_negative,
                              null_score_diff_threshold=null_score_diff_threshold)
        # Evaluate with the official SQuAD script
        self.logger.info("Evaluating")
        evaluate_options = EVAL_OPTS(data_file=predict_file,
                                     pred_file=output_prediction_file,
                                     na_prob_file=output_null_log_odds_file)
        results = evaluate_on_squad(evaluate_options)
        return results

    def load_and_cache_examples(self,
                                input_file,
                                tokenizer,
                                evaluate=False,
                                output_examples=False,
                                version_2_with_negative=False,
                                overwrite_cache=False):

        if self.local_rank not in [-1, 0] and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
            'dev' if evaluate else 'train',
            list(filter(None, self.pretrained_model_name_or_path.split('/'))).pop(),
            str(self.max_seq_length)))

        if os.path.exists(cached_features_file) and not overwrite_cache and not output_examples:
            self.logger.info("Loading features from cached file %s", cached_features_file)
            features_and_dataset = torch.load(cached_features_file)
            features, dataset = features_and_dataset["features"], features_and_dataset["dataset"]
        else:
            self.logger.info("Reading dataset file at %s", input_file)
            examples = read_squad_examples(input_file=input_file,
                                           is_training=not evaluate,
                                           version_2_with_negative=version_2_with_negative)
            self.logger.info("Creating features from dataset file at %s", input_file)
            features, dataset = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=self.max_seq_length,
                doc_stride=self.doc_stride,
                max_query_length=self.max_query_length,
                is_training=not evaluate,
                return_dataset="pt",
                threads=self.threads,
            )

            if self.local_rank in [-1, 0]:
                self.logger.info("Saving features into cached file %s", cached_features_file)
                torch.save({"features": features, "dataset": dataset}, cached_features_file)
        if self.local_rank == 0 and not evaluate:
            # Make sure only the first process in distributed training process the dataset, and the others will use the cache
            torch.distributed.barrier()

        if output_examples:
            return dataset, examples, features
        return dataset

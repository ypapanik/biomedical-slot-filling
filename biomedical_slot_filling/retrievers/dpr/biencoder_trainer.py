import logging
import math
import os
import random
import time
from typing import Tuple

import torch
from torch import Tensor as T, nn

from biomedical_slot_filling.retrievers.dpr.data.biencoder_data import RepStaticPosTokenSelector
from biomedical_slot_filling.retrievers.dpr.data.json_qa_dataset import JsonQADataset
from biomedical_slot_filling.retrievers.dpr.models.biencoder import (
    BiEncoder,
    BiEncoderNllLoss,
    BiEncoderBatch,
)
from biomedical_slot_filling.retrievers.dpr.models.hf_models import (
    get_bert_biencoder_components,
)
from biomedical_slot_filling.retrievers.dpr.options import (
    setup_cfg_gpu,
    get_encoder_params_state_from_cfg, setup_logger,
)
from biomedical_slot_filling.retrievers.dpr.utils.data_utils import (
    ShardedDataIterator,
    MultiSetDataIterator,
    Tensorizer,
)
from biomedical_slot_filling.retrievers.dpr.utils.dist_utils import all_gather_list
from biomedical_slot_filling.retrievers.dpr.utils.model_utils import (
    load_states_from_checkpoint,
    setup_for_distributed_mode,
    get_schedule_linear,
    get_model_obj,
    CheckpointState,
    move_to_device,
)

logger = logging.getLogger()
setup_logger(logger)

class BiEncoderTrainer(object):
    """
    BiEncoder training pipeline component. Can be used to initiate or resume training and validate the trained model
    using either binary classification's NLL loss or average rank of the question's gold passages across dataset
    provided pools of negative passages. For full IR accuracy evaluation, please see generate_dense_embeddings.py
    and dense_retriever.py CLI tools.
    """

    def __init__(
        self,
        local_rank,
        distributed_world_size,
        output_dir,
        model_file=None,
        learning_rate=1e-5,
        adam_eps=1e-8,
        weight_decay=0.0,
        fp16=False,
        fp16_opt_level="01",
        do_lower_case=True,
    ):
        self.shard_id = local_rank if local_rank != -1 else 0
        self.distributed_factor = distributed_world_size or 1
        self.output_dir = output_dir
        logger.info("***** Initializing components for training *****")

        # if model file is specified, encoder parameters from saved state should be used for initialization
        saved_state = None
        if model_file:
            saved_state = load_states_from_checkpoint(model_file)

        tensorizer, encoder, optimizer = get_bert_biencoder_components(
            state=saved_state.encoder_params,
            inference_only=False,
            fix_ctx_encoder=False,
            special_tokens=None,
            learning_rate=learning_rate,
            adam_eps=adam_eps,
            weight_decay=weight_decay,
        )
        n_gpu, device = setup_cfg_gpu(local_rank=local_rank)
        model, optimizer = setup_for_distributed_mode(
            model=encoder,
            optimizer=optimizer,
            device=device,
            n_gpu=n_gpu,
            local_rank=local_rank,
            fp16=fp16,
            fp16_opt_level=fp16_opt_level,
        )
        self.device = device
        self.n_gpu = n_gpu
        self.biencoder = model
        self.optimizer = optimizer
        self.tensorizer = tensorizer
        self.start_epoch = 0
        self.start_batch = 0
        self.scheduler_state = None
        self.best_validation_result = None
        self.best_cp_name = None
        self.local_rank = local_rank
        self.do_lower_case = do_lower_case
        self.fp16 = fp16

        if saved_state:
            self._load_saved_state(saved_state)

        self.dev_iterator = None
        self.selector = RepStaticPosTokenSelector()

    def get_data_iterator(
        self,
        datasets_list,
        batch_size: int,
        is_train_set: bool,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
        rank: int = 0,
        sampling_rates=None,
    ):

        # hydra_datasets = (
        #     self.ds_train_datasets if is_train_set else self.ds_dev_datasets
        # )
        # sampling_rates = self.ds_sampling_rates

        # randomized data loading to avoid file system congestion
        rnd = random.Random(rank)
        rnd.shuffle(datasets_list)
        [ds.load_data() for ds in datasets_list]

        sharded_iterators = [
            ShardedDataIterator(
                ds,
                shard_id=self.shard_id,
                num_shards=self.distributed_factor,
                batch_size=batch_size,
                shuffle=shuffle,
                shuffle_seed=shuffle_seed,
                offset=offset,
            )
            for ds in datasets_list
        ]

        return MultiSetDataIterator(
            sharded_iterators,
            shuffle_seed,
            shuffle,
            sampling_rates=sampling_rates if (is_train_set and sampling_rates) else [1],
            rank=rank,
        )

    def run_train(
        self,
        training_files,
        dev_files,
        batch_size,
        gradient_accumulation_steps,
        num_train_epochs,
        warmup_steps,
        eval_per_epoch,
        dev_batch_size,
    ):
        training_datasets = [JsonQADataset(f) for f in training_files]
        dev_datasets = [JsonQADataset(f) for f in dev_files]
        train_iterator = self.get_data_iterator(
            datasets_list=training_datasets,
            batch_size=batch_size,
            is_train_set=True,
            shuffle=True,
            offset=self.start_batch,
            rank=self.local_rank,
            sampling_rates=[1] * len(training_datasets),
        )
        max_iterations = train_iterator.get_max_iterations()
        logger.info("  Total iterations per epoch=%d", max_iterations)
        if max_iterations == 0:
            logger.warning("No data found for training.")
            return

        updates_per_epoch = train_iterator.max_iterations // gradient_accumulation_steps

        total_updates = updates_per_epoch * num_train_epochs
        logger.info(" Total updates=%d", total_updates)

        if self.scheduler_state:
            # TODO: ideally we'd want to just call
            # scheduler.load_state_dict(self.scheduler_state)
            # but it doesn't work properly as of now

            logger.info("Loading scheduler state %s", self.scheduler_state)
            shift = int(self.scheduler_state["last_epoch"])
            logger.info("Steps shift %d", shift)
            scheduler = get_schedule_linear(
                self.optimizer,
                warmup_steps,
                total_updates,
                steps_shift=shift,
            )
        else:
            scheduler = get_schedule_linear(self.optimizer, warmup_steps, total_updates)

        eval_step = math.ceil(updates_per_epoch / eval_per_epoch)
        logger.info("  Eval step = %d", eval_step)
        logger.info("***** Training *****")

        for epoch in range(self.start_epoch, int(num_train_epochs)):
            logger.info("***** Epoch %d *****", epoch)
            self._train_epoch(
                scheduler=scheduler,
                epoch=epoch,
                eval_step=eval_step,
                train_data_iterator=train_iterator,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=2.0,
                log_result_step=100,
                loss_scale_factors=None,
                num_hard_negatives=1,
                num_other_negatives=0,
                rolling_loss_step=100,
                shuffle_positives=False,
                special_token=None,
                dev_batch_size=dev_batch_size,
                dev_datasets=dev_datasets,
            )

        if self.local_rank in [-1, 0]:
            logger.info(
                "Training finished. Best validation checkpoint %s", self.best_cp_name
            )

    def validate_and_save(
        self,
        dev_datasets,
        epoch: int,
        iteration: int,
        dev_batch_size,
        gradient_accumulation_steps,
        scheduler,
        val_av_rank_start_epoch=5,
    ):
        # for distributed mode, save checkpoint for only one process
        save_cp = self.local_rank in [-1, 0]

        if epoch == val_av_rank_start_epoch:
            self.best_validation_result = None

        if not dev_datasets:
            validation_loss = 0
        else:
            if epoch >= val_av_rank_start_epoch:
                self.validate_average_rank(
                    dev_datasets=dev_datasets,
                    dev_batch_size=dev_batch_size,
                    log_batch_step=100,
                    val_av_rank_hard_neg=30,
                    val_av_rank_other_neg=30,
                    val_av_rank_bsz=128,
                    val_av_rank_max_qs=10000,
                )
            else:
                self.validate_nll(
                    dev_datasets=dev_datasets,
                    dev_batch_size=dev_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    hard_negatives=1,
                    log_batch_step=100,
                    other_negatives=0,
                )

        if save_cp:
            cp_name = self._save_checkpoint(scheduler, epoch, iteration)
            logger.info("Saved checkpoint to %s", cp_name)

            if validation_loss < (self.best_validation_result or validation_loss + 1):
                self.best_validation_result = validation_loss
                self.best_cp_name = cp_name
                logger.info("New Best validation checkpoint %s", cp_name)

    def validate_nll(
        self,
        dev_datasets,
        dev_batch_size,
        hard_negatives,
        other_negatives,
        log_batch_step,
        gradient_accumulation_steps,
    ) -> float:
        logger.info("NLL validation ...")
        self.biencoder.eval()

        if not self.dev_iterator:
            self.dev_iterator = self.get_data_iterator(
                datasets_list=dev_datasets,
                batch_size=dev_batch_size,
                is_train_set=False,
                shuffle=False,
                rank=self.local_rank,
            )
        data_iterator = self.dev_iterator

        total_loss = 0.0
        start_time = time.time()
        total_correct_predictions = 0
        num_hard_negatives = hard_negatives
        num_other_negatives = other_negatives
        log_result_step = log_batch_step
        batches = 0

        for i, samples_batch in enumerate(data_iterator.iterate_ds_data()):
            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch
            logger.info("Eval step: %d ,rnk=%s", i, self.local_rank)
            biencoder_input = BiEncoder.create_biencoder_input2(
                samples_batch,
                self.tensorizer,
                True,
                num_hard_negatives,
                num_other_negatives,
                shuffle=False,
            )

            # get the token to be used for representation selection
            rep_positions = self.selector.get_positions(
                biencoder_input.question_ids, self.tensorizer
            )

            loss, correct_cnt = _do_biencoder_fwd_pass(
                model=self.biencoder,
                input=biencoder_input,
                tensorizer=self.tensorizer,
                encoder_type=None,
                rep_positions=rep_positions,
                device=self.device,
                local_rank=self.local_rank,
                n_gpu=self.n_gpu,
                gradient_accumulation_steps=gradient_accumulation_steps,
                global_loss_buf_sz=592000,
            )
            total_loss += loss.item()
            total_correct_predictions += correct_cnt
            batches += 1
            if (i + 1) % log_result_step == 0:
                logger.info(
                    "Eval step: %d , used_time=%f sec., loss=%f ",
                    i,
                    time.time() - start_time,
                    loss.item(),
                )

        total_loss = total_loss / batches
        total_samples = batches * dev_batch_size * self.distributed_factor
        correct_ratio = float(total_correct_predictions / total_samples)
        logger.info(
            "NLL Validation: loss = %f. correct prediction ratio  %d/%d ~  %f",
            total_loss,
            total_correct_predictions,
            total_samples,
            correct_ratio,
        )
        return total_loss

    def validate_average_rank(
        self,
        dev_datasets,
        dev_batch_size,
        val_av_rank_bsz,
        val_av_rank_hard_neg,
        val_av_rank_other_neg,
        log_batch_step,
        val_av_rank_max_qs,
    ) -> float:
        """
        Validates biencoder model using each question's gold passage's rank across the set of passages from the dataset.
        It generates vectors for specified amount of negative passages from each question (see --val_av_rank_xxx params)
        and stores them in RAM as well as question vectors.
        Then the similarity scores are calculted for the entire
        num_questions x (num_questions x num_passages_per_question) matrix and sorted per quesrtion.
        Each question's gold passage rank in that  sorted list of scores is averaged across all the questions.
        :return: averaged rank number
        """
        logger.info("Average rank validation ...")

        self.biencoder.eval()
        distributed_factor = self.distributed_factor

        if not self.dev_iterator:
            self.dev_iterator = self.get_data_iterator(
                datasets_list=dev_datasets,
                batch_size=dev_batch_size,
                is_train_set=False,
                shuffle=False,
                rank=self.local_rank,
            )
        data_iterator = self.dev_iterator

        sub_batch_size = val_av_rank_bsz
        sim_score_f = BiEncoderNllLoss.get_similarity_function()
        q_represenations = []
        ctx_represenations = []
        positive_idx_per_question = []

        num_hard_negatives = val_av_rank_hard_neg
        num_other_negatives = val_av_rank_other_neg

        log_result_step = log_batch_step
        dataset = 0
        for i, samples_batch in enumerate(data_iterator.iterate_ds_data()):
            # samples += 1
            if len(q_represenations) > val_av_rank_max_qs / distributed_factor:
                break

            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch

            biencoder_input = BiEncoder.create_biencoder_input2(
                samples_batch,
                self.tensorizer,
                True,
                num_hard_negatives,
                num_other_negatives,
                shuffle=False,
            )
            total_ctxs = len(ctx_represenations)
            ctxs_ids = biencoder_input.context_ids
            ctxs_segments = biencoder_input.ctx_segments
            bsz = ctxs_ids.size(0)

            # get the token to be used for representation selection
            rep_positions = self.selector.get_positions(
                biencoder_input.question_ids, self.tensorizer
            )

            # split contexts batch into sub batches since it is supposed to be too large to be processed in one batch
            for j, batch_start in enumerate(range(0, bsz, sub_batch_size)):

                q_ids, q_segments = (
                    (biencoder_input.question_ids, biencoder_input.question_segments)
                    if j == 0
                    else (None, None)
                )

                if j == 0 and self.n_gpu > 1 and q_ids.size(0) == 1:
                    # if we are in DP (but not in DDP) mode, all model input tensors should have batch size >1 or 0,
                    # otherwise the other input tensors will be split but only the first split will be called
                    continue

                ctx_ids_batch = ctxs_ids[batch_start : batch_start + sub_batch_size]
                ctx_seg_batch = ctxs_segments[
                    batch_start : batch_start + sub_batch_size
                ]

                q_attn_mask = self.tensorizer.get_attn_mask(q_ids)
                ctx_attn_mask = self.tensorizer.get_attn_mask(ctx_ids_batch)
                with torch.no_grad():
                    q_dense, ctx_dense = self.biencoder(
                        q_ids,
                        q_segments,
                        q_attn_mask,
                        ctx_ids_batch,
                        ctx_seg_batch,
                        ctx_attn_mask,
                        encoder_type=None,
                        representation_token_pos=rep_positions,
                    )

                if q_dense is not None:
                    q_represenations.extend(q_dense.cpu().split(1, dim=0))

                ctx_represenations.extend(ctx_dense.cpu().split(1, dim=0))

            batch_positive_idxs = biencoder_input.is_positive
            positive_idx_per_question.extend(
                [total_ctxs + v for v in batch_positive_idxs]
            )

            if (i + 1) % log_result_step == 0:
                logger.info(
                    "Av.rank validation: step %d, computed ctx_vectors %d, q_vectors %d",
                    i,
                    len(ctx_represenations),
                    len(q_represenations),
                )

        ctx_represenations = torch.cat(ctx_represenations, dim=0)
        q_represenations = torch.cat(q_represenations, dim=0)

        logger.info(
            "Av.rank validation: total q_vectors size=%s", q_represenations.size()
        )
        logger.info(
            "Av.rank validation: total ctx_vectors size=%s", ctx_represenations.size()
        )

        q_num = q_represenations.size(0)
        assert q_num == len(positive_idx_per_question)

        scores = sim_score_f(q_represenations, ctx_represenations)
        values, indices = torch.sort(scores, dim=1, descending=True)

        rank = 0
        for i, idx in enumerate(positive_idx_per_question):
            # aggregate the rank of the known gold passage in the sorted results for each question
            gold_idx = (indices[i] == idx).nonzero()
            rank += gold_idx.item()

        if distributed_factor > 1:
            # each node calcuated its own rank, exchange the information between node and calculate the "global" average rank
            # NOTE: the set of passages is still unique for every node
            eval_stats = all_gather_list([rank, q_num], max_size=100)
            for i, item in enumerate(eval_stats):
                remote_rank, remote_q_num = item
                if i != self.local_rank:
                    rank += remote_rank
                    q_num += remote_q_num

        av_rank = float(rank / q_num)
        logger.info(
            "Av.rank validation: average rank %s, total questions=%d", av_rank, q_num
        )
        return av_rank

    def _train_epoch(
        self,
        scheduler,
        epoch: int,
        eval_step: int,
        train_data_iterator: MultiSetDataIterator,
        log_result_step,
        rolling_loss_step,
        num_hard_negatives,
        num_other_negatives,
        special_token,
        shuffle_positives,
        loss_scale_factors,
        max_grad_norm,
        gradient_accumulation_steps,
        dev_batch_size,
        dev_datasets,
    ):

        rolling_train_loss = 0.0
        epoch_loss = 0
        epoch_correct_predictions = 0
        self.biencoder.train()
        epoch_batches = train_data_iterator.max_iterations
        data_iteration = 0

        dataset = 0
        for i, samples_batch in enumerate(
            train_data_iterator.iterate_ds_data(epoch=epoch)
        ):
            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch

            # to be able to resume shuffled ctx- pools
            data_iteration = train_data_iterator.get_iteration()
            random.seed(epoch + data_iteration)

            biencoder_batch = BiEncoder.create_biencoder_input2(
                samples_batch,
                self.tensorizer,
                True,
                num_hard_negatives,
                num_other_negatives,
                shuffle=True,
                shuffle_positives=shuffle_positives,
                query_token=special_token,
            )

            selector = RepStaticPosTokenSelector()

            rep_positions = selector.get_positions(
                biencoder_batch.question_ids, self.tensorizer
            )

            loss_scale = loss_scale_factors[dataset] if loss_scale_factors else None
            loss, correct_cnt = _do_biencoder_fwd_pass(
                self.biencoder,
                biencoder_batch,
                self.tensorizer,
                encoder_type=None,
                rep_positions=rep_positions,
                loss_scale=loss_scale,
                local_rank=self.local_rank,
                device=self.device,
                n_gpu=self.n_gpu,
                gradient_accumulation_steps=gradient_accumulation_steps,
                global_loss_buf_sz=592000,
            )

            epoch_correct_predictions += correct_cnt
            epoch_loss += loss.item()
            rolling_train_loss += loss.item()

            if self.fp16:
                from apex import amp

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(self.optimizer), max_grad_norm
                    )
            else:
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.biencoder.parameters(), max_grad_norm
                    )

            if (i + 1) % gradient_accumulation_steps == 0:
                self.optimizer.step()
                scheduler.step()
                self.biencoder.zero_grad()

            if i % log_result_step == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch: %d: Step: %d/%d, loss=%f, lr=%f",
                    epoch,
                    data_iteration,
                    epoch_batches,
                    loss.item(),
                    lr,
                )

            if (i + 1) % rolling_loss_step == 0:
                logger.info("Train batch %d", data_iteration)
                latest_rolling_train_av_loss = rolling_train_loss / rolling_loss_step
                logger.info(
                    "Avg. loss per last %d batches: %f",
                    rolling_loss_step,
                    latest_rolling_train_av_loss,
                )
                rolling_train_loss = 0.0

            if data_iteration % eval_step == 0:
                logger.info(
                    "rank=%d, Validation: Epoch: %d Step: %d/%d",
                    self.local_rank,
                    epoch,
                    data_iteration,
                    epoch_batches,
                )
                self.validate_and_save(
                    epoch=epoch,
                    iteration=train_data_iterator.get_iteration(),
                    scheduler=scheduler,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    dev_batch_size=dev_batch_size,
                    dev_datasets=dev_datasets,
                )
                self.biencoder.train()

        logger.info("Epoch finished on %d", self.local_rank)
        self.validate_and_save(
            epoch=epoch,
            iteration=data_iteration,
            scheduler=scheduler,
            dev_datasets=dev_datasets,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dev_batch_size=dev_batch_size,
        )

        epoch_loss = (epoch_loss / epoch_batches) if epoch_batches > 0 else 0
        logger.info("Av Loss per epoch=%f", epoch_loss)
        logger.info("epoch total correct predictions=%d", epoch_correct_predictions)

    def _save_checkpoint(
        self, scheduler, epoch: int, offset: int, checkpoint_file_name="checkpoint"
    ) -> str:

        model_to_save = get_model_obj(self.biencoder)
        cp = os.path.join(self.output_dir, checkpoint_file_name + "." + str(epoch))
        meta_params = get_encoder_params_state_from_cfg(
            do_lower_case=self.do_lower_case, encoder=model_to_save
        )
        state = CheckpointState(
            model_to_save.get_state_dict(),
            self.optimizer.state_dict(),
            scheduler.state_dict(),
            offset,
            epoch,
            meta_params,
        )
        torch.save(state._asdict(), cp)
        logger.info("Saved checkpoint at %s", cp)
        return cp

    def _load_saved_state(
        self,
        saved_state: CheckpointState,
        ignore_checkpoint_optimizer=False,
        ignore_checkpoint_offset=False,
    ):
        epoch = saved_state.epoch
        # offset is currently ignored since all checkpoints are made after full epochs
        offset = saved_state.offset
        if offset == 0:  # epoch has been completed
            epoch += 1
        logger.info("Loading checkpoint @ batch=%s and epoch=%s", offset, epoch)

        if ignore_checkpoint_offset:
            self.start_epoch = 0
            self.start_batch = 0
        else:
            self.start_epoch = epoch
            # TODO: offset doesn't work for multiset currently
            self.start_batch = 0  # offset

        model_to_load = get_model_obj(self.biencoder)
        logger.info("Loading saved model state ...")

        model_to_load.load_state(saved_state)

        if not ignore_checkpoint_optimizer:
            if saved_state.optimizer_dict:
                logger.info("Loading saved optimizer state ...")
                self.optimizer.load_state_dict(saved_state.optimizer_dict)

            if saved_state.scheduler_dict:
                self.scheduler_state = saved_state.scheduler_dict


def _calc_loss(
    loss_function,
    local_q_vector,
    local_ctx_vectors,
    local_positive_idxs,
    global_loss_buf_sz,
    local_rank,
    distributed_world_size=1,
    local_hard_negatives_idxs: list = None,
    loss_scale: float = None,
) -> Tuple[T, bool]:
    """
    Calculates In-batch negatives schema loss and supports to run it in DDP mode by exchanging the representations
    across all the nodes.
    """
    if distributed_world_size > 1:
        q_vector_to_send = (
            torch.empty_like(local_q_vector).cpu().copy_(local_q_vector).detach_()
        )
        ctx_vector_to_send = (
            torch.empty_like(local_ctx_vectors).cpu().copy_(local_ctx_vectors).detach_()
        )

        global_question_ctx_vectors = all_gather_list(
            [
                q_vector_to_send,
                ctx_vector_to_send,
                local_positive_idxs,
                local_hard_negatives_idxs,
            ],
            max_size=global_loss_buf_sz,
        )

        global_q_vector = []
        global_ctxs_vector = []

        # ctxs_per_question = local_ctx_vectors.size(0)
        positive_idx_per_question = []
        hard_negatives_per_question = []

        total_ctxs = 0

        for i, item in enumerate(global_question_ctx_vectors):
            q_vector, ctx_vectors, positive_idx, hard_negatives_idxs = item

            if i != local_rank:
                global_q_vector.append(q_vector.to(local_q_vector.device))
                global_ctxs_vector.append(ctx_vectors.to(local_q_vector.device))
                positive_idx_per_question.extend([v + total_ctxs for v in positive_idx])
                hard_negatives_per_question.extend(
                    [[v + total_ctxs for v in l] for l in hard_negatives_idxs]
                )
            else:
                global_q_vector.append(local_q_vector)
                global_ctxs_vector.append(local_ctx_vectors)
                positive_idx_per_question.extend(
                    [v + total_ctxs for v in local_positive_idxs]
                )
                hard_negatives_per_question.extend(
                    [[v + total_ctxs for v in l] for l in local_hard_negatives_idxs]
                )
            total_ctxs += ctx_vectors.size(0)
        global_q_vector = torch.cat(global_q_vector, dim=0)
        global_ctxs_vector = torch.cat(global_ctxs_vector, dim=0)

    else:
        global_q_vector = local_q_vector
        global_ctxs_vector = local_ctx_vectors
        positive_idx_per_question = local_positive_idxs
        hard_negatives_per_question = local_hard_negatives_idxs

    loss, is_correct = loss_function.calc(
        global_q_vector,
        global_ctxs_vector,
        positive_idx_per_question,
        hard_negatives_per_question,
        loss_scale=loss_scale,
    )

    return loss, is_correct


def _do_biencoder_fwd_pass(
    model: nn.Module,
    input: BiEncoderBatch,
    tensorizer: Tensorizer,
    encoder_type,
    gradient_accumulation_steps: int,
    n_gpu: int,
    local_rank: int,
    device: str,
    global_loss_buf_sz,
    rep_positions=0,
    loss_scale: float = None,
) -> Tuple[torch.Tensor, int]:

    input = BiEncoderBatch(**move_to_device(input._asdict(), device))

    q_attn_mask = tensorizer.get_attn_mask(input.question_ids)
    ctx_attn_mask = tensorizer.get_attn_mask(input.context_ids)

    if model.training:
        model_out = model(
            input.question_ids,
            input.question_segments,
            q_attn_mask,
            input.context_ids,
            input.ctx_segments,
            ctx_attn_mask,
            encoder_type=encoder_type,
            representation_token_pos=rep_positions,
        )
    else:
        with torch.no_grad():
            model_out = model(
                input.question_ids,
                input.question_segments,
                q_attn_mask,
                input.context_ids,
                input.ctx_segments,
                ctx_attn_mask,
                encoder_type=encoder_type,
                representation_token_pos=rep_positions,
            )

    local_q_vector, local_ctx_vectors = model_out

    loss_function = BiEncoderNllLoss()

    loss, is_correct = _calc_loss(
        loss_function=loss_function,
        local_q_vector=local_q_vector,
        local_ctx_vectors=local_ctx_vectors,
        local_hard_negatives_idxs=input.hard_negatives,
        local_positive_idxs=input.is_positive,
        local_rank=local_rank,
        loss_scale=loss_scale,
        global_loss_buf_sz=global_loss_buf_sz,
    )

    is_correct = is_correct.sum().item()

    if n_gpu > 1:
        loss = loss.mean()
    if gradient_accumulation_steps > 1:
        loss = loss / gradient_accumulation_steps
    return loss, is_correct

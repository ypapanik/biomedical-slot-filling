import argparse
import json
import logging
import os
from typing import Dict, Tuple, List

from biomedical_slot_filling.data_processing.utils import download_locally_if_needed
from biomedical_slot_filling.retrievers.bm25 import BM25
from biomedical_slot_filling.retrievers.dense_passage_retriever import DPR, indexers
from biomedical_slot_filling.retrievers.dpr.data.biencoder_data import load_data

from biomedical_slot_filling.retrievers.dpr.options import (
    setup_logger,
)
from biomedical_slot_filling.retrievers.dpr.utils.evaluate import calculate_matches

logger = logging.getLogger()
setup_logger(logger)


def validate(
    passages: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    match_stats = calculate_matches(
        passages, answers, result_ctx_ids, workers_num, match_type
    )
    top_k_hits = match_stats.top_k_hits

    logger.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    return match_stats.questions_doc_hits


def save_eval_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    # assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id.replace('pubmed:','')] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        merged_data.append(
            {
                "question": q,
                "answers": q_answers,
                "ctxs": [
                    {
                        "id": results_and_scores[0][c],
                        "title": docs[c][1],
                        "text": docs[c][0],
                        "score": scores[c],
                        "has_answer": hits[c],
                    }
                    for c in range(ctxs_num)
                ],
            }
        )
    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info("Saved results * scores  to %s", out_file)


def save_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    out_file: str,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    # assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id.replace('pubmed:','')] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(results_and_scores)

        merged_data.append(
            {
                "question": q,
                "ctxs": [
                    {
                        "id": results_and_scores[0][c],
                        "title": docs[c][1],
                        "text": docs[c][0],
                        "score": scores[c],
                    }
                    for c in range(ctxs_num)
                ],
            }
        )
    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info("Saved results * scores  to %s", out_file)


def get_questions(qa_dataset):
    questions = []
    answers = []
    questions_answers = {}
    logger.info("qa_dataset: %s", qa_dataset)
    with open(qa_dataset, 'r') as f:
        data = json.load(f)['data'][0]['paragraphs']
        for instance in data:
            for qa in instance['qas']:
                q = qa['question']
                ans = [a['text'] for a in qa['answers']]
                if not q in questions_answers.keys():
                    questions_answers[q] = []
                questions_answers[q] += ans

    for q, ans in questions_answers.items():
        questions.append(q)
        answers.append(ans)
    return questions, answers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval"
    )
    parser.add_argument("--method", default='dpr', help="method to evaluate")
    parser.add_argument("--topn", default=100)
    parser.add_argument("--data-indices-dir",
                        default='https://drive.google.com/drive/folders/1dwhfvl7zy6BEGhPWVAFBTOAg8YYd_jQz')
    parser.add_argument("--index-type", default="flat")
    parser.add_argument(
        "--predictions-json",
        default="predictions.json",
        help="where to write predictions",
    )
    parser.add_argument("--query-encoder", default="healx/biomedical-dpr-qry-encoder")

    args = parser.parse_args()
    data_indices_dir = download_locally_if_needed(args.data_indices_dir)
    qa_dataset = os.path.join(data_indices_dir, 'biomedical_slot_filling_dev.json')
    index_path = os.path.join(data_indices_dir, 'biomedical-dpr-index-1m')
    passages = os.path.join(data_indices_dir, '1m_abstracts_with_gold_passages_included.tsv')

    questions, question_answers = get_questions(qa_dataset=qa_dataset)

    if args.method == "bm25":
        model = BM25()
        model.load_index(index_path=index_path)
        top_ids_and_scores = model.query(questions=questions, n=int(args.topn))
    elif args.method == "dpr":
        model = DPR()
        model.load_query_model(query_encoder=args.query_encoder)
        model.load_index(
            index_path=index_path,
            index=indexers[args.index_type],
        )
        top_ids_and_scores = model.query(questions=questions, n=int(args.topn),
                                         del_index_after_use=True, batch_size=512)
    else:
        raise ValueError('Wrong method, can either be bm25 or dpr.')


    all_passages = load_data(ctx_src=passages, return_as_list_of_tuples=False)
    out_file = args.predictions_json

    if question_answers:
        questions_doc_hits = validate(
            passages=all_passages,
            answers=question_answers,
            result_ctx_ids=top_ids_and_scores,
            match_type="string",
            workers_num=16,
        )
        save_eval_results(
            all_passages,
            questions,
            question_answers,
            top_ids_and_scores,
            questions_doc_hits,
            out_file,
        )
    else:
        save_results(
            all_passages,
            questions,
            top_ids_and_scores,
            out_file,
        )

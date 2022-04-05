import argparse
import logging
import os

import pandas

from biomedical_slot_filling.data_processing.squad_questions import write_questions_and_context_in_squad_json
from biomedical_slot_filling.data_processing.utils import download_locally_if_needed
from biomedical_slot_filling.reader.reader import QuestionAnsweringModel
from biomedical_slot_filling.retrievers.bm25 import BM25
from biomedical_slot_filling.retrievers.dense_passage_retriever import DPR, indexers
from biomedical_slot_filling.retrievers.dpr.data.biencoder_data import load_data
from biomedical_slot_filling.retrievers.dpr.options import (
    setup_logger,
)

logger = logging.getLogger()
setup_logger(logger)


def read_triples(triples_tsv):
    df = pandas.read_csv(triples_tsv, sep='\t')
    questions = []
    questions_answers = {}
    for i,row in df.iterrows():
        q = row['head'] + '[SEP]' + row['relation']
        a = row['tail']
        questions.append(q)
        if not q in questions_answers.keys():
            questions_answers[q] = []
        questions_answers[q].append(a)
    return questions, questions_answers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate end to end"
    )
    parser.add_argument("--data-indices-dir",
                        default='https://drive.google.com/drive/folders/1dwhfvl7zy6BEGhPWVAFBTOAg8YYd_jQz')
    parser.add_argument("--method", required=True, help="method to evaluate")
    parser.add_argument("--topn", default=100)
    parser.add_argument("--index-type", default="flat")
    parser.add_argument("--reader", default='healx/biomedical-slot-filling-reader-base', help='reader model')
    parser.add_argument("--model-type", help="QA model", default='bert')
    parser.add_argument("--eval-batch-size", default=64)
    parser.add_argument(
        "--predictions-json",
        default="predictions.json",
        help="where to write predictions",
    )
    parser.add_argument("--query-encoder", default="healx/biomedical-dpr-qry-encoder")
    args = parser.parse_args()

    data_indices_dir = download_locally_if_needed(args.data_indices_dir)
    triples = os.path.join(data_indices_dir, 'biosf_triples_test.tsv')
    index_path = os.path.join(data_indices_dir, 'biomedical-dpr-index-1m')
    passages = os.path.join(data_indices_dir, '1m_abstracts_with_gold_passages_included.tsv')

    questions, questions_answers_dict = read_triples(triples_tsv=triples)

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
        top_ids_and_scores = model.query(questions=questions, n=int(args.topn), del_index_after_use=True,
                                         batch_size=64)
    else:
        raise ValueError('Wrong method, can either be bm25 or dpr.')

    all_passages = load_data(ctx_src=passages, return_as_list_of_tuples=False)
    results_with_passages = {}
    for question, results_per_question in zip(questions, top_ids_and_scores):
        results_with_passages_per_doc = []
        for docid in results_per_question[0]:
            docid = docid.replace('pubmed:', '')
            passage = all_passages[docid]
            results_with_passages_per_doc.append(
                (docid, passage.title + ' ' + passage.text)
        )
        results_with_passages[question] = results_with_passages_per_doc
    all_passages = None
    model.retriever = None
    model = None
    output_temp_file = '/tmp/question_contexts.json'
    write_questions_and_context_in_squad_json(questions=questions, results=results_with_passages,
                                              output_file=output_temp_file)

    qa = QuestionAnsweringModel(pretrained_model_name_or_path=args.reader,
                                model_type=args.model_type,
                                qa_model_path=args.reader,
                                local_rank=-1,
                                fp16=False,
                                no_cuda=False,
                                cache_dir='data/pretrained/',
                                do_lower_case=True,
                                max_query_length=16,
                                max_seq_length=160,
                                doc_stride=128
                                )

    results = qa.predict(predict_file=output_temp_file,
                 per_gpu_eval_batch_size=int(args.eval_batch_size),
                 output_prediction_file='predictions.json',
                 output_nbest_file='n_best.json',
                 n_best_size=10
               )
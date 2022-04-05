import argparse

import logging
import os

from biomedical_slot_filling.data_processing.utils import natural_key
from biomedical_slot_filling.retrievers.dense_passage_retriever import DPR, indexers
from biomedical_slot_filling.retrievers.dpr.options import setup_logger

logger = logging.getLogger()
setup_logger(logger)

def possible_encoders():
    encoders = ["facebook/dpr-ctx_encoder-multiset-base",
            "facebook/dpr-question_encoder-multiset-base",
            "facebook/dpr-ctx_encoder-single-nq-base",
            "facebook/dpr-question_encoder-single-nq-base",
            "michaelrglass/dpr-ctx_encoder-multiset-base-kgi0-zsre",
            "michaelrglass/dpr-ctx_encoder-multiset-base-kgi0-trex",
            "healx/biomedical-dpr-qry-encoder",
            "healx/biomedical-dpr-ctx-encoder"]
    return encoders

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode passages with DPR")
    parser.add_argument("--passages", required=True, help="passages file")
    parser.add_argument("--context-encoder",
                        default="healx/biomedical-dpr-ctx-encoder",
                        help="context-encoder")
    parser.add_argument("--index-type", default="flat")
    parser.add_argument(
        "--encoded-content-dir",
        required=True,
        help="where to write the encoded data",
    )
    parser.add_argument("--index-path", required=True)

    args = parser.parse_args()

    dpr = DPR()
    dpr.load_context_model(context_encoder=args.context_encoder)

    dpr.encode_context(
        ctx_src=args.passages,
        out_dir=args.encoded_content_dir,
        batch_size=8
    )

    dpr.build_index(
        index=indexers[args.index_type],
        encoded_files=sorted([os.path.join(args.encoded_content_dir,f )
                              for f in os.listdir(args.encoded_content_dir)], key=natural_key),
        index_path=args.index_path,
        qa_selector=None,
    )

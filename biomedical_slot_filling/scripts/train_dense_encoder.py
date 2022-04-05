import argparse
import logging

from biomedical_slot_filling.retrievers.dense_passage_retriever import DPR
from biomedical_slot_filling.retrievers.dpr.options import setup_logger

logger = logging.getLogger()
setup_logger(logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train DPR")
    parser.add_argument("--training-file", required=True, help="training file")
    parser.add_argument("--validation-file", required=False, help="dev file")
    parser.add_argument("--output-dir", required=True, help="output dir")
    parser.add_argument("--training-batch-size", default=4)
    parser.add_argument("--num-epochs", default=40)
    parser.add_argument("--gradient-accumulation-steps", default=1)
    parser.add_argument("--lr", default=3e-5)
    args = parser.parse_args()

    dpr = DPR()
    dpr.train(
        output_dir=args.output_dir,
        training_file=args.training_file,
        validation_file=args.validation_file,
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        batch_size=int(args.training_batch_size),
        num_train_epochs=int(args.num_epochs),
        lr=float(args.lr),
    )

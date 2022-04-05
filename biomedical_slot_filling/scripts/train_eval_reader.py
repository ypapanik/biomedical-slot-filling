import argparse
import os
from biomedical_slot_filling.data_processing.utils import download_locally_if_needed
from biomedical_slot_filling.reader.reader import QuestionAnsweringModel



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data-indices-dir",
                        default='https://drive.google.com/drive/folders/1dwhfvl7zy6BEGhPWVAFBTOAg8YYd_jQz')
    parser.add_argument("--model-type", default='bert',
                        help="Model type")
    parser.add_argument("--model-name-or-path",
                        default='healx/biomedical-slot-filling-reader-base',
                        help="Path to pre-trained model or shortcut name. If training, "
                             "this should be a biobert model")
    parser.add_argument("--output-dir", default='biosf-reader',
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--training-batch-size", default=32)
    parser.add_argument("--gradient-accumulation-steps", default=1)
    parser.add_argument("--test-batch-size", default=64)
    parser.add_argument("--epochs", default=10)
    parser.add_argument("--lr", default=3e-5)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--eval", action='store_true')
    args = parser.parse_args()

    data_indices_dir = download_locally_if_needed(args.data_indices_dir)
    training_file = os.path.join(data_indices_dir, 'biomedical_slot_filling_train.json')
    dev_file = os.path.join(data_indices_dir, 'biomedical_slot_filling_dev.json')
    test_file = os.path.join(data_indices_dir, 'biomedical_slot_filling_test.json')

    output_dir='reader_model'

    qa = QuestionAnsweringModel(pretrained_model_name_or_path=args.model_name_or_path,
                                model_type=args.model_type,
                                qa_model_path=output_dir,
                                local_rank=-1,
                                fp16=False,
                                no_cuda=False,
                                cache_dir='data/pretrained/',
                                do_lower_case=True,
                                max_query_length=16,
                                max_seq_length=200,
                                doc_stride=128
                                )

    # Training
    if args.train:
        qa.train(train_file=training_file,
                 eval_file=dev_file,
                 learning_rate=float(args.lr),
                 num_train_epochs=int(args.epochs),
                 per_gpu_train_batch_size=int(args.training_batch_size),
                 gradient_accumulation_steps=int(args.gradient_accumulation_steps),
                 )

    if args.eval:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        result = qa.predict(predict_file=test_file, per_gpu_eval_batch_size=int(args.test_batch_size))
        print(result)
import argparse
import json
import os

from biomedical_slot_filling.retrievers.bm25 import BM25
from biomedical_slot_filling.retrievers.dpr.data.biencoder_data import normalize_passage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode passages with bm25")
    parser.add_argument("--passages", required=True, help="passages file")
    args = parser.parse_args()

    if not os.path.exists('passages'):
        os.mkdir('passages')
    with open(args.passages, 'r') as fr, open('passages/passages.json', 'w') as fw:
        for i, line in enumerate(fr):
            if i == 0:
                continue
            pmid, title, text = line.strip().split('\t')
            text = normalize_passage(text)
            fw.write(json.dumps({'id': pmid,
                                 'contents': title + '\n\n' + text}) + '\n')

    bm25 = BM25()
    bm25.build_index(collection_path='passages/', index_path='bm25_index')

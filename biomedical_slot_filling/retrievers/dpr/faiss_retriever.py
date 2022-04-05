import logging
import pickle
import time
from typing import List, Tuple, Iterator

import numpy as np

from biomedical_slot_filling.retrievers.dpr.options import setup_logger
from biomedical_slot_filling.retrievers.dpr.dense_retriever import DenseRetriever
from biomedical_slot_filling.retrievers.dpr.indexer.faiss_indexers import (
    DenseIndexer,
)

logger = logging.getLogger()
setup_logger(logger)


def iterate_encoded_files(
    vector_files: list, path_id_prefixes: List = None
) -> Iterator[Tuple]:
    for i, file in enumerate(vector_files):
        logger.info("Reading file %s", file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            if type(doc_vectors) == list:
                for pmid, doc in doc_vectors:
                    yield (pmid, doc.astype(np.float32))
            else:
                for pmid, doc in doc_vectors.items():
                    yield (pmid, doc.astype(np.float32))

class LocalFaissRetriever(DenseRetriever):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
            self,
            index: DenseIndexer,
            question_encoder=None,
            batch_size=16,
            tensorizer=None,
    ):
        super().__init__(question_encoder, batch_size, tensorizer)
        self.index = index

    def index_encoded_data(
        self,
        vector_files: List[str],
        buffer_size: int,
        path_id_prefixes: List = None,
    ):
        """
        Indexes encoded passages takes form a list of files
        :param vector_files: file names to get passages vectors from
        :param buffer_size: size of a buffer (amount of passages) to send for the indexing at once
        :return:
        """
        buffer = []
        for i, item in enumerate(
            iterate_encoded_files(vector_files, path_id_prefixes=path_id_prefixes)
        ):
            buffer.append(item)
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        self.index.index_data(buffer)
        logger.info("Data indexing completed.")

    def get_top_docs(
        self, query_vectors: np.array, top_docs: int = 100, del_index_after_use=False
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info("index search time: %f sec.", time.time() - time0)
        if del_index_after_use:
            self.index = None
        return results

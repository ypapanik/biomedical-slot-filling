import os
import subprocess
from pyserini.search import SimpleSearcher

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ['ANSERINI_CLASSPATH'] = 'anserini/target'


class BM25:
    def load_index(self, index_path):
        self.searcher = SimpleSearcher(index_dir=index_path)
        self.searcher.set_bm25(0.9, 0.4)
        self.searcher.set_rm3()

    def query(self, questions, n):
        results = []
        for question in questions:
            result = self.searcher.search(question, k=n)
            results.append(
                (
                    [res.docid for res in result],
                    [res.score for res in result]
                )
            )
        return results

    def build_index(
            self,
            collection_path,
            index_path,
            nthreads=6
    ):
        cmd = './anserini/target/appassembler/bin/IndexCollection -storeContents -collection JsonCollection -generator DefaultLuceneDocumentGenerator -storePositions -storeDocvectors -input ' + str(collection_path) + ' -index ' + str(index_path) + ' -threads ' + str(nthreads)
        subprocess.run(cmd, shell=True, check=True)


import os

import pandas as pd
from Bio import SeqIO

from src.tools.random_seed import RandomSeed
from src.training_and_evaluation.embeddings_generator import EmbeddingsGenerator


class GenerateEmbeddings:

    @staticmethod
    def read_sequences(dir_path: str, file_name: str, seq_col: str = "sequence") -> pd.DataFrame:
        records = list(SeqIO.parse(f"{dir_path}/{file_name}", "fasta"))
        data = {
            "id": [record.id for record in records],
            seq_col: [str(record.seq) for record in records],
        }
        return pd.DataFrame(data)

    @staticmethod
    def generate_benchmark_embeddings(benchmark_dir_path: str,
                                      output_dir: str,
                                      seq_file_name: str,
                                      output_file_name: str):
        seq_with_idx_df, _ = embeddings_generator.generate_embeddings(
            df=GenerateEmbeddings.read_sequences(
                dir_path=benchmark_dir_path,
                file_name=seq_file_name,
                seq_col="sequence"
            ),
            seq_col="sequence",
            output_file_path=f"{output_dir}/{output_file_name}",
        )
        print(seq_with_idx_df.shape)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_generator: EmbeddingsGenerator = EmbeddingsGenerator()

    generate_benchmark = True
    generate_general = True

    # region generate benchmark embeddings
    if generate_benchmark:
        benchmark_dir_path = os.path.join(base_dir, "../..", "Dataset", "Benchmark_dataset")
        output_dir = os.path.join(base_dir, "../..", "Dataset", "Benchmark_dataset", "new")

        benchmark_embeddings_map = {
            "Pos-Test.txt": "esmc_pos_test.npy",
            "Pos-Train.txt": "esmc_pos_train.npy",
            "Neg-Test.txt": "esmc_neg_test.npy",
            "Neg-Train.txt": "esmc_neg_train.npy",
        }
        for seq_file_name, output_file_name in benchmark_embeddings_map.items():
            RandomSeed.random_seed(777)
            GenerateEmbeddings.generate_benchmark_embeddings(
                benchmark_dir_path=benchmark_dir_path,
                output_dir=output_dir,
                seq_file_name=seq_file_name,
                output_file_name=output_file_name,
            )
    # end region

    # region generate external general dataset embedding
    if generate_general:
        general_dir_path = os.path.join(base_dir, "../..", "Dataset", "Generalized_dataset")
        output_dir = os.path.join(base_dir, "../..", "Dataset", "Generalized_dataset", "new")
        generalized_embeddings_map = {
            "Neg_test_dataset.fasta": "esmc_neg_test.npy",
            "Neg_train_dataset.fasta": "esmc_neg_train.npy",
            "Pos_test_dataset.fasta": "esmc_pos_test.npy",
            "Pos_train_dataset.fasta": "esmc_pos_train.npy"
        }
        for seq_file_name, output_file_name in generalized_embeddings_map.items():
            RandomSeed.random_seed(777)
            GenerateEmbeddings.generate_benchmark_embeddings(
                benchmark_dir_path=general_dir_path,
                output_dir=output_dir,
                seq_file_name=seq_file_name,
                output_file_name=output_file_name,
            )
    # end region

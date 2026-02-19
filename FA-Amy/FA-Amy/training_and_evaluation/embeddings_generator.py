import os
import warnings

import numpy as np
import pandas as pd
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMCInferenceClient, ESMProtein, LogitsConfig, LogitsOutput
from tqdm import tqdm


class EmbeddingsGenerator:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_embeddings(self,
                            df: pd.DataFrame,
                            seq_col: str,
                            output_file_path: str,
                            max_seq_length: int = 900,
                            row_col_name: str = "embedding_row_idx"):
        """
        Generate per-token embeddings for df[seq_col] and write them incrementally to output_file_path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_file_path)), exist_ok=True)
        model = ESMC.from_pretrained("esmc_600m").to(self.device)

        df = df.copy()
        sequences = df[seq_col].astype(str).tolist()

        n = len(sequences)
        if n == 0:
            raise ValueError("DataFrame is empty after preprocessing; no sequences to embed.")

        # Before embedding is generated we probe the first embedding and allocate the output file
        first_sequence_padded = EmbeddingsGenerator.pad_sequence(sequences[0], max_seq_length)
        first_embedding = EmbeddingsGenerator.generate_single_embedding(
            seq=first_sequence_padded,
            client=model
        ).squeeze().detach().cpu().numpy().astype(np.float32, copy=False)

        length, embedding_dim = first_embedding.shape
        if length != max_seq_length:
            warnings.warn(f"Expected L={max_seq_length}, got L={length}. Check padding.")

        mm = np.lib.format.open_memmap(
            output_file_path,
            mode="w+",
            dtype=np.float32,
            shape=(n, length, embedding_dim),
        )
        mm[0] = first_embedding

        # Write the rest incrementally
        for i in tqdm(range(1, n), desc="Embedding sequences", unit="seq"):
            padded = EmbeddingsGenerator.pad_sequence(sequences[i], max_seq_length)
            embedding = EmbeddingsGenerator.generate_single_embedding(
                seq=padded,
                client=model
            ).squeeze().detach().cpu().numpy().astype(np.float32, copy=False)
            mm[i] = embedding
        del mm

        df[row_col_name] = np.arange(n, dtype=np.int64)
        return df, output_file_path

    @staticmethod
    def pad_sequence(seq: str, max_len: int, pad_char: str = "X"):
        return seq[:max_len] + pad_char * max(0, max_len - len(seq))

    @staticmethod
    def generate_single_embedding(seq: str, client: ESMCInferenceClient, show_log: bool = False):
        protein = ESMProtein(seq)
        protein_tensor = client.encode(protein)
        output = client.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )

        assert isinstance(output, LogitsOutput)
        assert output.logits is not None and output.logits.sequence is not None
        assert output.embeddings is not None

        if show_log:
            print(
                f"Client returned logits with shape: {output.logits.sequence.shape} and embeddings with shape: {output.embeddings.shape}"
            )
        return output.embeddings

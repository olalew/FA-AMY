import numpy as np
from esm.models.esmc import ESMC
from esm.sdk.api import ESMCInferenceClient, ESMProtein, LogitsConfig, LogitsOutput
from Bio import SeqIO
import os

from tqdm import tqdm


def main(client: ESMCInferenceClient, seq):
    protein = ESMProtein(seq)
    protein_tensor = client.encode(protein)
    output = client.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    assert isinstance(output, LogitsOutput)
    assert output.logits is not None and output.logits.sequence is not None
    assert output.embeddings is not None
    print(
        f"Client returned logits with shape: {output.logits.sequence.shape} and embeddings with shape: {output.embeddings.shape}"
    )
    return output.embeddings  # shape: (sequence_length, embedding_dim)


def pad_sequence(seq: str, max_len: int, pad_char: str = "X"):
    return seq[:max_len] + pad_char * max(0, max_len - len(seq))


def generate_embeddings(input_file_name: str, output_file_name: str):
    model = ESMC.from_pretrained("esmc_600m")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "FA-Amy", "../Dataset", "Benchmark_dataset", input_file_name)
    records = list(SeqIO.parse(data_path, "fasta"))
    sequences = [str(record.seq) for record in records]

    embedding_list = []
    max_length = 900  # Expected maximum sequence length
    count = 0

    print(f"Loaded {len(sequences)} sequences")

    for seq in tqdm(sequences):
        padded_seq = pad_sequence(seq, max_length)
        emb = main(model, padded_seq).squeeze().cpu().detach().numpy()  # emb shape: (L, D)
        seq_len, emb_dim = emb.shape
        count += 1
        embedding_list.append(emb)

    embedding_array = np.stack(embedding_list)  # shape: (N, max_length, emb_dim)

    output_dir = os.path.join(base_dir, "FA-Amy", "../Dataset", "Benchmark_dataset", "new")
    np.save(f"{output_dir}/{output_file_name}", embedding_array)


if __name__ == "__main__":
    generate_embeddings(
        input_file_name="Pos-Test.txt",
        output_file_name="esmc_pos_test.npy",
    )
    generate_embeddings(
        input_file_name="Pos-Train.txt",
        output_file_name="esmc_pos_train.npy",
    )
    generate_embeddings(
        input_file_name="Neg-Test.txt",
        output_file_name="esmc_neg_test.npy",
    )
    generate_embeddings(
        input_file_name="Neg-Train.txt",
        output_file_name="esmc_neg_train.npy",
    )

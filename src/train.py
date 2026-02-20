import os

from tools.random_seed import RandomSeed
from training_and_evaluation.model_trainer import ModelTrainer

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))

    train_benchmark: bool = True
    train_general: bool = True

    # region train benchmark
    if train_benchmark:
        RandomSeed.random_seed(777)
        neg_data_path = os.path.join(base_dir, "..", "Dataset", "Benchmark_dataset", "new", "esmc_neg_train.npy")
        pos_data_path = os.path.join(base_dir, "..", "Dataset", "Benchmark_dataset", "new", "esmc_pos_train.npy")
        model_path = os.path.join(base_dir, "..", "Model-saved", "Model-Saved-Benchmark")

        trainer: ModelTrainer = ModelTrainer(
            pos_data_path=pos_data_path,
            neg_data_path=neg_data_path
        )

        trainer.train(model_path=model_path)
    # end region

    # region train general
    # Processed dataset (originated from https://github.com/KOALA-L/ECAmyloid/blob/main/README.md)
    if train_general:
        RandomSeed.random_seed(777)
        neg_data_path = os.path.join(base_dir, "..", "Dataset", "Generalized_dataset", "new",
                                     "esmc_neg_train.npy")
        pos_data_path = os.path.join(base_dir, "..", "Dataset", "Generalized_dataset", "new",
                                     "esmc_pos_train.npy")
        model_path = os.path.join(base_dir, "..", "Model-saved", "Model-Saved-General")

        trainer: ModelTrainer = ModelTrainer(
            pos_data_path=pos_data_path,
            neg_data_path=neg_data_path
        )

        trainer.train(model_path=model_path)
    # end region

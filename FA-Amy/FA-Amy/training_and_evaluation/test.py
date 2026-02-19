import os

from tools.random_seed import RandomSeed
from training_and_evaluation.model_tester import ModelTester

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))

    test_benchmark: bool = False
    test_general: bool = True

    # region train benchmark
    if test_benchmark:
        RandomSeed.random_seed(777)
        neg_data_path = os.path.join(base_dir, "..", "..", "Dataset", "Benchmark_dataset", "new", "esmc_neg_test.npy")
        pos_data_path = os.path.join(base_dir, "..", "..", "Dataset", "Benchmark_dataset", "new", "esmc_pos_test.npy")
        model_path = os.path.join(base_dir, "..", "..", "Model-saved", "Model-Saved-Benchmark")

        tester: ModelTester = ModelTester(
            pos_data_path=pos_data_path,
            neg_data_path=neg_data_path
        )

        tester.test(model_path=model_path)
    # end region

    # region train general
    # Processed dataset (originated from https://github.com/KOALA-L/ECAmyloid/blob/main/README.md)
    if test_general:
        neg_data_path = os.path.join(base_dir, "..", "..", "Dataset", "Generalized_dataset", "new",
                                     "esmc_neg_test.npy")
        pos_data_path = os.path.join(base_dir, "..", "..", "Dataset", "Generalized_dataset", "new",
                                     "esmc_pos_test.npy")
        model_path = os.path.join(base_dir, "..", "..", "Model-saved", "Model-Saved-General")

        tester: ModelTester = ModelTester(
            pos_data_path=pos_data_path,
            neg_data_path=neg_data_path
        )

        tester.test(model_path=model_path)
    # end region

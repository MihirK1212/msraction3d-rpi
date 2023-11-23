import warnings
import timeit
import random

import data.read as data_read
import data.utils as data_utils
import constants
from multimethod_ensemble import MultimethodEnsemble


def get_avg_len(sequences):
    lens = [sequence.shape[0] for sequence in sequences]
    return sum(lens) / len(lens)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        num_iters = 3
        execution_time = timeit.timeit(
            lambda: data_read.read_msr_data(data_dir=constants.MSR_ACTION_3D_DATA_DIR),
            number=num_iters,
        )
        execution_time /= num_iters
        print(f"Execution time for reading data: {execution_time} seconds")

        data, labels, subjects = data_read.read_msr_data(
            data_dir=constants.MSR_ACTION_3D_DATA_DIR
        )
        data, labels, subjects = data_utils.remove_anomalies(
            data=data, labels=labels, subjects=subjects
        )

        total_samples = len(data)

        multimethod_ensemble = MultimethodEnsemble(training=False)
        multimethod_ensemble.load_training_config()

        data_curr = data[0:5]
        multimethod_ensemble.get_predictions(data_curr)
        print("Loaded Models")

        with open("output.txt", "w+") as f:
            for num_test_samples in [1, 5, 10, 20, 50, 100]:
                for _ in range(5):
                    lb = random.randint(0, total_samples - num_test_samples - 1)
                    assert (
                        lb >= 0
                        and lb < total_samples
                        and (lb + num_test_samples) < total_samples
                    )

                    data_curr, labels_curr, subjects_curr = (
                        data[lb : (lb + num_test_samples)],
                        labels[lb : (lb + num_test_samples)],
                        subjects[lb : (lb + num_test_samples)],
                    )

                    assert len(data_curr) == num_test_samples
                    assert len(labels_curr) == num_test_samples
                    assert len(subjects_curr) == num_test_samples

                    avg_len = get_avg_len(data_curr)

                    num_iters = 3
                    execution_time = timeit.timeit(
                        lambda: multimethod_ensemble.get_predictions(data_curr),
                        number=num_iters,
                    )
                    execution_time /= num_iters
                    print(
                        f"Execution time for {num_test_samples} samples and avg len {avg_len}: {execution_time} seconds"
                    )
                    f.write(f"{num_test_samples} {avg_len} {execution_time}\n")

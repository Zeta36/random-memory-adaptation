from __future__ import division

import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from RMA import RMA


def plot_results(num_tasks_to_run, baseline_mlp, memoryadapted):
    import matplotlib.pyplot as plt
    tasks = range(1, num_tasks_to_run + 1)
    plt.plot(tasks, baseline_mlp[::-1])
    plt.plot(tasks, memoryadapted[::-1])
    plt.legend(["Baseline-MLP", "RMA"], loc='lower right')
    plt.xlabel("Number of Tasks")
    plt.ylabel("Accuracy (%)")
    plt.ylim([1, 100])
    plt.xticks(tasks)
    plt.show()


def main(_):

    with tf.Session() as sess:

        print("\nParameters used:", args, "\n")

        # We create the baseline model
        baseline_model = RMA(sess, args)
        # We create the memory adapted model
        rma_model = RMA(sess, args)

        # Permuted MNIST
        # Generate the tasks specifications as a list of random permutations of the input pixels.
        mnist = input_data.read_data_sets("/tmp/", one_hot=True)
        task_permutation = []
        for task in range(args.num_tasks_to_run):
            task_permutation.append(np.random.permutation(784))

        print("\nBaseline MLP training...")
        start = time.time()
        last_performance_baseline = training(baseline_model, mnist, task_permutation, False)
        end = time.time()
        time_needed_baseline = round(end - start)
        print("Training time elapsed: ", time_needed_baseline, "s")

        print("\nMemory adapted (RMA) training...")
        start = time.time()
        last_performance_ma = training(rma_model, mnist, task_permutation, True)
        end = time.time()
        time_needed_ma = round(end - start)
        print("Training time elapsed: ", time_needed_ma, "s")

        # Stats
        print("\nDifference in time between using or not memory: ", time_needed_ma - time_needed_baseline, "s")
        print("Test accuracy mean gained due to the memory: ",
              np.round(np.mean(last_performance_ma) - np.mean(last_performance_baseline), 2))

        # Plot the results
        plot_results(args.num_tasks_to_run, last_performance_baseline, last_performance_ma)


def training(model, mnist, task_permutation, use_memory=True):
    # Training the model using or not memory adaptation
    last_performance = []
    for task in range(args.num_tasks_to_run):
        print("\nTraining task: ", task + 1, "/", args.num_tasks_to_run)

        for i in range(int(10000)):
            # Permute batch elements
            batch = mnist.train.next_batch(args.batch_size)
            batch = (batch[0][:, task_permutation[task]], batch[1])

            if use_memory:
                model.train(batch[0], batch[1], args.batch_size)
                # We just store a batch sample each args.memory_each steps
                if i % args.memory_each == 0:
                    model.add_to_memory(batch[0], batch[1])
            else:
                model.train(batch[0], batch[1], 0)

        # Print test set accuracy to each task encountered so far
        for test_task in range(task + 1):
            test_images = mnist.test.images

            # Permute batch elements
            test_images = test_images[:, task_permutation[test_task]]

            acc = model.test(test_images, mnist.test.labels)
            acc = acc * 100

            if args.num_tasks_to_run == task + 1:
                last_performance.append(acc)

            print("Testing, Task: ", test_task + 1, " \tAccuracy: ", acc)

    return last_performance


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_tasks_to_run', type=int, default=20,
                        help='Number of task to run')
    parser.add_argument('--memory_size', type=int, default=15000,
                        help='Memory size')
    parser.add_argument('--memory_each', type=int, default=1000,
                        help='Add to memory after these number of steps')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Size of batch for updates')
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='Learning rate')

    args = parser.parse_args()

    tf.app.run()

from __future__ import division

import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class RandomMemory:
    def __init__(self, capacity, input_result, output_dimension):
        self.capacity = capacity
        self.states = np.zeros((capacity, input_result))
        self.values = np.zeros((capacity, output_dimension))
        # Pointer
        self.curr_capacity = 0

    def nn(self, n_samples):
        # We seek and return n random memory locations
        idx = np.random.choice(np.arange(len(self.states)), n_samples, replace=False)
        embs = self.states[idx]
        values = self.values[idx]

        return embs, values

    def add(self, keys, values):
        # We add {k, v} pairs to the memory
        if self.curr_capacity + len(keys) >= self.capacity:
            self.curr_capacity = 0

        for i, _ in enumerate(keys):
            self.states[self.curr_capacity] = keys[i]
            self.values[self.curr_capacity] = values[i]
            self.curr_capacity += 1


class RMA:
    def __init__(self, session, args):
        self.learning_rate = args.learning_rate
        self.session = session

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        self.memory_sample_batch = tf.placeholder(tf.int16, shape=())

        # Memory Sampling
        embs_and_values = tf.py_func(self.get_memory_sample, [self.memory_sample_batch], [tf.float64, tf.float64])
        self.memory_batch_x = tf.to_float(embs_and_values[0])
        self.memory_batch_y = tf.to_float(embs_and_values[1])
        self.xa = tf.concat(values=[self.x, self.memory_batch_x], axis=0)
        self.ya_ = tf.concat(values=[self.y_, self.memory_batch_y], axis=0)

        # Network
        self.y = self.network(self.xa)

        # Memory M
        self.M = RandomMemory(args.memory_size, self.x.get_shape()[-1], self.y.get_shape()[-1])

        # Loss function
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ya_, logits=self.y))
        self.optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # Initialize the variables
        session.run(tf.global_variables_initializer())

    def get_memory_sample(self, batch_size):
        # Return the embeddings and values sample from memory
        x, y_ = self.M.nn(batch_size)
        return x, y_

    def add_to_memory(self, xs, ys):
        # Adds the given embedding to the memory
        self.M.add(xs, ys)

    def train(self, xs, ys, memory_sample_batch):
        self.session.run(self.optim, feed_dict={self.x: xs, self.y_: ys, self.memory_sample_batch: memory_sample_batch})

    def test(self, xs_test, ys_test):
        acc = self.session.run(self.accuracy,
                               feed_dict={self.x: xs_test, self.y_: ys_test, self.memory_sample_batch: 0})
        return acc

    @staticmethod
    def network(x):
        # Basic 2 layers MLP
        w = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        y = tf.matmul(x, w) + b
        return y


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

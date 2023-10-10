from math import log2
from logger import SummaryWriter
from skimage import filters
import utils
import numpy as np
import matplotlib.pyplot as plt

from model import nn_one_layer, nn_autoencoder


class Trainer:
    def __init__(
        self,
        model,
        data,
        labels,
        data_test,
        labels_test,
        summary_writer,
        batch_size=256,
        weight_transport=True,
        activation_derivative=True,
        prob_not_backprop=0,
    ):
        self.model = model
        self.data = data
        self.labels = labels
        self.data_test = data_test
        self.labels_test = labels_test
        self.batch_size = batch_size
        self.weight_transport = weight_transport
        self.activation_derivative = activation_derivative
        self.prob_not_backprop = prob_not_backprop
        self.summary_writer = summary_writer

    def generate_batch(self, inputs, targets):
        assert len(inputs) == len(targets)
        rng = np.random.default_rng()
        rand_inds = rng.choice(
            np.arange(0, len(inputs)), size=self.batch_size, replace=False
        )
        inputs_batch = inputs[rand_inds]
        targets_batch = targets[rand_inds]
        return inputs_batch, targets_batch

    def bin_average(self, activity: np.array) -> np.array:
        average_activity = np.mean(np.abs(activity), axis=1).reshape((-1, 1))
        binned_average_activity = np.ones(activity.shape)
        binned_average_activity[activity < average_activity] = 0

        return binned_average_activity

    def train_one_batch(self, lr, step):
        inputs, targets = self.generate_batch(self.data, self.labels)
        preds, H, Z = self.model.forward(inputs)
        # binary representation of hidden layer activity for whole batch
        bin_hidden = self.bin_average(H)
        bin_output = self.bin_average(preds)

        activity_strings = []  # string binary representation of hidden layer activity
        output_strings = []  # string one-hot label representation for each batch item
        target_strings = []  # string one-hot label representation for each batch item
        corner_strings = []  # string 0-3 representation of corner for each batch item
        for i in range(0, self.batch_size):
            activity_strings.append("".join(map(str, bin_hidden[i].astype(int))))
            output_strings.append("".join(map(str, bin_output[i].astype(int))))
            target_strings.append("".join(map(str, targets[i].astype(int))))
            corner_strings.append(str(brightest_corner(inputs[i])))

        (HY_X, IY_X) = self.calculate_HX_Y_IX_Y(activity_strings, target_strings)
        (HX_Y, IY_X) = self.calculate_HX_Y_IX_Y(target_strings, activity_strings)
        self.summary_writer.add_scalars("H(Y|X) train", HY_X, step)
        self.summary_writer.add_scalars("H(X|Y) train", HX_Y, step)
        self.summary_writer.add_scalars("I(Y,X) train", IY_X, step)

        (HZ_X, IZ_X) = self.calculate_HX_Y_IX_Y(output_strings, target_strings)
        (HX_Z, IZ_X) = self.calculate_HX_Y_IX_Y(target_strings, output_strings)
        self.summary_writer.add_scalars("H(Z|X) train", HZ_X, step)
        self.summary_writer.add_scalars("H(X|Z) train", HX_Z, step)
        self.summary_writer.add_scalars("I(Z,X) train", IZ_X, step)

        (HW_X, IW_X) = self.calculate_HX_Y_IX_Y(corner_strings, target_strings)
        (HX_W, IW_X) = self.calculate_HX_Y_IX_Y(target_strings, corner_strings)
        self.summary_writer.add_scalars("H(W|X) train", HW_X, step)
        self.summary_writer.add_scalars("H(X|W) train", HX_W, step)
        self.summary_writer.add_scalars("I(W,X) train", IW_X, step)

        (HY_Z, IY_Z) = self.calculate_HX_Y_IX_Y(activity_strings, output_strings)
        (HZ_Y, IY_Z) = self.calculate_HX_Y_IX_Y(output_strings, activity_strings)
        self.summary_writer.add_scalars("H(Y|Z) train", HY_Z, step)
        self.summary_writer.add_scalars("H(Z|Y) train", HZ_Y, step)
        self.summary_writer.add_scalars("I(Y,Z) train", IY_Z, step)

        (HY_W, IY_W) = self.calculate_HX_Y_IX_Y(activity_strings, corner_strings)
        (HW_Y, IY_W) = self.calculate_HX_Y_IX_Y(corner_strings, activity_strings)
        self.summary_writer.add_scalars("H(Y|W) train", HY_W, step)
        self.summary_writer.add_scalars("H(W|Y) train", HW_Y, step)
        self.summary_writer.add_scalars("I(Y,W) train", IY_W, step)

        (HZ_W, IZ_W) = self.calculate_HX_Y_IX_Y(output_strings, corner_strings)
        (HW_Z, IZ_W) = self.calculate_HX_Y_IX_Y(corner_strings, output_strings)
        self.summary_writer.add_scalars("H(Z|W) train", HZ_W, step)
        self.summary_writer.add_scalars("H(W|Z) train", HW_Z, step)
        self.summary_writer.add_scalars("I(Z,W) train", IZ_W, step)

        loss = self.model.loss_mse(preds, targets) / len(preds)
        accuracy = compute_accuracy(preds, targets)

        self.summary_writer.add_scalars("accuracy train", accuracy, step)
        self.summary_writer.add_scalars("loss train", loss, step)

        dL_dPred = self.model.loss_deriv(preds, targets)

        W1_feedback = self.model.W1
        W2_feedback = self.model.W2

        if not self.weight_transport:
            W1_feedback = np.random.randn(input_size, hidden_size)
            W2_feedback = np.random.randn(hidden_size, output_size)

        dL_dW1, dL_dW2 = self.model.backprop(
            W1_feedback,
            W2_feedback,
            dL_dPred,
            U=inputs,
            H=H,
            Z=Z,
            activate=self.activation_derivative,
            prob_not_backprop=self.prob_not_backprop,
        )

        self.model.W1 -= lr * dL_dW1
        self.model.W2 -= lr * dL_dW2
        return (
            loss,
            accuracy,
            HY_X,
            IY_X,
            HZ_X,
            IZ_X,
            HY_W,
            IY_W,
            HZ_W,
            IZ_W,
        )

    def entropy(self, activity_counts):
        H_X = 0
        total = sum(activity_counts.values())

        for h in activity_counts:
            H_X += (activity_counts[h] / total) * log2(activity_counts[h] / total)
        H_X = H_X * -1
        return H_X

    def calculate_HX_Y_IX_Y(self, X, Y):
        assert len(X) == len(Y)
        label_freq = {}
        X_counts = {}
        Y_counts = {}

        for i in range(0, len(X)):
            Y_i = Y[i]
            X_i = X[i]

            if X_i not in Y_counts:
                Y_counts[X_i] = 1
            else:
                Y_counts[X_i] += 1

            if Y_i not in X_counts:
                X_counts[Y_i] = 1
            else:
                X_counts[Y_i] += 1

            if Y_i not in label_freq:
                label_freq[Y_i] = {X_i: 1}
            else:
                if X_i not in label_freq[Y_i]:
                    label_freq[Y_i][X_i] = 1
                else:
                    label_freq[Y_i][X_i] += 1

        label_entropys = {key: 0 for key in label_freq.keys()}
        conditional_entropy = 0
        for x in label_freq:
            t = sum(label_freq[x].values())
            s = 0
            for y in label_freq[x]:
                s += (label_freq[x][y] / t) * log2((label_freq[x][y] / t))
            label_entropys[x] = s * -1
            conditional_entropy += (X_counts[x] / len(X)) * label_entropys[x]

        H_Y = self.entropy(Y_counts)

        mutual_information = H_Y - conditional_entropy

        return conditional_entropy, mutual_information

    def validate(self, step):
        preds, H, Z = self.model.forward(self.data_test)
        loss_test = self.model.loss_mse(preds, self.labels_test) / len(preds)
        accuracy_test = compute_accuracy(preds, self.labels_test)

        self.summary_writer.add_scalars("accuracy test", accuracy_test, step)
        self.summary_writer.add_scalars("loss test", loss_test, step)

        bin_hidden = self.bin_average(H)
        bin_output = self.bin_average(preds)

        activity_strings = []  # string binary representation of hidden layer activity
        output_strings = []  # string one-hot label representation for each batch item
        target_strings = []  # string one-hot label representation for each batch item
        corner_strings = []  # string 0-3 representation of corner for each batch item
        for x in range(0, self.batch_size):
            activity_strings.append("".join(map(str, bin_hidden[x].astype(int))))
            output_strings.append("".join(map(str, bin_output[x].astype(int))))
            target_strings.append("".join(map(str, self.labels_test[x].astype(int))))
            corner_strings.append(str(brightest_corner(self.data_test[x])))

        (HY_X, IY_X) = self.calculate_HX_Y_IX_Y(activity_strings, target_strings)
        (HX_Y, IY_X) = self.calculate_HX_Y_IX_Y(target_strings, activity_strings)
        self.summary_writer.add_scalars("H(Y|X) test", HY_X, step)
        self.summary_writer.add_scalars("H(X|Y) test", HX_Y, step)
        self.summary_writer.add_scalars("I(Y,X) test", IY_X, step)

        (HZ_X, IZ_X) = self.calculate_HX_Y_IX_Y(output_strings, target_strings)
        (HX_Z, IZ_X) = self.calculate_HX_Y_IX_Y(target_strings, output_strings)
        self.summary_writer.add_scalars("H(Z|X) test", HZ_X, step)
        self.summary_writer.add_scalars("H(X|Z) test", HX_Z, step)
        self.summary_writer.add_scalars("I(Z,X) test", IZ_X, step)

        (HW_X, IW_X) = self.calculate_HX_Y_IX_Y(corner_strings, target_strings)
        (HX_W, IW_X) = self.calculate_HX_Y_IX_Y(target_strings, corner_strings)
        self.summary_writer.add_scalars("H(W|X) test", HW_X, step)
        self.summary_writer.add_scalars("H(X|W) test", HX_W, step)
        self.summary_writer.add_scalars("I(W,X) test", IW_X, step)

        (HY_Z, IY_Z) = self.calculate_HX_Y_IX_Y(activity_strings, output_strings)
        (HZ_Y, IY_Z) = self.calculate_HX_Y_IX_Y(output_strings, activity_strings)
        self.summary_writer.add_scalars("H(Y|Z) test", HY_Z, step)
        self.summary_writer.add_scalars("H(Z|Y) test", HZ_Y, step)
        self.summary_writer.add_scalars("I(Y,Z) test", IY_Z, step)

        (HY_W, IY_W) = self.calculate_HX_Y_IX_Y(activity_strings, corner_strings)
        (HW_Y, IY_W) = self.calculate_HX_Y_IX_Y(corner_strings, activity_strings)
        self.summary_writer.add_scalars("H(Y|W) test", HY_W, step)
        self.summary_writer.add_scalars("H(W|Y) test", HW_Y, step)
        self.summary_writer.add_scalars("I(Y,W) test", IY_W, step)

        (HZ_W, IZ_W) = self.calculate_HX_Y_IX_Y(output_strings, corner_strings)
        (HW_Z, IZ_W) = self.calculate_HX_Y_IX_Y(corner_strings, output_strings)
        self.summary_writer.add_scalars("H(Z|W) test", HZ_W, step)
        self.summary_writer.add_scalars("H(W|Z) test", HW_Z, step)
        self.summary_writer.add_scalars("I(Z,W) test", IZ_W, step)

        print(
            f"epoch: {step}, loss_validation: {loss_test}, accuracy_validation: {accuracy_test}"
        )
        return (
            loss_test,
            accuracy_test,
            HY_X,
            IY_X,
            HZ_X,
            IZ_X,
            HY_W,
            IY_W,
            HZ_W,
            IZ_W,
        )

    def test(self, nn, inputs, targets):
        preds, H, Z = nn.forward(inputs)
        loss = nn.loss_mse(preds, targets) / len(preds)
        accuracy = compute_accuracy(preds, targets)
        return loss, accuracy

    def train(
        self,
        nbatches=10000,
        lr=0.00005,
        validation_frequency=100,
    ):
        xs = np.arange(nbatches)
        losses = np.zeros(nbatches).astype(np.double)
        accuracies = np.zeros(nbatches).astype(np.double)
        losses_test = np.zeros(nbatches).astype(np.double)
        accuracies_test = np.zeros(nbatches).astype(np.double)

        HY_Xs = np.zeros((nbatches))
        IY_Xs = np.zeros((nbatches))
        HZ_Xs = np.zeros((nbatches))
        IZ_Xs = np.zeros((nbatches))
        HY_Ws = np.zeros((nbatches))
        IY_Ws = np.zeros((nbatches))
        HZ_Ws = np.zeros((nbatches))
        IZ_Ws = np.zeros((nbatches))

        HY_Xs_test = np.zeros((nbatches))
        IY_Xs_test = np.zeros((nbatches))
        HZ_Xs_test = np.zeros((nbatches))
        IZ_Xs_test = np.zeros((nbatches))
        HY_Ws_test = np.zeros((nbatches))
        IY_Ws_test = np.zeros((nbatches))
        HZ_Ws_test = np.zeros((nbatches))
        IZ_Ws_test = np.zeros((nbatches))

        for i in range(nbatches):
            (
                loss,
                accuracy,
                HY_X,
                IY_X,
                HZ_X,
                IZ_X,
                HY_W,
                IY_W,
                HZ_W,
                IZ_W,
            ) = self.train_one_batch(lr, i)
            losses[i] = loss
            accuracies[i] = accuracy

            HY_Xs[i] = HY_X
            IY_Xs[i] = IY_X
            HZ_Xs[i] = HZ_X
            IZ_Xs[i] = IZ_X
            HY_Ws[i] = HY_W
            IY_Ws[i] = IY_W
            HZ_Ws[i] = HZ_W
            IZ_Ws[i] = IZ_W

            HY_Xs_test[i] = None
            IY_Xs_test[i] = None
            HZ_Xs_test[i] = None
            IZ_Xs_test[i] = None
            HY_Ws_test[i] = None
            IY_Ws_test[i] = None
            HZ_Ws_test[i] = None
            IZ_Ws_test[i] = None

            loss_test = None
            accuracy_test = None
            if ((i + 1) % validation_frequency) == 0:
                (
                    loss_test,
                    accuracy_test,
                    HY_X_test,
                    IY_X_test,
                    HZ_X_test,
                    IZ_X_test,
                    HY_W_test,
                    IY_W_test,
                    HZ_W_test,
                    IZ_W_test,
                ) = self.validate(i)
                HY_Xs_test[i] = HY_X_test
                IY_Xs_test[i] = IY_X_test
                HZ_Xs_test[i] = HZ_X_test
                IZ_Xs_test[i] = IZ_X_test
                HY_Ws_test[i] = HY_W_test
                IY_Ws_test[i] = IY_W_test
                HZ_Ws_test[i] = HZ_W_test
                IZ_Ws_test[i] = IZ_W_test

            losses_test[i] = loss_test
            accuracies_test[i] = accuracy_test

        self.summary_writer.to_csv()

        return {
            "loss": {
                "train": losses,
                "test": losses_test,
            },
            "accuracy": {
                "train": accuracies,
                "test": accuracies_test,
            },
            "entropy": [
                {"X": "Y", "Y": "X", "mode": "train", "data": HY_Xs},
                {"X": "Y", "Y": "X", "mode": "test", "data": HY_Xs_test},
                {"X": "Z", "Y": "X", "mode": "train", "data": HZ_Xs},
                {"X": "Z", "Y": "X", "mode": "test", "data": HZ_Xs_test},
                {"X": "Y", "Y": "W", "mode": "train", "data": HY_Ws},
                {"X": "Y", "Y": "W", "mode": "test", "data": HY_Ws_test},
                {"X": "Z", "Y": "W", "mode": "train", "data": HZ_Ws},
                {"X": "Z", "Y": "W", "mode": "test", "data": HZ_Ws_test},
            ],
            "info": [
                {"X": "Y", "Y": "X", "mode": "train", "data": IY_Xs},
                {"X": "Y", "Y": "X", "mode": "test", "data": IY_Xs_test},
                {"X": "Z", "Y": "X", "mode": "train", "data": IZ_Xs},
                {"X": "Z", "Y": "X", "mode": "test", "data": IZ_Xs_test},
                {"X": "Y", "Y": "W", "mode": "train", "data": IY_Ws},
                {"X": "Y", "Y": "W", "mode": "test", "data": IY_Ws_test},
                {"X": "Z", "Y": "W", "mode": "train", "data": IZ_Ws},
                {"X": "Z", "Y": "W", "mode": "test", "data": IZ_Ws_test},
            ],
        }


def compute_accuracy(preds, targets):
    assert len(preds) == len(targets)
    preds = np.argmax(preds, axis=1)
    targets = np.argmax(targets, axis=1)
    return float((targets == preds).sum()) / len(targets)


def plot_across_epochs(entries, mode):
    for x in entries:
        results = x["results"]
        for e in results:
            xs = np.arange(len(e["data"]))
            plt.plot(
                xs[np.isfinite(e["data"])],
                e["data"][np.isfinite(e["data"])],
                label=f"H({e['X']}|{e['Y']}), {x['hidden_size']} hidden neurons, {e['mode']}"
                if mode == "entropy"
                else f"I({e['Y']},{e['X']}), {x['hidden_size']} hidden neurons, {e['mode']}",
            )
    plt.xlabel("Epoch")
    plt.ylabel("H(X|Y)" if mode == "entropy" else "I(Y,X)")
    plt.legend()
    plt.show()


def plot_accuracy_across_epochs(accuracies, xs):
    for a in accuracies:
        plt.plot(
            xs, a["results"]["train"], label=f"hidden size: {a['hidden_size']}, train"
        )
        plt.plot(
            xs[np.isfinite(a["results"]["test"])],
            a["results"]["test"][np.isfinite(a["results"]["test"])],
            label=f"hidden size: {a['hidden_size']}, test",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Raw Classification Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()


def plot_loss_across_epochs(losses, xs):
    for l in losses:
        plt.plot(
            xs, l["results"]["train"], label=f"hidden size: {l['hidden_size']} train"
        )
        plt.plot(
            xs[np.isfinite(l["results"]["test"])],
            l["results"]["test"][np.isfinite(l["results"]["test"])],
            label=f"hidden size: {l['hidden_size']} test",
        )
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()


def plot_loss(values, titles):
    for i, ys in enumerate(values):
        xs = np.arange(len(ys["train"]))
        losses = ys["train"]
        losses_test = ys["test"]
        plt.plot(xs, losses, label=f"train {titles[i]}")
        plt.plot(
            xs[np.isfinite(losses_test)],
            losses_test[np.isfinite(losses_test)],
            label=f"validation {titles[i]}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()


def plot_accuracy(values, titles):
    for i, ys in enumerate(values):
        xs = np.arange(len(ys["train"]))
        losses = ys["train"]
        losses_test = ys["test"]
        plt.plot(xs, losses, label=f"train {titles[i]}")
        plt.plot(
            xs[np.isfinite(losses_test)],
            losses_test[np.isfinite(losses_test)],
            label=f"validation {titles[i]}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim((0, 1))
    plt.legend()
    plt.show()


def brightest_corner(image: np.array) -> int:
    i = image.reshape((28, 28))
    corners = np.array(
        [
            i[0:14, 0:14],
            i[0:14, 14:28],
            i[14:28, 0:14],
            i[14:28, 14:28],
        ]
    )
    brightest = np.argmax(corners.sum(axis=1).sum(axis=1))
    return brightest


def symmetry(image: np.array) -> int:
    i = image.reshape((28, 28))
    return (
        np.sum(np.abs(i[0:14, 0:28] - np.flip(i[14:28, 0:28], 0)))
        - np.sum(np.abs(i[0:28, 0:14] - np.flip(i[0:28, 14:28], 0)))
    ) > 0


def edge_side(image: np.array) -> int:
    i = image.reshape((28, 28))
    edge_sobel = filters.sobel(i)
    return brightest_corner(edge_sobel.reshape((784)))


if __name__ == "__main__":
    input_size = 784
    hidden_size = 30
    output_size = 10

    nn = nn_one_layer(input_size, hidden_size, output_size)

    train_size = 60000
    test_size = 10000
    data = (
        utils.read_mnist_image(
            "datasets/train-images-idx3-ubyte.gz",
            28,
            28,
            train_size,
        )
        / 255
    )
    labels = utils.read_mnist_label(
        "datasets/train-labels-idx1-ubyte.gz",
        train_size,
    )

    data_test = (
        utils.read_mnist_image(
            "datasets/t10k-images-idx3-ubyte.gz",
            28,
            28,
            test_size,
        )
        / 255
    )
    labels_test = utils.read_mnist_label(
        "datasets/t10k-labels-idx1-ubyte.gz",
        test_size,
    )

    n_batches = 10000
    xs = np.arange(n_batches)
    # summary_writer = SummaryWriter(nbatches=n_batches, logdir="logs/run_2.csv")
    # summary_writer.title_prefix = "120 neurons"
    # results_120 = Trainer(
    #     nn_one_layer(input_size, 120, output_size),
    #     data,
    #     labels,
    #     data_test,
    #     labels_test,
    #     summary_writer,
    # ).train(n_batches)

    # summary_writer.title_prefix = "60 neurons"
    # results_60 = Trainer(
    #     nn_one_layer(input_size, 60, output_size),
    #     data,
    #     labels,
    #     data_test,
    #     labels_test,
    #     summary_writer,
    # ).train(n_batches)

    # summary_writer.title_prefix = "30 neurons"
    # results_30 = Trainer(
    #     nn_one_layer(input_size, 30, output_size),
    #     data,
    #     labels,
    #     data_test,
    #     labels_test,
    #     summary_writer,
    # ).train(n_batches)

    # summary_writer.title_prefix = "15 neurons"
    # results_15 = Trainer(
    #     nn_one_layer(input_size, 15, output_size),
    #     data,
    #     labels,
    #     data_test,
    #     labels_test,
    #     summary_writer,
    # ).train(n_batches)

    # summary_writer.title_prefix = "5 neurons"
    # results_5 = Trainer(
    #     nn_one_layer(input_size, 5, output_size),
    #     data,
    #     labels,
    #     data_test,
    #     labels_test,
    #     summary_writer,
    # ).train(n_batches)

    summary_writer = SummaryWriter(
        nbatches=n_batches, logdir="logs/run_autoencoder.csv"
    )
    summary_writer.title_prefix = "autoencoder"
    results_5 = Trainer(
        nn_one_layer(input_size, 30, input_size),
        data,
        data,
        data_test,
        data_test,
        summary_writer,
    ).train(4000)

    # results_entropy_a = [
    #     {"hidden_size": 15, "results": results_15["entropy"][:2]},
    #     {"hidden_size": 30, "results": results_30["entropy"][:2]},
    #     {"hidden_size": 60, "results": results_60["entropy"][:2]},
    #     {"hidden_size": 120, "results": results_120["entropy"][:2]},
    # ]
    # results_entropy_b = [
    #     {"hidden_size": 15, "results": results_15["entropy"][2:4]},
    #     {"hidden_size": 30, "results": results_30["entropy"][2:4]},
    #     {"hidden_size": 60, "results": results_60["entropy"][2:4]},
    #     {"hidden_size": 120, "results": results_60["entropy"][2:4]},
    # ]
    # results_entropy_c = [
    #     {"hidden_size": 15, "results": results_15["entropy"][4:6]},
    #     {"hidden_size": 30, "results": results_30["entropy"][4:6]},
    #     {"hidden_size": 60, "results": results_60["entropy"][4:6]},
    #     {"hidden_size": 120, "results": results_120["entropy"][4:6]},
    # ]
    # results_entropy_d = [
    #     {"hidden_size": 15, "results": results_15["entropy"][6:]},
    #     {"hidden_size": 30, "results": results_30["entropy"][6:]},
    #     {"hidden_size": 60, "results": results_60["entropy"][6:]},
    #     {"hidden_size": 120, "results": results_120["entropy"][6:]},
    # ]

    # results_info_a = [
    #     {"hidden_size": 15, "results": results_15["info"][:2]},
    #     {"hidden_size": 30, "results": results_30["info"][:2]},
    #     {"hidden_size": 60, "results": results_60["info"][:2]},
    #     {"hidden_size": 120, "results": results_120["info"][:2]},
    # ]
    # results_info_b = [
    #     {"hidden_size": 15, "results": results_15["info"][2:4]},
    #     {"hidden_size": 30, "results": results_30["info"][2:4]},
    #     {"hidden_size": 60, "results": results_60["info"][2:4]},
    #     {"hidden_size": 120, "results": results_120["info"][2:4]},
    # ]
    # results_info_c = [
    #     {"hidden_size": 15, "results": results_15["info"][4:6]},
    #     {"hidden_size": 30, "results": results_30["info"][4:6]},
    #     {"hidden_size": 60, "results": results_60["info"][4:6]},
    #     {"hidden_size": 120, "results": results_120["info"][4:6]},
    # ]
    # results_info_d = [
    #     {"hidden_size": 15, "results": results_15["info"][6:]},
    #     {"hidden_size": 30, "results": results_30["info"][6:]},
    #     {"hidden_size": 60, "results": results_60["info"][6:]},
    #     {"hidden_size": 120, "results": results_120["info"][6:]},
    # ]

    # accuracies = [
    #     {"hidden_size": 15, "results": results_15["accuracy"]},
    #     {"hidden_size": 30, "results": results_30["accuracy"]},
    #     {"hidden_size": 60, "results": results_60["accuracy"]},
    #     {"hidden_size": 120, "results": results_120["accuracy"]},
    # ]
    # losses = [
    #     {"hidden_size": 15, "results": results_15["loss"]},
    #     {"hidden_size": 30, "results": results_30["loss"]},
    #     {"hidden_size": 60, "results": results_60["loss"]},
    #     {"hidden_size": 120, "results": results_120["loss"]},
    # ]

    # plot_accuracy_across_epochs(accuracies, xs)
    # plot_loss_across_epochs(losses, xs)

    # plot_across_epochs(
    #     [{"hidden_size": 30, "results": results_30["entropy"]}], "entropy"
    # )
    # plot_across_epochs([{"hidden_size": 30, "results": results_30["info"]}], "info")

    # plt.scatter(
    #     results_30["info"],
    #     results_30["info"],
    #     c=range(0, len(HY_Xs_test)),
    #     label="hidden",
    # )
    # plt.xlabel("Conditional Entropy")
    # plt.ylabel("Mutual information")
    # plt.show()

    # plot_across_epochs(results_entropy_a, "entropy")
    # plot_across_epochs(results_info_a, "info")
    # plot_across_epochs(results_entropy_b, "entropy")
    # plot_across_epochs(results_info_b, "info")
    # plot_across_epochs(results_entropy_c, "entropy")
    # plot_across_epochs(results_info_c, "info")
    # plot_across_epochs(results_entropy_d, "entropy")
    # plot_across_epochs(results_info_d, "info")

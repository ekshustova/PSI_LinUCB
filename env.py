
import numpy as np
import struct
import gzip
import os


def load_mnist(path="datasets/MNIST/raw"):
    for f in os.listdir(path):
        if 'train-images' in f:
            images = _load_images(os.path.join(path, f))
        elif 'train-labels' in f:
            labels = _load_labels(os.path.join(path, f))
    return images, labels


def _load_images(filepath):
    opener = gzip.open if filepath.endswith('.gz') else open
    with opener(filepath, 'rb') as f:
        f.read(4)  # magic
        n = struct.unpack('>I', f.read(4))[0]
        f.read(8)  # rows, cols
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, 784).astype(np.float64)


def _load_labels(filepath):
    opener = gzip.open if filepath.endswith('.gz') else open
    with opener(filepath, 'rb') as f:
        f.read(8)  # magic, n
        return np.frombuffer(f.read(), dtype=np.uint8)


def group_by_class(features, labels):
    clusters = {}
    for k in range(10):
        clusters[k] = features[labels == k]
    return clusters


class MNISTBanditEnv:
    def __init__(self, clusters, target_class=0):
        self.clusters = clusters
        self.target_class = target_class
        self.K = 10
        self.d = 784
        self.reset()

    def reset(self):
        self.t = 0
        self.mistakes = 0
        self.cumulative_mistakes = []

    def get_contexts(self):
        contexts = np.zeros((self.K, self.d))
        for k in range(self.K):
            idx = np.random.randint(len(self.clusters[k]))
            contexts[k] = self.clusters[k][idx] #это перестановки
        return contexts

    def step(self, action):
        self.t += 1

        if action == self.target_class:
            reward = 1.0
        else:
            reward = 0.0
            self.mistakes += 1

        self.cumulative_mistakes.append(self.mistakes)
        return reward


if __name__ == "__main__":

    np.random.seed(42)
    clusters = {k: np.random.randn(100, 784) for k in range(10)}

    env = MNISTBanditEnv(clusters, target_class=0)

    T = 1000
    for _ in range(T):
        contexts = env.get_contexts()
        action = np.random.randint(10)
        env.step(action)

    print(f"Mistakes: {env.mistakes} / {T}")
    print(f"Expected: ~{T * 0.9:.0f}")
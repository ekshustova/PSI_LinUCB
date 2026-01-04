import numpy as np
import pickle
import os
import tarfile


def load_cifar10(path="datasets/CIFAR10"):
    extracted_path = os.path.join(path, "cifar-10-batches-py")
    images, labels = [], []
    for i in range(1, 6):
        with open(os.path.join(extracted_path, f"data_batch_{i}"), 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        images.append(batch[b'data'])
        labels.extend(batch[b'labels'])

    return np.vstack(images).astype(np.float64), np.array(labels)


def group_by_class(features, labels):
    return {k: features[labels == k] for k in range(10)}


class CIFAR10BanditEnv:
    def __init__(self, clusters, target_class=0):
        self.clusters = clusters
        self.target_class = target_class
        self.K = 10
        self.d = 3072
        self.reset()

    def reset(self):
        self.t = 0
        self.mistakes = 0
        self.cumulative_mistakes = []

    def get_contexts(self):
        contexts = np.zeros((self.K, self.d))
        for k in range(self.K):
            idx = np.random.randint(len(self.clusters[k]))
            contexts[k] = self.clusters[k][idx]
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
    images, labels = load_cifar10()
    features = images / np.max(np.linalg.norm(images, axis=1))
    clusters = group_by_class(features, labels)

    print(f"Features: {features.shape}")
    print(f"Samples per class: {[len(clusters[k]) for k in range(10)]}")
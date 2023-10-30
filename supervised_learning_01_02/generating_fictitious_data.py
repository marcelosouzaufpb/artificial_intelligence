import matplotlib.pyplot as plt
import numpy as np


def generate_data(random=10, time=60):
    rng = np.random.RandomState(random)
    x = 12 * rng.rand(time)
    y = 2 * x - 1 + rng.randn(time)
    return x, y


def main():
    x, y = generate_data()
    plt.scatter(x, y)
    plt.show()


if __name__ == "__main__":
    main()

from pathlib import Path
from PIL import Image
import numpy as np

from Neuron import NeuralNetwork

output = []
dataset = []
size = (100, 100)

def read_image(path):
    image = Image.open(path).resize(size)
    arr = np.asarray(image)
    arr = arr[:, :, 0]
    arr = arr.reshape((size[0] * size[1],))
    arr = arr / 255
    for i in range(len(arr)):
        if (arr[i] < 1):
            arr[i] = 0.01
        else:
            arr[i] = 0
    return arr

def create_dataset():
    p = Path("data")
    for x in p.rglob("*"):
        if x.match("*.png"):
            arr = read_image(x)
            mas = str(x).split("\\")
            if mas[1] not in output:
                output.append(mas[1])
            dataset.append((np.asarray([arr]), output.index(mas[1])))


def write_mas(file_name, mas):
    f = open(file_name, 'w')
    for i in mas:
        f.write(i + '\n')
    f.close()

def read_mas(file_name):
    f = open(file_name, 'r')
    mas = []
    for line in f.readlines():
        mas.append(eval(line))
    return mas

create_dataset()
print(dataset)
print(output)
INPUT_DIM = size[0]**2
OUT_DIM = len(output)
H_DIM = size[0]

# W1 = np.random.rand(INPUT_DIM, H_DIM)
# b1 = np.random.rand(1, H_DIM)
# W2 = np.random.rand(H_DIM, OUT_DIM)
# b2 = np.random.rand(1, OUT_DIM)
#
# network = NeuralNetwork(W1, b1, W2, b2)
# network.train(dataset)
#
# np.savetxt("W1.csv", network.W1, delimiter=",")
# np.savetxt("W2.csv", network.W2, delimiter=",")
# np.savetxt("b1.csv", network.b1, delimiter=",")
# np.savetxt("b2.csv", network.b2, delimiter=",")

W1 = np.loadtxt("W1.csv", delimiter=",")
W2 = np.loadtxt("W2.csv", delimiter=",")
b1 = np.loadtxt("b1.csv", delimiter=",")
b2 = np.loadtxt("b2.csv", delimiter=",")
network = NeuralNetwork(W1, b1, W2, b2)
test = read_image("test.png")
print(output[np.argmax(network.predict(np.asarray(test)))])
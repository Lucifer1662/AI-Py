import numpy as np
import matplotlib.pyplot as plt
from ai import NeuralNet

image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "data/mnist/"
# train_data = np.loadtxt(data_path + "mnist_train.csv", 
#                         delimiter=",")

test_data = np.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",") 
# test_data[:10]


def extractData(offset, length):
    fac = 0.99 / 255

    imgs = np.asfarray(test_data[offset:offset+length, 1:]) * fac + 0.01
    labels = np.asarray(test_data[offset:offset+length, 0:1], dtype=np.int).flatten()

    lr = np.arange(10)
    labels_encoded = np.zeros((length,10))

    for i in range(length):
        label = labels[i]
        one_hot = (lr==label).astype(np.int)
        labels_encoded[i] = one_hot
    
    return (imgs, labels_encoded, labels.T)


train_size = 100

(train_imgs, train_labels_encoded, _) = extractData(0, train_size)

layerSizes = [train_imgs.shape[1], 32, 16, train_labels_encoded.shape[1]]

neural = NeuralNet(layerSizes)
neural.load()

#neural.TrainEpoch(train_imgs.T, train_labels_encoded.T, iterations=100)

#neural.save()

test_size = 10
(test_imgs, test_labels_encoded, test_labels) = extractData(0, train_size)

predications = neural.oneHotHighestValue(test_imgs.T)


print(neural.accuracyOneHotHighestValue(test_imgs.T, test_labels))

print("(expected, predications)")
print(list(zip(test_labels,predications)))




# for i in range(1):
#     img = test_imgs[i].reshape((28,28))
#     plt.imshow(img, cmap="Greys")
#     plt.show()



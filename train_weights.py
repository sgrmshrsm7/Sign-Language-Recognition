from PIL import Image
import numpy as np

# Activation func
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# convert image to binary array
def convertImg(img):
    grayscale_img = img.convert('L')
    grayscale_img = grayscale_img.resize((10, 10))  # gives 10x10 pixels img
    arr = np.array(grayscale_img)  # gives 10x10 array
    arr = np.reshape(arr, (100, 1))  # gives 100x1 array
    threshold = 30  # value for determining whether 1 or 0
    bin_arr = arr
    for i in range(100):
        if arr[i][0] < threshold:
            bin_arr[i][0] = 1
        else:
            bin_arr[i][0] = 0
    return bin_arr

alphabets_bin_arr = []

for i in range(26):
    alphabets_bin_arr.append([])

# binary input arrays for each alphabet
for alphabet in range(26):
    for i in range(0,5):
        alphabets_bin_arr[alphabet].append(convertImg(Image.open(chr(ord('A') + alphabet) + "/" + chr(ord('a') + alphabet) + str(i) + '.png')))

temp = []
for i in range(26):
    temp.append(0)

# binary output arrays
training_outputs_arr = []
for i in range(26):
    training_outputs_arr.append(np.array([temp]).T)
    training_outputs_arr[i][i][0] = 1

# leraning rate
alpha = 0.01

synaptic_weights1 = 2*np.random.random(
    (64, 100)) - 1  # gives a 64x100 array for weights btw first hidden layer and input layer
synaptic_weights2 = 2*np.random.random(
    (64, 64)) - 1  # gives a 64x64 array for weights btw first hidden layer and input layer
synaptic_weights3 = 2*np.random.random(
    (26, 64)) - 1 # gives a 26x64 array for weights btw first hidden layer and output layer

epochs = 0

while epochs < 1000:
    index = 0
    for letter_bin_arr in alphabets_bin_arr:
        for bin_arr in letter_bin_arr:
            hidden_output1 = sigmoid(np.dot(synaptic_weights1, bin_arr) + 1)
            hidden_output2 = sigmoid(np.dot(synaptic_weights2, hidden_output1) + 1)
            output = sigmoid(np.dot(synaptic_weights3, hidden_output2))

            dely = (training_outputs_arr[index] - output) * sigmoid_derivative(output)
            delz = np.dot(synaptic_weights3.T, dely) * sigmoid_derivative(hidden_output2)
            delx = np.dot(synaptic_weights2.T , delz) * sigmoid_derivative(hidden_output1)
            synaptic_weights3 += (alpha * np.dot(dely,hidden_output2.T))
            synaptic_weights2 += (alpha * np.dot(delz, hidden_output1.T))
            synaptic_weights1 += (alpha * np.dot(delx, bin_arr.T))

        index += 1

    epochs += 1

file = open(r"trained_weights.txt",'w')

for i in synaptic_weights1:
    for j in i:
        file.write(str(j))
        file.write("\n")
for i in synaptic_weights2:
    for j in i:
        file.write(str(j))
        file.write("\n")
for i in synaptic_weights3:
    for j in i:
        file.write(str(j))
        file.write("\n")

file.close()



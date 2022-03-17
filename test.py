from PIL import Image
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

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

synaptic_weights1 = []
synaptic_weights2 = []
synaptic_weights3 = []

file = open(r"trained_weights.txt",'r')

for row in range(64):
    array = []
    for i in range(100):
        a = file.readline()
        a = a[:len(a) - 1]
        array.append(float(a))
    synaptic_weights1.append(array)

for row in range(64):
    array = []
    for i in range(64):
        a = file.readline()
        a = a[:len(a) - 1]
        array.append(float(a))
    synaptic_weights2.append(array)

for row in range(26):
    array = []
    for i in range(64):
        a = file.readline()
        a = a[:len(a) - 1]
        array.append(float(a))
    synaptic_weights3.append(array)

file.close()

synaptic_weights1 = np.array(synaptic_weights1)
synaptic_weights2 = np.array(synaptic_weights2)
synaptic_weights3 = np.array(synaptic_weights3)

img = []
for alphabet in range(26):
    img.append(convertImg(Image.open(chr(ord('A') + alphabet) + '/' + chr(ord('a') + alphabet) + '5.png')))

print("Outputs after training for learning rate 0.01 : ")

for input_layer1 in img:
    outputs1 = sigmoid(np.dot( synaptic_weights1 , input_layer1 ))
    input_layer2 = outputs1
    outputs2 = sigmoid(np.dot( synaptic_weights2 , input_layer2 ))
    input_layer3=outputs2
    outputs3 = sigmoid(np.dot( synaptic_weights3 , input_layer3 ))

# input_layer1 = convertImg(Image.open("try1.jpg"))
# 
# outputs1 = sigmoid(np.dot( synaptic_weights1 , input_layer1 ))
# input_layer2 = outputs1
# outputs2 = sigmoid(np.dot( synaptic_weights2 , input_layer2 ))
# input_layer3=outputs2
# outputs3 = sigmoid(np.dot( synaptic_weights3 , input_layer3 ))

# print(outputs3)

    for i in range(len(outputs3)):
        if outputs3[i] > 0.5:
            outputs3[i] = 1
        else:
            outputs3[i] = 0
    
    char = ""
    for i in range(len(outputs3)):
        if outputs3[i] == 1:
            char += chr(ord('A') + i)
    
    print(char,end = " ")
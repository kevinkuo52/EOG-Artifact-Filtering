import csv
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

with open('data_eeg.csv', 'r') as f:
  reader = csv.reader(f)
  data = list(reader)


def getData(row):
    list = []
    for i in range(0, 22016):
        list.append(float(data[i][row]))
    return list


def fill():
    list = []
    for i in range(0, 22016):
        list.append(i+1)
    return list

def transpose(list):
    return [[list[j][i] for j in range(len(list))] for i in range(len(list[0]))]


def eogPresenceFilter(eeg1, eeg2, eeg3):
    nonEogIndex = []
    for i in range(0, 22016):
        if math.fabs(eeg1[i]) < 75 or math.fabs(eeg2[i]) < 75 or math.fabs(eeg3[i]) < 75:
            nonEogIndex.append(i)
    return nonEogIndex


def removeNonEogPresenceData(nonEogIndex, eog1, eog2, eog3, eeg1, eeg2, eeg3):
    print(len(nonEogInterval))
    for i in nonEogIndex:
        eog1[i] = 0
        eog2[i] = 0
        eog3[i] = 0
        eeg1[i] = 0
        eeg2[i] = 0
        eeg3[i] = 0
    return eog1, eog2, eog3, eeg1, eeg2, eeg3
    


eog22 = getData(21)
eog21 = getData(20)
eog20 = getData(19)
eeg1 = getData(0)
eeg2 = getData(1)
eeg3 = getData(2)

#weight = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
nonEogInterval = eogPresenceFilter(eog20, eog21, eog22)
eog20, eog21, eog22, eeg1, eeg2, eeg3 = removeNonEogPresenceData(nonEogInterval, eog20, eog21, eog22, eeg1, eeg2, eeg3)


b = transpose([eeg1, eeg2, eeg3])

x_train = transpose([eog20, eog21, eog22])

y_train = transpose([getData(3), getData(4), getData(5)])

y_train_offset = np.subtract(y_train,b)

weight = np.divide(np.dot(transpose(x_train), y_train_offset), np.dot(transpose(x_train), x_train))
print("weight:")
print(weight)

cleanData = np.subtract(y_train, np.dot(x_train, weight))
print("clean data:")
print(cleanData.shape)
print(cleanData)
print("row: ")
print(cleanData[:][0])
#plt.scatter(np.linspace(0,1,len(cleanData[:,0])),cleanData[:,0], cmap=plt.cm.Set1)
plt.figure(1, figsize=(30, 6))
plt.plot(getData(20)[1000:18000])
plt.plot(cleanData[:,1][1000:18000])
plt.xlabel('millisecond')
plt.ylabel('uV')
plt.title('EOG Filtering')
#11
#16
#20
'''
plt.plot(cleanData[:,0])
plt.figure(2, figsize=(30, 6))
plt.plot(getData(4))
plt.plot(cleanData[:,1])
plt.figure(3, figsize=(30, 6))
plt.plot(getData(5))
plt.plot(cleanData[:,2])
'''


plt.show()

'''
weight = tf.Variable(tf.zeros([3, 3], dtype=tf.float32))
b = tf.constant(transpose([eeg1, eeg2, eeg3]), dtype=tf.float32)
x = tf.placeholder(tf.float32)

product = tf.matmul(x, weight, transpose_b=True)

linear_model = product + b

y = tf.placeholder(tf.float32)

loss = tf.reduce_sum(tf.square(linear_model - y))
optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(loss)

x_train = transpose([eog20, eog21, eog22])
y_train = transpose([getData(3), getData(4), getData(5)])

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

curr_weight, curr_loss = sess.run([weight, loss], {x: x_train, y: y_train})
print("Weight: %s loss: %s" % (curr_weight, curr_loss))
'''
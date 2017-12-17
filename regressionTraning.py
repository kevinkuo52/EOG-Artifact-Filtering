import csv
import tensorflow as tf
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


eog22 = getData(21)
eog21 = getData(20)
eog20 = getData(19)
eeg1 = getData(0)
eeg2 = getData(1)
eeg3 = getData(2)


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

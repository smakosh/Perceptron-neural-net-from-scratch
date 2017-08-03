import numpy as np

# each point is height, weight, shoe-size & gender (0.1)
# 0 for male and 1 for female
data = [[1.81, 0.80, 0.44, 0],
        [1.77, 0.70, 0.43, 0],
        [1.60, 0.60, 0.38, 1],
        [1.54, 0.54, 0.37, 1],
        [1.66, 0.65, 0.40, 0],
        [1.90, 0.90, 0.47, 0],
        [1.75, 0.64, 0.39, 1],
        [1.77, 0.70, 0.40, 1],
        [1.59, 0.55, 0.37, 1],
        [1.71, 0.75, 0.42, 0],
        [1.81, 0.85, 0.43, 0]]

mystery_person = [1.63, 0.60, 0.37]

def sigmoid(x) :
    return 1/(1 + np.exp(-x))

def sigmoid_p(x) :
    return sigmoid(x) * (1-sigmoid(x))

# training loop
learning_rate = 0.2
costs = []

#weights & bias
w1 = np.random.randn()
w2 = np.random.randn()
w3 = np.random.randn()
b = np.random.randn()

for i in range(10000000) :
    ri = np.random.randint(len(data))
    point = data[ri]

    z = point[0] * w1 + point[1] * w2 + point[2] * w3 + b
    prediction = sigmoid(z)

    target = point[3]

    #cost function / squared error function
    cost = np.square(prediction - target)

    #derivative of cost function
    dcost_prediction = 2 * (prediction - target)
    dprediction_dz = sigmoid_p(z)

    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_dw3 = point[2]
    dz_db = 1

    dcost_dz = dcost_prediction * dprediction_dz

    dcost_dw1 = dcost_dz * dz_dw1
    dcost_dw2 = dcost_dz * dz_dw2
    dcost_dw3 = dcost_dz * dz_dw3
    dcost_db = dcost_dz * dz_db

    #new weights & bias values
    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    w3 = w3 - learning_rate * dcost_dw3
    b = b - learning_rate * dcost_db

    if i % 100 == 0 :
        cost_sum = 0
        for j in range(len(data)) :
            point = data[ri]

            z = point[0] * w1 + point[1] * w2 + point[2] * w3 + b
            prediction = sigmoid(z)

            target = point[3]
            cost_sum += np.square(prediction - target)

# prediction
for i in range(len(data)) :
    point = data[i]
    print(point)

    z = point[0] * w1 + point[1] * w2 + point[2] * w3 + b
    prediction = sigmoid(z)
    print("prediction : {}" .format(prediction))


z = mystery_person[0] * w1 + mystery_person[1] * w2 + mystery_person[2] * w3 + b
prediction = sigmoid(z)
prediction

def guess_gender(height, weight, shoe_size) :
    z = height * w1 + weight * w2 + shoe_size * w3 + b
    prediction = sigmoid(z)
    if prediction < .5:
        print('male')
    else:
        print('female')

guess_gender(1.75, 0.72, 0.41)

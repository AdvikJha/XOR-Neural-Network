import random
import math

inputs = [
    [0,1],
    [1,0],
    [0,0],
    [1,1]
]

correct_outputs = [1,1,0,0]

inputWeight1 = random.uniform(-1, 1)
inputWeight2 = random.uniform(-1, 1)
inputWeight3 = random.uniform(-1, 1)
inputWeight4 = random.uniform(-1, 1)
inputBias1 = random.uniform(-1, 1)
inputBias2 = random.uniform(-1, 1)

outputWeight1 = random.uniform(-1, 1)
outputWeight2 = random.uniform(-1, 1)
outputBias1 = random.uniform(-1, 1)

x1 = 0
x2 = 0

def Sigmoid(z):
    sigmoid = 1/(1+math.exp(-z))
    return sigmoid

def weightedSum(x1, x2, w1, w2, b):
    return w1*x1 + w2*x2 + b

def predict(x1, x2):
    activationVal1 = Sigmoid(weightedSum(x1, x2, inputWeight1, inputWeight2, inputBias1))
    activationVal2 = Sigmoid(weightedSum(x1, x2, inputWeight3, inputWeight4, inputBias2))
    prediction = Sigmoid(weightedSum(activationVal1, activationVal2, outputWeight1, outputWeight2, outputBias1))
    return activationVal1, activationVal2, prediction

def calculateLoss():
    sumOfSquareds = 0

    for index, i in enumerate(inputs):
        x1=i[0]
        x2=i[1]
        actVal1, actVal2, prediction = predict(x1, x2)
        error = prediction - correct_outputs[index]
        sumOfSquareds = sumOfSquareds + error **2

    LossMSE = 0.25*sumOfSquareds
    return LossMSE

def calculateGradients():
    gradient_w1 = 0
    gradient_w2 = 0
    gradient_w3 = 0
    gradient_w4 = 0
    gradient_w5 = 0
    gradient_w6 = 0
    gradient_b1 = 0
    gradient_b2 = 0
    gradient_b3 = 0

    for index, x in enumerate(inputs):
        x1=x[0]
        x2=x[1]

        activationVal1, activationVal2, prediction = predict(x1, x2)

        gradient_w1 += (
            2*(prediction - correct_outputs[index])
            * prediction * (1-prediction)
            * outputWeight1
            * activationVal1 * (1-activationVal1)
            * x1
        )
        
        gradient_w2 += (
            2*(prediction - correct_outputs[index])
            * prediction * (1-prediction)
            * outputWeight1
            * activationVal1 * (1-activationVal1)
            * x2
        )

        gradient_b1 += (
            2*(prediction - correct_outputs[index])
            * prediction * (1-prediction)
            * outputWeight1
            * activationVal1 * (1-activationVal1)
        )

        gradient_w3 += (
            2*(prediction - correct_outputs[index])
            * prediction * (1-prediction)
            * outputWeight2
            * activationVal2 * (1-activationVal2)
            * x1
        )

        gradient_w4 += (
            2*(prediction - correct_outputs[index])
            * prediction * (1-prediction)
            * outputWeight2
            * activationVal2 * (1-activationVal2)
            * x2
        )

        gradient_b2 += (
            2*(prediction - correct_outputs[index])
            * prediction * (1-prediction)
            * outputWeight2
            * activationVal2 * (1-activationVal2)
        )

        gradient_w5 += (
            2*(prediction - correct_outputs[index])
            * prediction * (1-prediction)
            * activationVal1
        )

        gradient_w6 += (
            2*(prediction - correct_outputs[index])
            * prediction * (1-prediction)
            * activationVal2
        )

        gradient_b3 += (
            2*(prediction - correct_outputs[index])
            * prediction * (1-prediction)
        )

    gradient_w1 /= len(inputs)
    gradient_w2 /= len(inputs)
    gradient_b1 /= len(inputs)
    gradient_w3 /= len(inputs)
    gradient_w4 /= len(inputs)
    gradient_b2 /= len(inputs)
    gradient_w5 /= len(inputs)
    gradient_w6 /= len(inputs)
    gradient_b3 /= len(inputs)

    return gradient_w1, gradient_w2, gradient_w3, gradient_w4, gradient_w5, gradient_w6, gradient_b1, gradient_b2, gradient_b3

LossMSE=calculateLoss()

print("Weight1 is ", inputWeight1)
print("Weight2 is ", inputWeight2)
print("Weight3 is ", inputWeight3)
print("Weight4 is ", inputWeight4)
print("Weight5 is ", outputWeight1)
print("Weight6 is ", outputWeight2)
print("Bias1 is ", inputBias1)
print("Bias2 is ", inputBias2)
print("Bias3 is ", outputBias1, "\n")

iterationCount = 0
while iterationCount < 100000:
    iterationCount += 1
    learningRate = 1
    g_w1, g_w2, g_w3, g_w4, g_w5, g_w6, g_b1, g_b2, g_b3 = calculateGradients()

    inputWeight1 -= g_w1*learningRate
    inputWeight2 -= g_w2*learningRate
    inputWeight3 -= g_w3*learningRate
    inputWeight4 -= g_w4*learningRate
    inputBias1 -=g_b1*learningRate
    inputBias2 -= g_b2*learningRate
    outputWeight1 -= g_w5*learningRate
    outputWeight2 -= g_w6*learningRate
    outputBias1 -= g_b3*learningRate

    LossMSE = calculateLoss()

    if iterationCount % 10000 == 0:
        print(iterationCount, LossMSE)

print("\n\n\nIteration: ", iterationCount)
print("Weight1 is ", inputWeight1)
print("Weight2 is ", inputWeight2)
print("Weight3 is ", inputWeight3)
print("Weight4 is ", inputWeight4)
print("Weight5 is ", outputWeight1)
print("Weight6 is ", outputWeight2)
print("Bias1 is ", inputBias1)
print("Bias2 is ", inputBias2)
print("Bias3 is ", outputBias1, "\n")

for x in inputs:
    _, _, prediction = predict(x[0], x[1])
    print(x, "->", prediction)
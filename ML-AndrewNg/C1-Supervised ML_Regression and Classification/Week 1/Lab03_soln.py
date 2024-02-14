import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0, 2.0]);
y_train = np.array([300, 500]);

print(x_train)
print(y_train)

m = x_train.shape[0]
print(f"m = ",m)
print(f"m = {m}")

i = 0;
x_i = x_train[i];
y_i = y_train[i];

print(f"x_{i} = {x_i}, y_{i} = {y_i}")

plt.scatter(x_train, y_train, label='Training Data')
plt.show()

def compute_model_output(x_train, w, b):

    f = np.zeros(x_train.shape[0])

    for i in range(x_train.shape[0]):
        f[i] = w * x_train[i] + b
    
    return f

f = compute_model_output(x_train, 200, 100)
print(f)

plt.plot(x_train, f, label='Model Output')
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
plt.legend()  # This line is added to show the label in the plot
plt.show()  # This line is added to display the plot

#we now have weights and bias, we can use them to make predictions
w = 200
b = 100
x_i = 1.2
y_i = w * x_i + b
print(f"Prediction for x_{i} = {x_i} is {y_i}")
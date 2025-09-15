from sympy import symbols, fourier_series, pi, cos, sin
import numpy as np
import matplotlib.pyplot as plt

# Define the function
t = symbols('t')
f = 5*sin(t)

# Define the interval and number of periods to show
period = 2 * np.pi
num_periods = 3
start_time = -num_periods * period / 2 
end_time = num_periods * period / 2

# Generate time values across multiple periods
t_values = np.linspace(start_time, end_time, 500)
f_numeric = np.vectorize(lambda t_values: f.subs(t, t_values).evalf())

# Calculate the function values

f_values = f_numeric(t_values)

# Plot the function
plt.figure(figsize=(10, 5))
plt.plot(t_values, f_values)
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('График периодической функции f(t) = 5*sin(t)')
plt.grid(True)
plt.show()
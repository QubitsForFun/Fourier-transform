'''
Лабораторная работа 1. Изучение методов спектрального анализа.

'''
#-----------------------------------------------------------------------
#                         Импорт библиотек
from sympy import symbols, fourier_series, pi, cos, sin
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
#import pandas as pd

#-----------------------------------------------------------------------
#                 Функция f(t)=5sin(wt) и ее график
t = symbols('t')
w = 1 #Циклическая частота [рад/с], можно поменять, она будет распространяться на весь код
f = 5*sin(w*t)

# Согласно заданию, нужно отрисовать на нескольких периодах, так и сделаем
period = 2 * np.pi
num_periods = 3
start_time = 0.0 #изменено 15.09.25 , t теперь >= 0
end_time = num_periods * period

t_values = np.linspace(start_time, end_time, 500)
f_numeric = np.vectorize(lambda t_values: f.subs(t, t_values).evalf())
f_values = f_numeric(t_values)

# Рисуем график
plt.figure(figsize=(10, 5))
plt.plot(t_values, f_values)
plt.xlabel('t,с')
plt.ylabel('f(t)')
plt.title('График периодической функции f(t) = $5sin(\omega t)$')
plt.grid(True)
plt.show()

#-----------------------------------------------------------------------
#                 Функция g(t) = 5sin(1t) + 10cos(2t) + 3sin(3t) + 6cos(4t) + 10 и ее график
g = 5*sin(w*t) + 10*cos(2*t) + 3*sin(3*t) + 6*cos(4*t) + 10.0

g_numeric = np.vectorize(lambda t_values: g.subs(t, t_values).evalf())

g_values = g_numeric(t_values)

# Рисуем график
plt.figure(figsize=(10, 5))
plt.plot(t_values, g_values)
plt.xlabel('t,с')
plt.ylabel('g(t)')
plt.title('График периодической функции $g(t) = 5 sin(\omega t) + 10 cos(2 \omega t) + 3 sin(3 \omega t) + 6 cos(4 \omega t) + 10$')
plt.grid(True)
plt.show()

#------------------------------------------------------------------------
#                        Табулирование функции g(t) в пределах периода.
N=5 #можно подобрать и другое N
T=2*pi
i = np.array(range(N))
deltha_T = T / N
t_n = i * deltha_T

#print(i)
#print(deltha_T)
#print(t_n)

#------------------------------------------------------------------------
#                         Дискретные значения G(t_n), решил считать для новой функции, а не Y(t_n)
def G(a): #=Y(t_n)
  g_numeric = np.vectorize(lambda value: g.subs(t, value).evalf())
  g_values = g_numeric(a)
  return g_values

#print(G(t_n))

#------------------------------------------------------------------------
#                                     Ищем a_0/2
integrand = lambda t : (2/T)*(5*sin(t) + 10*cos(2*t) + 3*sin(3*t) + 6*cos(4*t) + 10.0)
integral, integral_error = quad(integrand, 0, T)
half_a_0 = integral/2

#print(f'a_0/2 = ', {half_a_0})

#------------------------------------------------------------------------
#                                     Ищем a_n
a = np.zeros(N)
for n in range(N):
  integrand = lambda t : (5*sin(w*t) + 10*cos(2*w*t) + 3*sin(3*w*t) + 6*cos(4*w*t) + 10.0)*cos(n*w*t)
  integral, integral_error = quad(integrand, 0, T)
  a[n] = integral*2/T
#print(f"a)
#print(pd.DataFrame(a))
print('a_n = ')
#print(a)
for coef in a:
  print(f'{coef:.0f}')

#------------------------------------------------------------------------
#                                     Ищем b_n
b = np.zeros(N)
for n in range(N):
  integrand = lambda t : (5*sin(t) + 10*cos(2*t) + 3*sin(3*t) + 6*cos(4*t) + 10.0)*sin(n*w*t)
  integral, integral_error = quad(integrand, 0, T)
  b[n] = integral*2/T
print('b_n = ')
#print(b)
#print(pd.DataFrame(b))
for coef in b:
  print(f'{coef:.0f}')

#------------------------------------------------------------------------
#                                     Ищем A_n

An = np.sqrt(a**2 + b**2) #Сначала думал там нужно по индексам, но pandas сам все сделал, я проверил, все ок

#print(pd.DataFrame(An))
print('A_n = ')

for coef in An:
  print(f'{coef:.0f}')
#------------------------------------------------------------------------
#                                 Строим АЧХ A_n(w_n)
w_n = np.zeros(N)
for n in range(N):
  w_n[n] = n * w

#print(w_n)

#Рисуем АЧХ
plt.figure(figsize=(10, 5))
plt.stem(w_n, An.flatten()) # An is already a NumPy array, use flatten() for safety
plt.xlabel('$w_n$, Гц')
plt.ylabel('$An$')
plt.title('График зависимости $A_n$ от $\omega_n$')
plt.xticks(np.arange(0, N)) # Set x-ticks to show all integer values from 0 to N-1
plt.grid(axis='y') # Add grid lines for the y-axis
plt.show()
#------------------------------------------------------------------------
#Итого ряд Фурье g(t)=10*cos(2*w*t)+5.98*cos(4*w*t)+5*sin(1*w*t)+3*sin(3*w*t) сходится к исходной функции

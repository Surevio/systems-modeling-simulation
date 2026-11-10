import numpy as np
from scipy.stats import chi2
from math import sqrt
import matplotlib.pyplot as plt

def vychet_met(a, m, x0):
    numbers = []
    seen = set()
    x = x0

    while x not in seen:
        seen.add(x)
        numbers.append(x / m)
        x = (a * x) % m
        if len(numbers) > 100000:
            break
    if len(numbers) > 1 and numbers[-1] == numbers[0]:
        numbers = numbers[:-1]
    if len(numbers) < 1000:
        print(f"Предупреждение: период = {len(numbers)} < 1000")
    return numbers

def file(numbers,file_name):
    with open(file_name,'w') as f:
        for n in numbers:
            f.write(str(n)+' ')

def func_raspr(numbers):
    sorted_numbers = sorted(numbers)
    y = np.arange(len(sorted_numbers))/float(len(sorted_numbers)-1)
    plt.plot(sorted_numbers, y, 'darkslateblue')
    plt.title('Функция распределения')
    plt.legend(['function','gamma'])
    plt.plot([0, 1], [0, 1], 'r--', label='Теоретическая F(x)=x')
    plt.legend()
    plt.show()
    return y

def hists(numbers):
    n = len(numbers)
    k = int(sqrt(n))
    if k % 2 == 0:
        k += 1
    k = max(min(k, 20), 5)
    
    plt.hist(numbers, bins=k, range=(0, 1), edgecolor='black', alpha=0.7, density=True)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Гистограмма распределения')
    plt.grid(True)
    plt.axhline(y=1.0, color='red', linestyle='--', label='Теоретическая f(x)=1')
    plt.legend()
    plt.show()

def sluch(numbers):
    n = len(numbers)
    if n < 2:
        print("Ошибка: нужно минимум 2 числа")
        return

    median = np.median(numbers)
    signs = []
    for x in numbers:
        if x > median:
            signs.append(1)
        elif x < median:
            signs.append(-1)
        else:
            signs.append(0)

    znaki = [s for s in signs if s != 0]
    n_zn = len(znaki)

    if len(znaki) < 2:
        print("Ошибка: после удаления нулей осталось меньше 2 знаков")
        return

    kolvo_seriy = 1
    for i in range(1, len(znaki)):
        if znaki[i] != znaki[i-1]:
            kolvo_seriy += 1
    print(f"Наблюдаемое количество серий: {kolvo_seriy}")
    
    n_plus = sum(1 for z in znaki if z == 1)
    n_minus = sum(1 for z in znaki if z == -1)

    print(f"n(+) = {n_plus}, n(-) = {n_minus}, всего знаков = {n_zn}")
    print(f"len(numbers) = {n}")

    if n_plus < 20 or n_minus < 20:
        print("Внимание: n(+) или n(-) < 20, нормальная аппроксимация может быть неточной")

    mu_r = (2 * n_plus * n_minus) / n_zn + 1
    sigma_r_sq = (2 * n_plus * n_minus * (2 * n_plus * n_minus - n_zn)) / ((n_zn ** 2) * (n_zn - 1))
    sigma_r = sqrt(sigma_r_sq) if sigma_r_sq > 0 else 0

    z_crit = 1.96
    lower_bound = mu_r - z_crit * sigma_r
    upper_bound = mu_r + z_crit * sigma_r

    print(f'Мат. ожидание: {mu_r:.4f}, Стандартное отклонение: {sigma_r:.4f}')
    print(f'95% доверительный интервал: [{lower_bound:.4f}, {upper_bound:.4f}]')
    print(f'Наблюдаемое значение: {kolvo_seriy}')

    if lower_bound <= kolvo_seriy <= upper_bound:
        print('Проверка на случайность пройдена')
    else:
        print('Проверка на случайность НЕ пройдена')

def hi(numbers):
    n = len(numbers)
    alpha = 0.05

    k = int(sqrt(n))
    if k % 2 == 0:
        k += 1
    k = max(k, 5)
    print(f"Количество интервалов k = {k}")
    
    granitsy = [i / k for i in range(k + 1)]
    
    chast_nabl = [0] * k
    for x in numbers:
        found = False
        for i in range(k):
            if granitsy[i] <= x < granitsy[i + 1]:
                chast_nabl[i] += 1
                found = True
                break
        if not found and x == 1:
            chast_nabl[k - 1] += 1

    chast_oj = n / k
    if chast_oj < 5:
        print("Внимание: ожидаемая частота < 5. Тест может быть неточным!")
    hi_2 = 0
    for i in range(k):
        hi_2 += ((chast_nabl[i] - chast_oj) ** 2) / chast_oj
    m = k - 1
    hi_krit = chi2.ppf(1 - alpha, m)

    print(f"Наблюдаемое значение hi_2 = {hi_2:.4f}")
    print(f"Критическое значение hi_2 (a={alpha}, m={m}) = {hi_krit:.4f}")

    if hi_2 < hi_krit:
        print("Гипотеза о равномерности принимается")
        return True
    else:
        print("Гипотеза о равномерности НЕ принимается")
        return False

a = 16807
m = 2147483647
x0 = 123456789

numbers = vychet_met(a, m, x0)
file(numbers, 'numbers.txt')
hists(numbers)
func_raspr(numbers)
sluch(numbers)
hi(numbers) 
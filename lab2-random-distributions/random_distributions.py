import numpy as np
from scipy.stats import chi2, norm
from math import sqrt, log, exp
import matplotlib.pyplot as plt

# Базовый генератор равномерных чисел (метод вычетов, ЛР №1)
a_lcg = 16807
m_lcg = 2147483647
x0_lcg = 123456789

def uniform_generator(a, m, x0):
    # Бесконечный генератор псевдослучайных чисел, равномерно
    # распределённых на [0,1], методом вычетов (Лемера).
    x = x0
    while True:
        x = (a * x) % m
        yield x / m

gen = uniform_generator(a_lcg, m_lcg, x0_lcg)

def get_uniforms(n):
    return [next(gen) for _ in range(n)]

# Параметры варианта 2 (строка 38 таблицы; № по списку 97,
# 97 = ((97-1) % 59) + 1 = 38-я строка таблицы, Вариант = 2)
A_PAR = 196            # равномерное на [-a, a]
K_PAR, LAM_PAR = 7, 5  # гамма(k, λ); λ также переиспользуется для Пуассона
M_PAR, S_PAR = 88, 9   # нормальное(m, s) и отражённое нормальное(m, s)

N = 2000  # объём выборки для каждой случайной величины (см. обоснование в отчёте)


# 1. Равномерное распределение на [-a, a]
def gen_uniform_symmetric(a, n):
    g = get_uniforms(n)
    return [a * (2 * u - 1) for u in g]


# 2. Гамма-распределение (k, λ) - сумма k экспоненциальных величин
def gen_gamma(k, lam, n):
    result = []
    for _ in range(n):
        g = get_uniforms(k)
        x = -(1 / lam) * sum(log(u) for u in g)
        result.append(x)
    return result


# 3. Нормальное (0,1) методом суммы 12 равномерных величин
def gen_normal_01(n):
    result = []
    for _ in range(n):
        g = get_uniforms(12)
        y = sum(g) - 6
        result.append(y)
    return result


# 4. Нормальное (m, s)
def gen_normal_ms(m, s, n):
    z = gen_normal_01(n)
    return [m + s * zi for zi in z]


# 5. Отражённое нормальное (m, s): X = |Y|, Y ~ N(m, s)
def gen_reflected_normal(m, s, n):
    y = gen_normal_ms(m, s, n)
    return [abs(yi) for yi in y]


# 6. Качественная (дискретная) СВ с заданным рядом событий
# Пример ряда событий: оценка качества детали на конвейере
EVENTS = ['брак', 'удовлетворительно', 'хорошо', 'отлично']
PROBS = [0.1, 0.3, 0.4, 0.2]

def gen_discrete(events, probs, n):
    cum = np.cumsum(probs)
    g = get_uniforms(n)
    result = []
    for u in g:
        for i, c in enumerate(cum):
            if u <= c:
                result.append(events[i])
                break
        else:
            result.append(events[-1])
    return result


# 7. Распределение Пуассона (λ) - алгоритм Кнута
def gen_poisson(lam, budget):
    # В отличие от остальных шести законов, метод Кнута тратит ПЕРЕМЕННОЕ
    # число исходных равномерных чисел на одно итоговое значение (в среднем
    # ~λ+1 чисел). Поэтому здесь budget - это не количество итоговых значений,
    # а лимит на расход исходных чисел из базового генератора (тот же N=2000,
    # что и у остальных законов) - а количество получившихся пуассоновских
    # значений заранее не известно и зависит от случайности.
    L = exp(-lam)
    result = []
    draws_used = 0
    while draws_used < budget:
        k = 0
        p = 1.0
        while True:
            k += 1
            u = next(gen)
            draws_used += 1
            p *= u
            if p <= L:
                break
        result.append(k - 1)
    return result, draws_used


# Визуализация: гистограмма + теоретическая плотность/масса
def hist_with_theory(data, title, fname, theory_x=None, theory_y=None, discrete=False, xlabel='Значение'):
    plt.figure(figsize=(7, 4.5))
    if discrete:
        if theory_x is not None and isinstance(theory_x[0], str):
            # сохраняем заданный логический порядок категорий, а не алфавитный
            from collections import Counter
            cnt = Counter(data)
            values = list(theory_x)
            freqs = [cnt.get(v, 0) / len(data) for v in values]
        else:
            values, counts = np.unique(data, return_counts=True)
            freqs = counts / len(data)
        plt.bar(values, freqs, color='steelblue', edgecolor='black', alpha=0.8, label='эмпирическая')
        if theory_x is not None:
            plt.plot(theory_x, theory_y, 'ro-', label='теоретическая', markersize=5)
    else:
        n = len(data)
        k = int(sqrt(n))
        if k % 2 == 0:
            k += 1
        k = max(min(k, 30), 5)
        plt.hist(data, bins=k, edgecolor='black', alpha=0.7, density=True, color='darkslateblue', label='эмпирическая')
        if theory_x is not None:
            plt.plot(theory_x, theory_y, 'r--', linewidth=2, label='теоретическая')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Плотность' if not discrete else 'Частота')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=110)
    plt.close()


# Критерий согласия Пирсона (хи-квадрат) - для проверки ОДНОЙ
# из величин: нормальной N(m, s)
def chi_square_test_normal(data, m, s, alpha=0.05):
    n = len(data)
    k = int(sqrt(n))
    if k % 2 == 0:
        k += 1
    k = max(min(k, 20), 5)

    edges = np.linspace(min(data), max(data), k + 1)
    observed, _ = np.histogram(data, bins=edges)

    cdf_vals = norm.cdf(edges, loc=m, scale=s)
    expected_probs = np.diff(cdf_vals)
    expected = expected_probs * n

    chi2_stat = sum(((o - e) ** 2) / e for o, e in zip(observed, expected) if e > 0)
    dof = k - 1  # m и s не оценивались по выборке, а были заранее известны (теми же, которыми выборка сгенерирована) - поэтому степени свободы не уменьшаются на число параметров
    chi_crit = chi2.ppf(1 - alpha, dof)

    print(f"Количество интервалов k = {k}")
    print(f"Наблюдаемое значение хи2 = {chi2_stat:.4f}")
    print(f"Критическое значение хи2 (a={alpha}, m={dof}) = {chi_crit:.4f}")
    if chi2_stat < chi_crit:
        print("Гипотеза о нормальности распределения ПРИНИМАЕТСЯ")
        return True
    else:
        print("Гипотеза о нормальности распределения ОТВЕРГАЕТСЯ")
        return False


if __name__ == "__main__":
    # 1. Равномерное на [-a, a]
    print("=" * 60)
    print("1. Равномерное распределение на [-a, a], a =", A_PAR)
    data1 = gen_uniform_symmetric(A_PAR, N)
    theory_x = np.linspace(-A_PAR, A_PAR, 100)
    theory_y = np.full_like(theory_x, 1 / (2 * A_PAR))
    hist_with_theory(data1, f'Равномерное распределение на [-{A_PAR}, {A_PAR}]', 'fig1_uniform.png', theory_x, theory_y)
    print(f"Среднее (выборочное) = {np.mean(data1):.3f}, теоретическое = 0")
    print(f"Дисперсия (выборочная) = {np.var(data1):.1f}, теоретическая = {(2*A_PAR)**2/12:.1f}")

    # 2. Гамма(k, λ)
    print("=" * 60)
    print(f"2. Гамма-распределение, k={K_PAR}, λ={LAM_PAR}")
    data2 = gen_gamma(K_PAR, LAM_PAR, N)
    from scipy.stats import gamma as gamma_dist
    theory_x = np.linspace(0.01, max(data2), 200)
    theory_y = gamma_dist.pdf(theory_x, a=K_PAR, scale=1/LAM_PAR)
    hist_with_theory(data2, f'Гамма-распределение (k={K_PAR}, λ={LAM_PAR})', 'fig2_gamma.png', theory_x, theory_y)
    print(f"Среднее (выборочное) = {np.mean(data2):.3f}, теоретическое = {K_PAR/LAM_PAR:.3f}")

    # 3. Нормальное (0,1)
    print("=" * 60)
    print("3. Нормальное распределение (0, 1)")
    data3 = gen_normal_01(N)
    theory_x = np.linspace(-4, 4, 200)
    theory_y = norm.pdf(theory_x, 0, 1)
    hist_with_theory(data3, 'Нормальное распределение (0, 1)', 'fig3_normal01.png', theory_x, theory_y)
    print(f"Среднее = {np.mean(data3):.3f}, СКО = {np.std(data3):.3f}")

    # 4. Нормальное (m, s)
    print("=" * 60)
    print(f"4. Нормальное распределение (m={M_PAR}, s={S_PAR})")
    data4 = gen_normal_ms(M_PAR, S_PAR, N)
    theory_x = np.linspace(M_PAR - 4*S_PAR, M_PAR + 4*S_PAR, 200)
    theory_y = norm.pdf(theory_x, M_PAR, S_PAR)
    hist_with_theory(data4, f'Нормальное распределение (m={M_PAR}, s={S_PAR})', 'fig4_normal_ms.png', theory_x, theory_y)
    print(f"Среднее (выборочное) = {np.mean(data4):.3f}, теоретическое = {M_PAR}")
    print(f"СКО (выборочное) = {np.std(data4):.3f}, теоретическое = {S_PAR}")

    # 5. Отражённое нормальное (m, s)
    print("=" * 60)
    print(f"5. Отражённое нормальное распределение (m={M_PAR}, s={S_PAR})")
    data5 = gen_reflected_normal(M_PAR, S_PAR, N)
    # теоретическая плотность |Y|: f(x) = f_norm(x) + f_norm(-x), x>=0
    theory_x = np.linspace(0, M_PAR + 4*S_PAR, 200)
    theory_y = norm.pdf(theory_x, M_PAR, S_PAR) + norm.pdf(-theory_x, M_PAR, S_PAR)
    hist_with_theory(data5, f'Отражённое нормальное распределение (m={M_PAR}, s={S_PAR})', 'fig5_reflected.png', theory_x, theory_y)
    print(f"Среднее (выборочное) = {np.mean(data5):.3f}")

    # 6. Качественная СВ
    print("=" * 60)
    print("6. Качественная СВ (ряд событий):", dict(zip(EVENTS, PROBS)))
    data6 = gen_discrete(EVENTS, PROBS, N)
    theory_x = EVENTS
    theory_y = PROBS
    hist_with_theory(data6, 'Качественная СВ: оценка качества детали', 'fig6_discrete.png', theory_x, theory_y, discrete=True, xlabel='Категория')
    from collections import Counter
    cnt = Counter(data6)
    print("Наблюдаемые частоты:", {e: cnt[e]/N for e in EVENTS})

    # 7. Пуассон(λ)
    print("=" * 60)
    print(f"7. Распределение Пуассона, λ={LAM_PAR}")
    data7, draws_used7 = gen_poisson(LAM_PAR, N)
    print(f"Реально получено пуассоновских значений: {len(data7)}")
    from scipy.stats import poisson as poisson_dist
    theory_x = np.arange(0, max(data7)+1)
    theory_y = poisson_dist.pmf(theory_x, LAM_PAR)
    hist_with_theory(data7, f'Распределение Пуассона (λ={LAM_PAR})', 'fig7_poisson.png', theory_x, theory_y, discrete=True, xlabel='k')
    print(f"Среднее (выборочное) = {np.mean(data7):.3f}, теоретическое = {LAM_PAR}")
    print(f"Дисперсия (выборочная) = {np.var(data7):.3f}, теоретическая = {LAM_PAR}")
    print(f"Минимум = {min(data7)}, Максимум = {max(data7)}, Сумма = {sum(data7)}")

    # Сохраняем все значения в отдельный файл
    with open('poisson_values.txt', 'w', encoding='utf-8') as f:
        f.write("Формула генерации Пуассона (алгоритм Кнута):\n")
        f.write(f"L = e^(-λ), λ = {LAM_PAR}\n")
        f.write("k = 0, p = 1\n")
        f.write("пока p > L: k += 1, p = p * (следующее равномерное число)\n")
        f.write("результат = k - 1\n\n")
        f.write(f"Все {len(data7)} сгенерированных значений:\n")
        f.write(' '.join(str(v) for v in data7))
    print("Файл сохранён: poisson_values.txt")

    # Проверка качества: критерий Пирсона для N(m,s)
    print("=" * 60)
    print("Проверка согласия для нормального распределения N(m, s):")
    chi_square_test_normal(data4, M_PAR, S_PAR)
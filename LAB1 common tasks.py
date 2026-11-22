import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# папка для графіків
base_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(base_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# -------------------------------------------------------------
# Завдання 1: модель СМО (M/M/1)
# -------------------------------------------------------------
def task1():
    print("=== Завдання 1: СМО web-сервера ===")
    mu = 40.0  # швидкість обробки сервером (запитів/хв)
    lambdas = [10.0, 20.0, 30.0, 35.0, 38.0]  # різні значення вхідного потоку
    
    print(f"{'lambda':<8} {'mu':<8} {'rho':<10} {'L':<10} {'W (сек)':<12} {'Lq':<10}")
    print("-" * 55)
    
    for l in lambdas:
        rho = l / mu  # навантаження
        if rho < 1:
            L = rho / (1 - rho)         # середня к-сть запитів у системі
            W = (1 / (mu - l)) * 60     # середній час очікування (у секундах)
            Lq = (rho ** 2) / (1 - rho) # середня черга
            print(f"{l:<8.1f} {mu:<8.1f} {rho:<10.3f} {L:<10.3f} {W:<12.2f} {Lq:<10.3f}")
        else:
            print(f"{l:<8.1f} {mu:<8.1f} {rho:<10.3f} черга переповнена (нестабільний режим)")

# -------------------------------------------------------------
# Завдання 2: оптимізація часу процесора (симплекс-метод)
# -------------------------------------------------------------
def task2():
    print("\n=== Завдання 2: Оптимізація часу процесора ===")
    
    # функція продуктивності: F = 5*x1 + 3*x2 + 4*x3 -> max
    # linprog робить мінімізацію, тому коефіцієнти беремо з мінусом
    c = [-5.0, -3.0, -4.0]
    
    # обмеження за часом: x1 + x2 + x3 <= 10
    A = [[1.0, 1.0, 1.0]]
    b = [10.0]
    
    # мінімальний час на кожен процес: x1 >= 2, x2 >= 1, x3 >= 1
    bounds = [(2.0, None), (1.0, None), (1.0, None)]
    
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    print(f"Оптимальний час процесу 1 (x1): {res.x[0]:.1f} год")
    print(f"Оптимальний час процесу 2 (x2): {res.x[1]:.1f} год")
    print(f"Оптимальний час процесу 3 (x3): {res.x[2]:.1f} год")
    print(f"Максимальна продуктивність F:   {-res.fun:.1f} ум. од.")
    
    # перевіряємо як зміниться F якщо міняти загальний доступний час
    print("\nАналіз чутливості (варіюємо загальний час T):")
    print(f"{'T (год)':<10} {'x1':<8} {'x2':<8} {'x3':<8} {'F_max':<8}")
    for T in [5, 8, 10, 12, 15]:
        r = linprog(c, A_ub=[[1.0, 1.0, 1.0]], b_ub=[T], bounds=bounds, method='highs')
        if r.success:
            print(f"{T:<10} {r.x[0]:<8.1f} {r.x[1]:<8.1f} {r.x[2]:<8.1f} {-r.fun:<8.1f}")

# -------------------------------------------------------------
# Завдання 3: імітаційна модель бази даних
# -------------------------------------------------------------
def task3():
    print("\n=== Завдання 3: Моделювання бази даних (1000 запитів) ===")
    np.random.seed(42)
    n = 1000
    arrival_rate = 100.0  # 100 запитів/сек
    
    # інтервали надходження за експоненційним законом (переводимо в мілісекунди)
    inter_arrivals = np.random.exponential(scale=1000.0 / arrival_rate, size=n)
    arrivals = np.cumsum(inter_arrivals)
    
    # 80% читання, 20% запис
    types = np.random.choice(['read', 'write'], size=n, p=[0.8, 0.2])
    
    # генеруємо час обробки
    service = np.zeros(n)
    for i in range(n):
        if types[i] == 'read':
            service[i] = np.random.uniform(10.0, 20.0)  # 10-20 мс
        else:
            service[i] = np.random.uniform(30.0, 50.0)  # 30-50 мс
            
    # моделюємо чергу до 2 воркерів
    workers = [0.0, 0.0]
    wait = np.zeros(n)
    
    for i in range(n):
        t = arrivals[i]
        # беремо того воркера, який звільниться першим
        idx = 0 if workers[0] <= workers[1] else 1
        start = max(t, workers[idx])
        wait[i] = start - t
        workers[idx] = start + service[i]
        
    print(f"Всього запитів: {n}")
    print(f"Читання: {np.sum(types == 'read')}, Запис: {np.sum(types == 'write')}")
    print(f"Сер. час обробки читання: {np.mean(service[types == 'read']):.2f} мс")
    print(f"Сер. час обробки запису:   {np.mean(service[types == 'write']):.2f} мс")
    print(f"Сер. час очікування в черзі: {np.mean(wait):.2f} мс")
    print(f"Загальний середній час:      {np.mean(wait + service):.2f} мс")
    
    # будуємо гістограму очікування
    plt.figure(figsize=(8, 4.5))
    plt.hist(wait, bins=25, color='#3470a3', edgecolor='black', alpha=0.8)
    plt.axvline(np.mean(wait), color='red', linestyle='--', linewidth=2, label=f'Сер. час = {np.mean(wait):.1f} мс')
    plt.title('Розподіл часу очікування запитів до БД')
    plt.xlabel('Час очікування (мс)')
    plt.ylabel('Кількість запитів')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'task3_db_latency_hist.png'), dpi=300)
    plt.close()
    print("Гістограму збережено у plots/task3_db_latency_hist.png")

if __name__ == '__main__':
    task1()
    task2()
    task3()

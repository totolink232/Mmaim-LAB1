import os
import numpy as np
import matplotlib.pyplot as plt

# папка під графіки
base_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(base_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# налаштування шрифту для графіків
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#999999'

# -------------------------------------------------------------
# Варіант 3: Моделювання мережі з пакетною передачею даних
# -------------------------------------------------------------
class NetworkSimulation:
    def __init__(self, mu=100.0, buffer_capacity=20):
        # mu - середня швидкість каналу (пакетів на секунду)
        # buffer_capacity - максимальна місткість буфера разом із передавачем
        self.mu = float(mu)
        self.K = int(buffer_capacity)

    def run(self, arrival_rate, num_packets=15000, seed=42):
        np.random.seed(seed)
        
        # інтервали прибуття пакетів (експоненційний розподіл з інтенсивністю lambda)
        dt = np.random.exponential(scale=1.0 / arrival_rate, size=num_packets)
        arrival_times = np.cumsum(dt)
        
        # час передачі для кожного пакета в каналі (експоненційний з mu)
        service_times = np.random.exponential(scale=1.0 / self.mu, size=num_packets)
        
        delays = []
        dropped = 0
        accepted = 0
        
        # список моментів виходу пакетів з черги
        departures = []
        queue_lengths = []
        
        for i in range(num_packets):
            t_arr = arrival_times[i]
            t_serv = service_times[i]
            
            # прибираємо з черги всі пакети, які вже відправилися до цього моменту
            departures = [t for t in departures if t > t_arr]
            cur_len = len(departures)
            queue_lengths.append(cur_len)
            
            # якщо буфер заповнений до K - пакет відкидається (loss)
            if cur_len >= self.K:
                dropped += 1
            else:
                accepted += 1
                # якщо черга порожня, передача йде одразу, інакше чекаємо попередній пакет
                start_time = t_arr if cur_len == 0 else departures[-1]
                end_time = start_time + t_serv
                departures.append(end_time)
                
                # рахуємо загальну затримку (очікування + передача)
                delays.append(end_time - t_arr)
                
        loss_rate = dropped / num_packets
        avg_delay = np.mean(delays) if delays else 0.0
        
        return {
            'loss_rate': loss_rate,
            'avg_delay_ms': avg_delay * 1000.0,
            'dropped': dropped,
            'accepted': accepted,
            'throughput': accepted / arrival_times[-1],
            'queue_trace': queue_lengths[:400],
            'time_trace': arrival_times[:400]
        }

    @staticmethod
    def get_theoretical(l, mu, K):
        # формули з теорії ймовірностей для черги M/M/1/K
        rho = l / mu
        if abs(rho - 1.0) < 1e-5:
            p0 = 1.0 / (K + 1)
            p_loss = 1.0 / (K + 1)
            L = K / 2.0
        else:
            p0 = (1.0 - rho) / (1.0 - rho ** (K + 1))
            p_loss = p0 * (rho ** K)
            n = np.arange(K + 1)
            pn = p0 * (rho ** n)
            L = np.sum(n * pn)
            
        l_eff = l * (1.0 - p_loss)
        avg_delay = (L / l_eff) if l_eff > 0 else 0.0
        return {
            'p_loss': p_loss,
            'avg_delay_ms': avg_delay * 1000.0,
            'throughput': l_eff
        }

def main():
    mu = 100.0  # швидкість передавача 100 пак/сек
    K = 15      # розмір черги
    sim = NetworkSimulation(mu=mu, buffer_capacity=K)
    
    # перевіряємо різні коефіцієнти завантаження від 0.2 до 1.5
    rhos = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.5]
    
    sim_losses = []
    sim_delays = []
    theo_losses = []
    theo_delays = []
    
    print("=== Моделювання мережі з комутацією пакетів (Варіант 3) ===")
    print(f"Параметри: mu = {mu} пак/с, розмір черги K = {K}")
    print(f"{'rho':<6} {'lambda':<8} {'Втрати іміт (%)':<18} {'Втрати теор (%)':<18} {'Затримка іміт':<15} {'Затримка теор'}")
    print("-" * 80)
    
    for rho in rhos:
        l = rho * mu
        res = sim.run(arrival_rate=l, num_packets=15000, seed=123)
        th = sim.get_theoretical(l, mu, K)
        
        sim_losses.append(res['loss_rate'] * 100.0)
        sim_delays.append(res['avg_delay_ms'])
        theo_losses.append(th['p_loss'] * 100.0)
        theo_delays.append(th['avg_delay_ms'])
        
        print(f"{rho:<6.2f} {l:<8.1f} {res['loss_rate']*100:<18.2f} {th['p_loss']*100:<18.2f} {res['avg_delay_ms']:<15.2f} {th['avg_delay_ms']:.2f}")

    # 1. Графік втрати пакетів
    plt.figure(figsize=(8, 4.8))
    plt.plot(rhos, sim_losses, 'o-', color='#d9534f', label='Імітація')
    plt.plot(rhos, theo_losses, '--', color='#333333', label='Теорія (M/M/1/K)')
    plt.title('Ймовірність втрати пакетів від навантаження (rho)')
    plt.xlabel('Коефіцієнт навантаження rho')
    plt.ylabel('Втрати пакетів (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'variant3_loss_vs_load.png'), dpi=300)
    plt.close()

    # 2. Графік затримки
    plt.figure(figsize=(8, 4.8))
    plt.plot(rhos, sim_delays, 's-', color='#0275d8', label='Імітація')
    plt.plot(rhos, theo_delays, '--', color='#333333', label='Теорія (M/M/1/K)')
    plt.title('Середня затримка пакета від навантаження (rho)')
    plt.xlabel('Коефіцієнт навантаження rho')
    plt.ylabel('Середня затримка (мс)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'variant3_delay_vs_load.png'), dpi=300)
    plt.close()

    # 3. Дослідження Bufferbloat (вплив розміру буфера K при високому навантаженні)
    buf_sizes = [5, 10, 20, 40]
    high_l = 0.95 * mu
    buf_losses = []
    buf_delays = []
    
    for b in buf_sizes:
        s_b = NetworkSimulation(mu=mu, buffer_capacity=b)
        r_b = s_b.run(arrival_rate=high_l, num_packets=15000, seed=123)
        buf_losses.append(r_b['loss_rate'] * 100.0)
        buf_delays.append(r_b['avg_delay_ms'])
        
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.set_xlabel('Розмір буфера K (пакетів)')
    ax1.set_ylabel('Втрати пакетів (%)', color='#d9534f')
    ax1.plot(buf_sizes, buf_losses, 'o-', color='#d9534f', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='#d9534f')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Середня затримка (мс)', color='#0275d8')
    ax2.plot(buf_sizes, buf_delays, 's--', color='#0275d8', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='#0275d8')
    plt.title('Вплив розміру буфера на втрати та затримку (rho = 0.95)')
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'variant3_buffer_tradeoff.png'), dpi=300)
    plt.close()

    # 4. Графік черги в часі
    trace_res = sim.run(arrival_rate=95.0, num_packets=1000, seed=777)
    plt.figure(figsize=(9, 4))
    plt.step(trace_res['time_trace'][:250], trace_res['queue_trace'][:250], where='post', color='#2ca02c')
    plt.axhline(K, color='red', linestyle='--', label=f'Ліміт буфера K = {K}')
    plt.title('Заповнення буфера маршрутизатора в часі')
    plt.xlabel('Час (секунди)')
    plt.ylabel('Пакетів у черзі')
    plt.ylim(-0.5, K + 2)
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'variant3_queue_trace.png'), dpi=300)
    plt.close()

    print("\nВсі графіки оновлено та збережено в папці plots/")

if __name__ == '__main__':
    main()

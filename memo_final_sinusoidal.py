import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


# ============================================================================
# STEP 1: DEFINE THE SYSTEM PARAMETERS
# ============================================================================

class GPUSystemParameters:

    def __init__(self):
       
        self.N_gpus = 1000  # Total number of GPUs

        # ── GPU Memory Classes ───────────────────────────────────────────────
        # Fleet is partitioned into G classes ordered by memory capacity.
        # A job with memory_req µ can ONLY run on class g where m_g >= µ.
        # (Technical note §5.1: assignment constraint M_g >= µ)
        self.G = 3
        n_per_class = self.N_gpus // self.G
        self.gpu_classes = [
            {'name': 'A16', 'memory': 16,  'n_gpus': n_per_class},
            {'name': 'A40', 'memory': 40,  'n_gpus': n_per_class},
            {'name': 'A80', 'memory': 80,  'n_gpus': self.N_gpus - 2 * n_per_class},
        ]

        all_memories = set(gc['memory'] for gc in self.gpu_classes)
        self.eligible_classes = {}
        for mu in all_memories:
            self.eligible_classes[mu] = [
                g for g, gc in enumerate(self.gpu_classes)
                if gc['memory'] >= mu
            ]

        self.n_idle_per_class = [gc['n_gpus'] for gc in self.gpu_classes]

        # Freq Levels
        self.frequencies = {
            0: {'rate': 0.00, 'power': 50},   # Idle (leakage power only)
            1: {'rate': 0.25, 'power': 150},  # 25% speed
            2: {'rate': 0.50, 'power': 200},  # 50% speed
            3: {'rate': 0.75, 'power': 250},  # 75% speed
            4: {'rate': 1.00, 'power': 300},  # 100% speed (max)
        }
        self.K = len(self.frequencies)

        # ── Job Types with Memory Requirements ──────────────────────────────
        self.job_types = {
            'prefill': {
                'tau_r_mean': 10.0,
                'tau_s_mean': 5.0,
                'memory_req': 16,
            },
            'decode': {
                'tau_r_mean': 30.0,
                'tau_s_mean': 20.0,
                'memory_req': 40,
            },
            'fine_tuning': {
                'tau_r_mean': 15.0,
                'tau_s_mean': 10.0,
                'memory_req': 40,
            },
            'training': {
                'tau_r_mean': 40.0,
                'tau_s_mean': 25.0,
                'memory_req': 80,
            },
        }

        for jt in self.job_types.values():
            jt['tau_r_p'] = 1.0 / jt['tau_r_mean']
            jt['tau_s_p'] = 1.0 / jt['tau_s_mean']

        for jt in self.job_types.values():
            mu = jt['memory_req']
            if mu not in self.eligible_classes:
                self.eligible_classes[mu] = [
                    g for g, gc in enumerate(self.gpu_classes)
                    if gc['memory'] >= mu
                ]

        self.lambda_arrival = 70.0
        self.dt     = 1.0
        self.T_sim  = 1000
        self.tau_max = 50
        self.s_max   = 20

        # Power Target — sinusoidal
        self.P_target_base      = 150_000   # W — centre of the wave
        self.P_target_amplitude =  30_000   # W — peak deviation (±30 kW)
        self.P_target_period    =    200    # timesteps — one full cycle
        self.P_target           = self.P_target_base  # initialised for startup

        # Smoothing penalty weight
        self.lambda_smooth = 100

    def get_rate(self, k):
        return self.frequencies[k]['rate']

    def get_power(self, k):
        return self.frequencies[k]['power']

    def class_memory(self, g):
        return self.gpu_classes[g]['memory']

    def class_n_gpus(self, g):
        return self.gpu_classes[g]['n_gpus']

    def eligible_classes_for(self, mu):
        return self.eligible_classes.get(mu, [])


# ============================================================================
# STEP 2: DEFINE THE STATE SPACE
# ============================================================================

class PopulationState:

    def __init__(self, params):
        self.params = params
        self.n = defaultdict(float)
        self.n_idle_per_class = [float(params.class_n_gpus(g))
                                  for g in range(params.G)]
        self.waiting = defaultdict(float)

    @property
    def n_idle(self):
        return sum(self.n_idle_per_class)

    @n_idle.setter
    def n_idle(self, value):
        total = sum(self.n_idle_per_class)
        if total > 0:
            ratio = value / total
            self.n_idle_per_class = [v * ratio for v in self.n_idle_per_class]
        else:
            per = value / self.params.G
            self.n_idle_per_class = [per] * self.params.G

    def get_count(self, tau, s):
        return self.n.get((tau, s), 0.0)

    def set_count(self, tau, s, count):
        if count > 0:
            self.n[(tau, s)] = count
        else:
            self.n.pop((tau, s), None)

    def get_all_states(self):
        return list(self.n.keys())

    def total_busy_gpus(self):
        return sum(self.n.values())

    def total_gpus(self):
        return self.n_idle + self.total_busy_gpus()

    def get_waiting_count(self, tau, s, mu=None):
        if mu is not None:
            return self.waiting.get((tau, s, mu), 0.0)
        return sum(v for (t, sl, m), v in self.waiting.items() if t == tau and sl == s)


# ============================================================================
# STEP 3: JOB ARRIVALS
# ============================================================================

class JobArrivalProcess:

    def __init__(self, params):
        self.params = params
        self.rng = np.random.default_rng(seed=42)
        self.job_type_names = list(params.job_types.keys())
        self.n_types        = len(self.job_type_names)

    def generate_arrivals(self):
        """Returns dict: (tau_r, tau_s, mu) -> int count"""
        n_arrivals      = self.rng.poisson(self.params.lambda_arrival)
        weights         = np.ones(self.n_types) / self.n_types
        counts_per_type = self.rng.multinomial(n_arrivals, weights)

        arrivals = defaultdict(int)
        for jt_name, n_jt in zip(self.job_type_names, counts_per_type):
            jt = self.params.job_types[jt_name]
            mu = jt['memory_req']
            for _ in range(n_jt):
                tau_r = min(self.rng.geometric(jt['tau_r_p']), self.params.tau_max)
                tau_s = min(self.rng.geometric(jt['tau_s_p']), self.params.s_max)
                arrivals[(tau_r, tau_s, mu)] += 1

        return arrivals


# ============================================================================
# STEP 4: CONTROL POLICY (FREQUENCY ASSIGNMENT)
# ============================================================================

import numpy as np
from scipy.optimize import linprog
from collections import defaultdict

class LPControlPolicy:

    def __init__(self, params):
        self.params = params
        self.u_prev = {}

    def compute_control_power(self, state, current_power):
        active_states = []
        for (tau, s) in state.get_all_states():
            if state.get_count(tau, s) > 0 and tau > 0:
                active_states.append((tau, s))

        if not active_states:
            return {}

        fixed_states = []
        free_states  = []
        for (tau, s) in active_states:
            if s == 0:
                fixed_states.append((tau, s))
            else:
                free_states.append((tau, s))

        u = {}

        for (tau, s) in fixed_states:
            for k in range(self.params.K):
                u[(tau, s, k)] = 1.0 if k == 4 else 0.0

        if not free_states:
            return u

        P_idle       = self.params.get_power(0)
        P_idle_total = P_idle * state.n_idle

        P_fixed = 0.0
        for (tau, s) in fixed_states:
            n = state.get_count(tau, s)
            P_fixed += n * self.params.get_power(4)

        P_busy_target = max(0.0, self.params.P_target - P_idle_total - P_fixed)

        K_busy   = [1, 2, 3, 4]
        n_free   = len(free_states)
        n_K      = len(K_busy)
        n_vars_u = n_free * n_K
        idx_e    = n_vars_u
        idx_d    = n_vars_u + 1
        n_vars   = n_vars_u + 1 + n_vars_u

        var_index = {}
        for i in range(n_free):
            tau, s = free_states[i]
            for j in range(n_K):
                k = K_busy[j]
                var_index[(tau, s, k)] = i * n_K + j

        lam = self.params.lambda_smooth
        c = np.zeros(n_vars)
        c[idx_e] = 1.0
        for col in range(n_vars_u):
            c[idx_d + col] = lam

        p_row = np.zeros(n_vars_u)
        for i in range(n_free):
            tau, s = free_states[i]
            n = state.get_count(tau, s)
            for j in range(n_K):
                k   = K_busy[j]
                P_k = self.params.get_power(k)
                col = i * n_K + j
                p_row[col] = n * P_k

        n_ineq = 2 + 2 * n_vars_u
        A_ub = np.zeros((n_ineq, n_vars))
        b_ub = np.zeros(n_ineq)

        A_ub[0, :n_vars_u] =  p_row
        A_ub[0, idx_e]     = -1.0
        b_ub[0]            =  P_busy_target

        A_ub[1, :n_vars_u] = -p_row
        A_ub[1, idx_e]     = -1.0
        b_ub[1]            = -P_busy_target

        for col in range(n_vars_u):
            i_state = col // n_K
            j_freq  = col  % n_K
            tau, s  = free_states[i_state]
            k       = K_busy[j_freq]
            u_p     = self.u_prev.get((tau, s, k), 0.5)

            row_pos = 2 + 2 * col
            row_neg = 2 + 2 * col + 1

            A_ub[row_pos, col]       =  1.0
            A_ub[row_pos, idx_d+col] = -1.0
            b_ub[row_pos]            =  u_p

            A_ub[row_neg, col]       = -1.0
            A_ub[row_neg, idx_d+col] = -1.0
            b_ub[row_neg]            = -u_p

        A_eq = np.zeros((n_free, n_vars))
        b_eq = np.ones(n_free)

        for i in range(n_free):
            for j in range(n_K):
                col = i * n_K + j
                A_eq[i, col] = 1.0

        bounds = []
        for col in range(n_vars_u):
            bounds.append((0.0, 1.0))
        bounds.append((0.0, None))
        for col in range(n_vars_u):
            bounds.append((0.0, None))

        result = linprog(
            c,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds,
            method='highs',
            options={'disp': False}
        )

        if result.success:
            x = result.x
            for i in range(n_free):
                tau, s = free_states[i]
                u[(tau, s, 0)] = 0.0
                for j in range(n_K):
                    k   = K_busy[j]
                    col = var_index[(tau, s, k)]
                    u_val = float(x[col])
                    u_val = max(0.0, min(1.0, u_val))
                    u[(tau, s, k)] = u_val
            self.u_prev = {key: val for key, val in u.items()}
        else:
            print(f"  [LP] Warning: solver status={result.status}, "
                  f"message='{result.message}'. Using uniform fallback.")
            for i in range(n_free):
                tau, s = free_states[i]
                u[(tau, s, 0)] = 0.0
                for j in range(n_K):
                    k = K_busy[j]
                    u[(tau, s, k)] = 1.0 / n_K
            self.u_prev = {key: val for key, val in u.items()}

        return u


# ============================================================================
# STEP 5: STATE TRANSITIONS (POPULATION DYNAMICS)
# ============================================================================

class PopulationDynamics:

    def __init__(self, params):
        self.params = params
        self.total_jobs_missed = 0.0
        self.missed_by_mu      = defaultdict(float)

    def evolve(self, state, control, arrivals, current_power):

        from collections import defaultdict

        new_state = PopulationState(self.params)
        new_state.waiting          = defaultdict(float, state.waiting)
        new_state.n_idle_per_class = list(state.n_idle_per_class)

        # 1. Enqueue new arrivals
        for (tau, s, mu), count in arrivals.items():
            new_state.waiting[(tau, s, mu)] += count

        # 2. Estimate power freed by finishing jobs
        predicted_power_loss = 0.0
        for (tau, s) in state.get_all_states():
            n_current = state.get_count(tau, s)
            if n_current == 0:
                continue
            for k in range(self.params.K):
                u_k = control.get((tau, s, k), 0.0)
                if u_k == 0:
                    continue
                n_at_k   = u_k * n_current
                r_k      = self.params.get_rate(k)
                tau_next = tau - r_k
                if tau_next <= 0:
                    P_k   = self.params.get_power(k)
                    P_idle = self.params.get_power(0)
                    predicted_power_loss += n_at_k * (P_k - P_idle)

        power_gap         = self.params.P_target - current_power
        adjusted_headroom = power_gap + predicted_power_loss

        avg_power_per_new_job = self._estimate_avg_power_for_new_jobs(state, control)
        if avg_power_per_new_job > 0:
            max_new_gpus = max(0.0, adjusted_headroom / avg_power_per_new_job)
        else:
            max_new_gpus = 0.0
        max_new_gpus = min(max_new_gpus, 500)

        # 3. Admit jobs — memory feasibility constraint
        gpus_assigned = 0.0

        def urgency_key(key):
            tau, s, mu = key
            return tau + s

        waiting_keys = sorted(
            [k for k, v in new_state.waiting.items() if v > 0],
            key=urgency_key
        )

        for (tau, s, mu) in waiting_keys:
            if gpus_assigned >= max_new_gpus:
                break
            waiting_count = new_state.waiting.get((tau, s, mu), 0.0)
            if waiting_count <= 0:
                continue
            eligible = self.params.eligible_classes_for(mu)
            for g in eligible:
                if gpus_assigned >= max_new_gpus:
                    break
                if new_state.n_idle_per_class[g] <= 0:
                    continue
                remaining_count = new_state.waiting.get((tau, s, mu), 0.0)
                if remaining_count <= 0:
                    break
                can_assign = min(
                    remaining_count,
                    new_state.n_idle_per_class[g],
                    max_new_gpus - gpus_assigned,
                )
                new_state.waiting[(tau, s, mu)] -= can_assign
                if new_state.waiting[(tau, s, mu)] <= 0:
                    new_state.waiting.pop((tau, s, mu), None)
                new_state.set_count(tau, s, new_state.get_count(tau, s) + can_assign)
                new_state.n_idle_per_class[g] -= can_assign
                gpus_assigned += can_assign

        # 4. Tick slack for remaining waiting jobs
        waiting_next = defaultdict(float)
        for (tau, s, mu), count in new_state.waiting.items():
            if count <= 0:
                continue
            s_next = s - 1
            if s_next < 0:
                self.total_jobs_missed += count
                self.missed_by_mu[mu]  += count
            else:
                waiting_next[(tau, s_next, mu)] += count
        new_state.waiting = waiting_next

        # 5. Advance active jobs
        for (tau, s) in state.get_all_states():
            n_current = state.get_count(tau, s)
            if n_current == 0:
                continue
            for k in range(self.params.K):
                u_k = control.get((tau, s, k), 0.0)
                if u_k == 0:
                    continue
                n_at_k   = u_k * n_current
                r_k      = self.params.get_rate(k)
                tau_next = tau - r_k
                s_next   = s   - (1 - r_k)

                if tau_next <= 0:
                    freed = n_at_k
                    for g in range(self.params.G):
                        share = self.params.class_n_gpus(g) / self.params.N_gpus
                        new_state.n_idle_per_class[g] += freed * share
                else:
                    tau_next = round(tau_next * 4) / 4
                    s_next   = max(0, round(s_next * 4) / 4)
                    new_state.set_count(tau_next, s_next,
                                        new_state.get_count(tau_next, s_next) + n_at_k)

        return new_state

    def _estimate_avg_power_for_new_jobs(self, state, control):
        typical_states = [(10, 5), (5, 2), (20, 10)]
        total_power = 0
        count = 0
        for (tau, s) in typical_states:
            for k in range(self.params.K):
                u_k = control.get((tau, s, k), 0.0)
                if u_k > 0:
                    total_power += u_k * self.params.get_power(k)
                    count += u_k
        return total_power / count if count > 0 else 200.0


# ============================================================================
# STEP 6: POWER CALCULATION
# ============================================================================

class PowerCalculator:

    def __init__(self, params):
        self.params = params

    def compute_power(self, state, control):
        P_idle          = self.params.get_power(0)
        power_from_idle = P_idle * state.n_idle
        power_from_busy = 0.0

        for (tau, s) in state.get_all_states():
            n = state.get_count(tau, s)
            if n == 0:
                continue
            for k in range(self.params.K):
                u_k = control.get((tau, s, k), 0.0)
                if u_k == 0:
                    continue
                power_from_busy += u_k * n * self.params.get_power(k)

        return power_from_idle + power_from_busy


# ============================================================================
# STEP 7: THE SIMULATOR (PUTS IT ALL TOGETHER)
# ============================================================================

class GPUSimulator:

    def __init__(self, params):
        self.params     = params
        self.arrivals   = JobArrivalProcess(params)
        self.policy     = LPControlPolicy(params)
        self.dynamics   = PopulationDynamics(params)
        self.power_calc = PowerCalculator(params)
        self.state      = PopulationState(params)
        self.total_jobs_arrived = 0
        self.history = {
            'time': [], 'power': [], 'n_idle': [],
            'n_busy': [], 'queue_size': [], 'waiting_queue': [],
        }

    def run(self):
        print("Starting simulation...")
        print(f"Target power: {self.params.P_target_base/1000:.1f} kW "
              f"+/- {self.params.P_target_amplitude/1000:.1f} kW "
              f"(period={self.params.P_target_period} steps)")
        print("GPU memory classes:")
        for g, gc in enumerate(self.params.gpu_classes):
            print(f"  Class {g} ({gc['name']}, {gc['memory']} GB): {gc['n_gpus']} GPUs")
        print("Job types and memory requirements:")
        for jt_name, jt in self.params.job_types.items():
            eligible = [self.params.gpu_classes[g]['name']
                        for g in self.params.eligible_classes_for(jt['memory_req'])]
            print(f"  {jt_name}: mu={jt['memory_req']} GB -> eligible: {eligible}")
        print()

        power_prev = self.params.get_power(0) * self.params.N_gpus

        for t in range(self.params.T_sim):
            # Update sinusoidal target for this timestep
            self.params.P_target = (
                self.params.P_target_base
                + self.params.P_target_amplitude
                * np.sin(2 * np.pi * t / self.params.P_target_period)
            )

            arrivals_t = self.arrivals.generate_arrivals()
            self.total_jobs_arrived += sum(arrivals_t.values())

            control_t  = self.policy.compute_control_power(self.state, power_prev)
            self.state = self.dynamics.evolve(self.state, control_t, arrivals_t, power_prev)
            power_t    = self.power_calc.compute_power(self.state, control_t)
            power_prev = power_t

            self.history['time'].append(t)
            self.history['power'].append(power_t)
            self.history['n_idle'].append(self.state.n_idle)
            self.history['n_busy'].append(self.state.total_busy_gpus())
            self.history['queue_size'].append(len(self.state.get_all_states()))
            self.history['waiting_queue'].append(sum(self.state.waiting.values()))

            if t % 10 == 0:
                idle_by_class = ", ".join(
                    f"{self.params.gpu_classes[g]['name']}:{self.state.n_idle_per_class[g]:.0f}"
                    for g in range(self.params.G)
                )
                print(f"t={t:3d}: Power={power_t/1000:6.1f} kW, "
                      f"Target={self.params.P_target/1000:6.1f} kW, "
                      f"Idle=[{idle_by_class}], "
                      f"Busy={self.state.total_busy_gpus():5.0f}")

        print()
        print("Simulation complete!")

        missed       = self.dynamics.total_jobs_missed
        missed_by_mu = self.dynamics.missed_by_mu
        print()
        print("=" * 55)
        print("JOB DEADLINE SUMMARY")
        print("=" * 55)
        print(f"  Total jobs arrived : {self.total_jobs_arrived:,.0f}")
        print(f"  Jobs missed        : {missed:,.0f}", end="")
        if self.total_jobs_arrived > 0:
            print(f"  ({100 * missed / self.total_jobs_arrived:.1f}%)")
        else:
            print()
        if missed > 0:
            print("  Breakdown by memory requirement:")
            for mu in sorted(missed_by_mu.keys()):
                print(f"    {mu:>3d} GB : {missed_by_mu[mu]:,.0f}  "
                      f"({100 * missed_by_mu[mu] / missed:.1f}% of missed)")
        print("=" * 55)

    def plot_results(self):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        T = self.history['time']
        P_wave = [
            (self.params.P_target_base + self.params.P_target_amplitude
             * np.sin(2 * np.pi * t / self.params.P_target_period)) / 1000
            for t in T
        ]

        # Plot 1: Power consumption
        ax = axes[0]
        ax.plot(T, [p/1000 for p in self.history['power']],
                label='Actual Power', linewidth=2)
        ax.plot(T, P_wave, color='r', linestyle='--',
                label='Target (sinusoidal)', linewidth=2)
        ax.set_ylabel('Power (kW)', fontsize=12)
        ax.set_title('Power Consumption Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 2: GPU utilization
        ax = axes[1]
        ax.plot(T, self.history['n_idle'], label='Idle GPUs', linewidth=2)
        ax.plot(T, self.history['n_busy'], label='Busy GPUs', linewidth=2)
        ax.axhline(self.params.N_gpus, color='k', linestyle='--', alpha=0.5,
                   label=f'Total ({self.params.N_gpus})')
        ax.set_ylabel('Number of GPUs', fontsize=12)
        ax.set_title('GPU Utilization', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 3: Waiting Queue
        ax = axes[2]
        ax.plot(T, self.history['waiting_queue'], linewidth=2, color='orange')
        ax.set_ylabel('Jobs Waiting', fontsize=12)
        ax.set_title('Waiting Queue Size', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.fill_between(T, 0, self.history['waiting_queue'], alpha=0.3, color='orange')

        plt.tight_layout()
        plt.savefig(r'C:/Users/lauso/OneDrive/Desktop/Cornell/AI flexible demand response/New model images/simulation_results_LP.png',
                    dpi=150, bbox_inches='tight')
        print("Plot saved to simulation_results_LP.png")
        return fig


# ============================================================================
# STEP 8: RUN IT!
# ============================================================================

if __name__ == "__main__":
    params = GPUSystemParameters()
    sim    = GPUSimulator(params)
    sim.run()
    sim.plot_results()

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    avg_power = np.mean(sim.history['power'])
    std_power = np.std(sim.history['power'])
    max_power = np.max(sim.history['power'])
    min_power = np.min(sim.history['power'])

    print(f"Average power:  {avg_power/1000:.2f} kW")
    print(f"Std dev:        {std_power/1000:.2f} kW")
    print(f"Max power:      {max_power/1000:.2f} kW")
    print(f"Min power:      {min_power/1000:.2f} kW")
    print(f"Target base:    {params.P_target_base/1000:.2f} kW")
    print(f"Deviation:      {abs(avg_power - params.P_target_base)/1000:.2f} kW")

    avg_util = np.mean(sim.history['n_busy']) / params.N_gpus * 100
    print(f"\nAverage GPU utilization: {avg_util:.1f}%")
    print(f"Average idle GPUs: {np.mean(sim.history['n_idle']):.0f}")
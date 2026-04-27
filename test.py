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

        # Precompute eligible GPU class indices for each memory requirement.
        # eligible_classes[mu] = [g_index, ...] sorted ascending (cheapest-fit first).
        all_memories = set(gc['memory'] for gc in self.gpu_classes)
        self.eligible_classes = {}
        for mu in all_memories:
            self.eligible_classes[mu] = [
                g for g, gc in enumerate(self.gpu_classes)
                if gc['memory'] >= mu
            ]

        # Idle GPU count per class (updated at runtime via PopulationState)
        # These are the *initial* counts at t=0.
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
        # memory_req (µ, GB): job can only be assigned to class g where m_g >= µ.
        #   prefill     : µ=16 GB  → eligible: A16, A40, A80 (all classes)
        #   decode      : µ=40 GB  → eligible: A40, A80
        #   fine_tuning : µ=40 GB  → eligible: A40, A80
        #   training    : µ=80 GB  → eligible: A80 only
        self.job_types = {
            'prefill': {
                'tau_r_mean': 2.0,   # F1 fix: was 10.0 (report1 §2)
                'tau_s_mean': 1.0,   # F1 fix: was 5.0
                'memory_req': 16,   # fits any GPU class
            },
            'decode': {
                'tau_r_mean': 30.0,
                'tau_s_mean': 20.0,
                'memory_req': 40,   # needs mid- or high-memory GPU
            },
            'fine_tuning': {
        
                'tau_r_mean': 15.0,
                'tau_s_mean': 10.0,
                'memory_req': 40,   # same as decode
            },
            'training': {
                'tau_r_mean': 40.0,
                'tau_s_mean': 25.0,
                'memory_req': 80,   # largest: only A80 class
            },
        }

        for jt in self.job_types.values():
            jt['tau_r_p'] = 1.0 / jt['tau_r_mean']
            jt['tau_s_p'] = 1.0 / jt['tau_s_mean']

        # Register every unique memory_req that appears in job_types so
        # eligible_classes covers them (they may not equal a class memory exactly).
        for jt in self.job_types.values():
            mu = jt['memory_req']
            if mu not in self.eligible_classes:
                self.eligible_classes[mu] = [
                    g for g, gc in enumerate(self.gpu_classes)
                    if gc['memory'] >= mu
                ]
       
        self.lambda_arrival = 16.0   

        self.dt = 1.0  
        self.T_sim = 1000 
        
        # Power Target 
        self.P_target = 150_000  #W 
        
     
        self.tau_max = 50  
        self.s_max = 20    

        # --- Smoothing penalty weight ---
        # Penalizes |u[i,k] - u_prev[i,k]| across timesteps.
        # Increase to get smoother (but slower-tracking) control.
        # Start around 100 given power is in ~100k W scale.
        self.lambda_smooth = 100
        
    def get_rate(self, k):
        return self.frequencies[k]['rate']
    
    def get_power(self, k):
        return self.frequencies[k]['power']

    def class_memory(self, g):
        """Return GPU memory capacity (GB) for class g."""
        return self.gpu_classes[g]['memory']

    def class_n_gpus(self, g):
        """Return fleet size for class g."""
        return self.gpu_classes[g]['n_gpus']

    def eligible_classes_for(self, mu):
        """Return list of GPU class indices eligible for a job with memory_req µ."""
        return self.eligible_classes.get(mu, [])


# ============================================================================
# STEP 2: DEFINE THE STATE SPACE
# ============================================================================

class PopulationState:
    """
    Tracks GPU population.

    Memory-class extension
    ----------------------
    * n_idle_per_class[g] : idle GPU count for class g
    * n_idle              : total idle GPUs (sum over classes, kept for compatibility)
    * waiting             : dict keyed by (tau, s, mu) -> float count
                            µ (memory_req) is stored so the admission step can
                            enforce the memory feasibility constraint m_g >= µ.
    * n                   : dict keyed by (tau, s) -> float count
                            (aggregate active busy count, class-agnostic for the LP)
    """

    def __init__(self, params):
        self.params = params

        self.n = defaultdict(float)

        # Idle GPU counts per class — initialised from params at t=0.
        self.n_idle_per_class = [float(params.class_n_gpus(g))
                                  for g in range(params.G)]

        # Waiting queue: keyed by (tau, s, mu) so admission can filter by memory.
        self.waiting = defaultdict(float)

    @property
    def n_idle(self):
        """Total idle GPUs across all classes."""
        return sum(self.n_idle_per_class)

    @n_idle.setter
    def n_idle(self, value):
        """Legacy setter — sets class-0 idle only. Prefer n_idle_per_class."""
        # kept for any code that still writes state.n_idle = X directly
        total = sum(self.n_idle_per_class)
        if total > 0:
            ratio = value / total
            self.n_idle_per_class = [v * ratio for v in self.n_idle_per_class]
        else:
            # distribute evenly
            per = value / self.params.G
            self.n_idle_per_class = [per] * self.params.G

    # ── Active-state helpers (class-agnostic, used by LP) ─────────────────

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
        # Legacy: sum over all µ for this (tau, s)
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
        """Returns dict: (tau_r, tau_s, mu, job_type) -> int count

        F3: job_type name is now included in the key so that PopulationDynamics
        can track missed jobs broken down by type, not just by memory class.
        """
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
                arrivals[(tau_r, tau_s, mu, jt_name)] += 1   # F3: type in key

        return arrivals
    
    
    
    def generate_arrivals_2(self):
       
       
        n_arrivals = self.rng.poisson(self.params.lambda_arrival)
        
        arrivals = defaultdict(int)
        
        for _ in range(n_arrivals):
        
            tau_r = self.rng.geometric(self.params.tau_r_p)
            tau_r = min(tau_r, self.params.tau_max)  # Cap at max
            
            
            tau_s = self.rng.geometric(self.params.tau_s_p)
            tau_s = min(tau_s, self.params.s_max)  # Cap at max
            
           
            arrivals[(tau_r, tau_s)] += 1
            
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
        # Memory: stores u solution from the previous timestep.
        # Keys: (tau, s, k) -> float in [0, 1]
        # Used to penalize large changes in u between timesteps.
        self.u_prev = {}

    def compute_control_power(self, state, current_power):
        active_states = []
        for (tau, s) in state.get_all_states():
            if state.get_count(tau, s) > 0 and tau > 0:
                active_states.append((tau, s))
 
        if not active_states:
            return {}
        
        #fixed states have slack of 0 therefore have to be run at 100% frequency
        #free states can oscillate between ks [1,2,3]
        fixed_states = []
        free_states  = []
        for (tau, s) in active_states:
            if s == 0:
                fixed_states.append((tau, s))
            else:
                free_states.append((tau, s))
        
        u = {}

        #deterministic assignment because 100% of fixed states have to run at max frequency
        for (tau, s) in fixed_states:
            for k in range(self.params.K):
                if k == 4:
                    u[(tau, s, k)] = 1.0
                else:
                    u[(tau, s, k)] = 0.0
 
        if not free_states:
            # All states are deadline-critical; LP not needed
            return u

        P_idle = self.params.get_power(0)
        P_idle_total = P_idle * state.n_idle
 
        P_fixed = 0.0
        for (tau, s) in fixed_states:
            n = state.get_count(tau, s)
            P_fixed = P_fixed + n * self.params.get_power(4)  # forced at k=4
 
        #Subtract P_idle ad P_fixed_states from my P_target
        P_busy_target = max(0.0, self.params.P_target - P_idle_total - P_fixed)

        ##---------- LP variables----------------------------
        # Variable layout:
        #   [u_vars (n_free * n_K)] | [e (1)] | [d_vars (n_free * n_K)]
        #
        # u[i,k]  : fraction of state i running at frequency k  (n_free * n_K vars)
        # e       : absolute power tracking error               (1 var)
        # d[i,k]  : |u[i,k] - u_prev[i,k]|, the change penalty (n_free * n_K vars)

        K_busy = [1, 2, 3, 4]  
        n_free   = len(free_states)
        n_K      = len(K_busy)
        n_vars_u = n_free * n_K
        idx_e    = n_vars_u          # Index of the slack variable e
        idx_d    = n_vars_u + 1      # Start index of d variables
        n_vars   = n_vars_u + 1 + n_vars_u  # u + e + d

        # Map (tau, s, k) -> column index for u and d blocks
        var_index = {}
        for i in range(n_free):
            tau, s = free_states[i]
            for j in range(n_K):
                k = K_busy[j]
                var_index[(tau, s, k)] = i * n_K + j

        # ----------------------------------------------------------------
        # OBJECTIVE: minimize  e  +  lambda * sum(d[i,k])
        #
        # The lambda_smooth weight trades off:
        #   - small lambda -> tracks power well, may be jumpy
        #   - large lambda -> smoother u, may miss power target slightly
        # ----------------------------------------------------------------
        lam = self.params.lambda_smooth
        c = np.zeros(n_vars)
        c[idx_e] = 1.0                          # penalize power error
        for col in range(n_vars_u):
            c[idx_d + col] = lam                # penalize u changes


        # p_row[col]: power contributed per unit of u variable col
        p_row = np.zeros(n_vars_u)
        for i in range(n_free):
            tau, s = free_states[i]
            n = state.get_count(tau, s)
            for j in range(n_K):
                k   = K_busy[j]
                P_k = self.params.get_power(k)
                col = i * n_K + j
                p_row[col] = n * P_k

        # ----------------------------------------------------------------
        # INEQUALITY CONSTRAINTS  A_ub @ x <= b_ub
        #
        # Power tracking (rows 0-1):
        #   Row 0:  p_row @ u  - e           <=  P_busy_target
        #   Row 1: -p_row @ u  - e           <= -P_busy_target
        #
        # Smoothing / linearized absolute value (rows 2 to 2+2*n_vars_u-1):
        #   For each variable col in [0, n_vars_u):
        #     u[col] - u_prev[col]  - d[col] <= 0   =>  d >= u - u_prev
        #    -u[col] + u_prev[col]  - d[col] <= 0   =>  d >= -(u - u_prev)
        # ----------------------------------------------------------------
        # Build sparse constraint matrices (F5-friendly: avoids dense OOM at large state counts)
        from scipy.sparse import lil_matrix, csc_matrix as _csc
        n_ineq = 2 + 2 * n_vars_u
        A_ub_sp = lil_matrix((n_ineq, n_vars))
        b_ub    = np.zeros(n_ineq)

        for col in range(n_vars_u):
            A_ub_sp[0, col] =  p_row[col]
            A_ub_sp[1, col] = -p_row[col]
        A_ub_sp[0, idx_e] = -1.0;  b_ub[0] =  P_busy_target
        A_ub_sp[1, idx_e] = -1.0;  b_ub[1] = -P_busy_target

        for col in range(n_vars_u):
            i_state = col // n_K
            j_freq  = col  % n_K
            tau, s  = free_states[i_state]
            k       = K_busy[j_freq]
            u_p     = self.u_prev.get((tau, s, k), 0.5)
            row_pos = 2 + 2 * col
            row_neg = 3 + 2 * col
            A_ub_sp[row_pos, col]       =  1.0;  A_ub_sp[row_pos, idx_d + col] = -1.0;  b_ub[row_pos] =  u_p
            A_ub_sp[row_neg, col]       = -1.0;  A_ub_sp[row_neg, idx_d + col] = -1.0;  b_ub[row_neg] = -u_p

        A_eq_sp = lil_matrix((n_free, n_vars))
        b_eq    = np.ones(n_free)
        for i in range(n_free):
            for j in range(n_K):
                A_eq_sp[i, i * n_K + j] = 1.0

        bounds = ([(0.0, 1.0)] * n_vars_u) + [(0.0, None)] + ([(0.0, None)] * n_vars_u)

        result = linprog(
            c,
            A_ub=_csc(A_ub_sp), b_ub=b_ub,
            A_eq=_csc(A_eq_sp), b_eq=b_eq,
            bounds=bounds,
            method='highs',
            options={'disp': False}
        )
 
       
        if result.success:
            x = result.x
            for i in range(n_free):
                tau, s = free_states[i]
                u[(tau, s, 0)] = 0.0  # k=0 not available to busy GPUs
                for j in range(n_K):
                    k   = K_busy[j]
                    col = var_index[(tau, s, k)]
                    u_val = float(x[col])
                    # Clip for numerical safety (solver may return tiny negatives)
                    if u_val < 0.0:
                        u_val = 0.0
                    if u_val > 1.0:
                        u_val = 1.0
                    u[(tau, s, k)] = u_val

            # --- Store solution as u_prev for next timestep ---
            self.u_prev = {key: val for key, val in u.items()}

        else:
            # Fallback: uniform frequency split across all free states.
            # u = 0.25 for each of k=1,2,3,4 always satisfies sum=1.
            print(f"  [LP] Warning: solver status={result.status}, "
                  f"message='{result.message}'. Using uniform fallback.")
            for i in range(n_free):
                tau, s = free_states[i]
                u[(tau, s, 0)] = 0.0
                for j in range(n_K):
                    k = K_busy[j]
                    u[(tau, s, k)] = 1.0 / n_K

            # Store fallback as u_prev too
            self.u_prev = {key: val for key, val in u.items()}
 
        return u
 

# ============================================================================
# STEP 5: STATE TRANSITIONS (POPULATION DYNAMICS)
# ============================================================================

class PopulationDynamics:
   #holds equation 10 of technical note
    
    def __init__(self, params):
        self.params = params
        # ── Job tracking counters ─────────────────────────────────────────
        self.total_jobs_missed    = 0.0
        self.missed_by_mu         = defaultdict(float)   # mu -> count
        self.missed_by_type       = defaultdict(float)   # F3: job_type -> count
        self.total_jobs_admitted  = 0.0                  # F5: conservation check 2
        self.total_jobs_completed = 0.0                  # F5: conservation check 3
        
    def evolve(self, state, control, arrivals, current_power):

        from collections import defaultdict
    
        new_state = PopulationState(self.params)
        new_state.waiting = defaultdict(float, state.waiting)
        # Copy per-class idle counts from previous state
        new_state.n_idle_per_class = list(state.n_idle_per_class)

        # ── 1. Enqueue new arrivals (tau, s, mu, job_type) ───────────────
        for (tau, s, mu, jtype), count in arrivals.items():
            new_state.waiting[(tau, s, mu, jtype)] += count

        # ── 2. Estimate power freed by finishing jobs ──────────────────────
        predicted_power_loss = 0.0

        for (tau, s) in state.get_all_states():
            n_current = state.get_count(tau, s)
            if n_current == 0:
                continue
            for k in range(self.params.K):
                u_k = control.get((tau, s, k), 0.0)
                if u_k == 0:
                    continue
                n_at_k = u_k * n_current
                r_k    = self.params.get_rate(k)
                tau_next = tau - r_k
                if tau_next <= 0:
                    P_k   = self.params.get_power(k)
                    P_idle = self.params.get_power(0)
                    predicted_power_loss += n_at_k * (P_k - P_idle)

        power_gap        = self.params.P_target - current_power
        adjusted_headroom = power_gap + predicted_power_loss

        avg_power_per_new_job = self._estimate_avg_power_for_new_jobs(state, control)
        if avg_power_per_new_job > 0:
            max_new_gpus = max(0.0, adjusted_headroom / avg_power_per_new_job)
        else:
            max_new_gpus = 0.0
        max_new_gpus = min(max_new_gpus, 500)

        # ── 3. Admit jobs — MEMORY FEASIBILITY CONSTRAINT ─────────────────
        #
        # Technical note §5.1, eq.(11): a job with memory requirement µ can
        # only be assigned to an idle GPU of class g where  m_g >= µ.
        #
        # Policy: cheapest-fit — within feasible classes, prefer the lowest-
        # memory class that still satisfies the constraint (smallest m_g >= µ).
        # This preserves high-memory GPUs for jobs that actually need them.
        #
        # Waiting jobs are sorted by urgency (ascending slack ratio s/(τ+s)):
        # most urgent jobs are admitted first when headroom is limited.

        gpus_assigned = 0.0

        # Sort waiting jobs: most urgent first (lowest slack ratio)
        def urgency_key(key):
            tau, s, mu, jtype = key   # F3: 4-tuple
            return tau + s            # EDF: ascending deadline

        waiting_keys = sorted(
            [k for k, v in new_state.waiting.items() if v > 0],
            key=urgency_key
        )

        for (tau, s, mu, jtype) in waiting_keys:
            if gpus_assigned >= max_new_gpus:
                break

            waiting_count = new_state.waiting.get((tau, s, mu, jtype), 0.0)
            if waiting_count <= 0:
                continue
            eligible = self.params.eligible_classes_for(mu)

            # Try to admit across eligible classes (cheapest-fit first)
            for g in eligible:
                if gpus_assigned >= max_new_gpus:
                    break
                if new_state.n_idle_per_class[g] <= 0:
                    continue

                remaining_count = new_state.waiting.get((tau, s, mu, jtype), 0.0)
                if remaining_count <= 0:
                    break

                can_assign = min(
                    remaining_count,
                    new_state.n_idle_per_class[g],
                    max_new_gpus - gpus_assigned,
                )

                new_state.waiting[(tau, s, mu, jtype)] -= can_assign
                if new_state.waiting[(tau, s, mu, jtype)] <= 0:
                    new_state.waiting.pop((tau, s, mu, jtype), None)

                new_state.set_count(tau, s, new_state.get_count(tau, s) + can_assign)
                new_state.n_idle_per_class[g] -= can_assign
                gpus_assigned += can_assign
                self.total_jobs_admitted += can_assign   # F5 check 2

        # ── 4. Tick slack for remaining waiting jobs ───────────────────────
        waiting_next = defaultdict(float)
        for (tau, s, mu, jtype), count in new_state.waiting.items():
            if count <= 0:
                continue
            s_next = s - 1
            if s_next < 0:
                # F3: record miss by both mu and job type
                self.total_jobs_missed    += count
                self.missed_by_mu[mu]     += count
                self.missed_by_type[jtype] += count   # F3 new
            else:
                waiting_next[(tau, s_next, mu, jtype)] += count
        new_state.waiting = waiting_next

        # ── 5. Advance active jobs ─────────────────────────────────────────
        for (tau, s) in state.get_all_states():
            n_current = state.get_count(tau, s)
            if n_current == 0:
                continue
            for k in range(self.params.K):
                u_k = control.get((tau, s, k), 0.0)
                if u_k == 0:
                    continue
                n_at_k = u_k * n_current
                r_k    = self.params.get_rate(k)
                tau_next = tau - r_k
                s_next   = s   - (1 - r_k)

                if tau_next <= 0:
                    # Job finished — return GPUs proportionally across classes
                    freed = n_at_k
                    self.total_jobs_completed += freed   # F5 check 3
                    total_busy = max(state.total_busy_gpus(), 1.0)
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
        """
        Estimate average power consumption for newly admitted jobs.
    
        Strategy: Look at current control policy to see what frequency
        new jobs with typical slack would run at.
        """
        # Sample some typical new job states
        typical_states = [
        (10, 5),   # Medium job, medium slack
        (5, 2),    # Short job, low slack
        (20, 10),  # Long job, high slack
        ]
    
        total_power = 0
        count = 0
    
        for (tau, s) in typical_states:
            # What frequency would control assign to this state?
            for k in range(self.params.K):
                u_k = control.get((tau, s, k), 0.0)
                if u_k > 0:
                    P_k = self.params.get_power(k)
                    total_power += u_k * P_k
                    count += u_k
    
        if count > 0:
            return total_power / count
        else:
            # Fallback: assume 50% frequency
            return 200.0



# ============================================================================
# STEP 6: POWER CALCULATION
# ============================================================================

class PowerCalculator:
    """
    This implements Equation 16 from the technical note.
    """
    
    def __init__(self, params):
        self.params = params
        
    def compute_power(self, state, control):
        """
        Calculate total power consumption.
        
        P_total = P_idle * n_idle + ∑_{tau,s} ∑_k u[tau,s,k] * n[tau,s] * P_k
        
        Returns:
            power in Watts
        """
        # Idle GPU power
        P_idle = self.params.get_power(0)
        power_from_idle = P_idle * state.n_idle
        
        # Busy GPU power
        power_from_busy = 0.0
        
        for (tau, s) in state.get_all_states():
            n = state.get_count(tau, s)
            
            if n == 0:
                continue
            
            for k in range(self.params.K):
                u_k = control.get((tau, s, k), 0.0)
                
                if u_k == 0:
                    continue
                
                P_k = self.params.get_power(k)
                power_from_busy += u_k * n * P_k
        
        return power_from_idle + power_from_busy


# ============================================================================
# STEP 7: THE SIMULATOR (PUTS IT ALL TOGETHER)
# ============================================================================

class GPUSimulator:
    """
    Main simulation loop.
    Coordinates all the pieces: arrivals, control, dynamics, power.
    """
    
    def __init__(self, params):
        self.params = params
        
        # Initialize components
        self.arrivals = JobArrivalProcess(params)
        self.policy = LPControlPolicy(params)
        self.dynamics = PopulationDynamics(params)
        self.power_calc = PowerCalculator(params)
        
        # Initial state
        self.state = PopulationState(params)
        
        # Job arrival/miss counters
        self.total_jobs_arrived = 0
        # F5: conservation violation counters
        self.conservation_violations = 0

        # History (for plotting)
        self.history = {
            'time': [],
            'power': [],
            'n_idle': [],
            'n_busy': [],
            'queue_size': [],
            'waiting_queue': [],
            # per-class idle counts (list of G-length lists, one per timestep)
            'n_idle_per_class': [],
        }
        
    def run(self):
        """Run the simulation for T time steps"""
        
        print("Starting simulation...")
        print(f"Target power: {self.params.P_target/1000:.1f} kW")
        print(f"GPU memory classes:")
        for g, gc in enumerate(self.params.gpu_classes):
            print(f"  Class {g} ({gc['name']}, {gc['memory']} GB): {gc['n_gpus']} GPUs")
        print(f"Job types and memory requirements:")
        for jt_name, jt in self.params.job_types.items():
            eligible = [self.params.gpu_classes[g]['name']
                        for g in self.params.eligible_classes_for(jt['memory_req'])]
            print(f"  {jt_name}: µ={jt['memory_req']} GB → eligible: {eligible}")
        print()

        power_prev = self.params.get_power(0) * self.params.N_gpus
        
        for t in range(self.params.T_sim):
            arrivals_t = self.arrivals.generate_arrivals()
            self.total_jobs_arrived += sum(arrivals_t.values())
            
            control_t = self.policy.compute_control_power(self.state, power_prev)

            self.state = self.dynamics.evolve(self.state, control_t, arrivals_t, power_prev)

            # ── F5: Conservation checks (every slot) ──────────────────────
            # Check 1: GPU conservation per class
            #   n_idle_g + busy_g (estimated as proportional share) ≈ N_g
            #   We use total idle + total busy = N_gpus as the aggregate check
            total_accounted = self.state.n_idle + self.state.total_busy_gpus()
            gpu_discrepancy = abs(total_accounted - self.params.N_gpus)
            if gpu_discrepancy > 1.0:   # tolerance of 1 GPU for floating point
                self.conservation_violations += 1
                if self.conservation_violations <= 5:   # print first 5 only
                    print(f"  [F5 VIOLATION t={t}] GPU conservation: "
                          f"idle({self.state.n_idle:.1f}) + "
                          f"busy({self.state.total_busy_gpus():.1f}) = "
                          f"{total_accounted:.1f} != N_gpus({self.params.N_gpus})")

            # Check 2: Arrival accounting (cumulative)
            #   total_arrived >= total_admitted + total_missed + currently_queued
            currently_queued = sum(self.state.waiting.values())
            accounted = (self.dynamics.total_jobs_admitted
                         + self.dynamics.total_jobs_missed
                         + currently_queued)
            arrival_discrepancy = abs(self.total_jobs_arrived - accounted)
            if arrival_discrepancy > 1.0:
                self.conservation_violations += 1
                if self.conservation_violations <= 5:
                    print(f"  [F5 VIOLATION t={t}] Arrival accounting: "
                          f"arrived({self.total_jobs_arrived:.0f}) != "
                          f"admitted({self.dynamics.total_jobs_admitted:.0f}) + "
                          f"missed({self.dynamics.total_jobs_missed:.0f}) + "
                          f"queued({currently_queued:.0f}) = {accounted:.0f}")

            # Check 3: Job population (cumulative)
            #   total_arrived = total_completed + total_missed + currently_in_system
            in_system = self.state.total_busy_gpus() + currently_queued
            population = (self.dynamics.total_jobs_completed
                          + self.dynamics.total_jobs_missed
                          + in_system)
            pop_discrepancy = abs(self.total_jobs_arrived - population)
            if pop_discrepancy > 1.0:
                self.conservation_violations += 1
                if self.conservation_violations <= 5:
                    print(f"  [F5 VIOLATION t={t}] Population: "
                          f"arrived({self.total_jobs_arrived:.0f}) != "
                          f"completed({self.dynamics.total_jobs_completed:.0f}) + "
                          f"missed({self.dynamics.total_jobs_missed:.0f}) + "
                          f"in_system({in_system:.0f}) = {population:.0f}")
            # ── end F5 checks ─────────────────────────────────────────────

            power_t = self.power_calc.compute_power(self.state, control_t)
            power_prev = power_t
            
            self.history['time'].append(t)
            self.history['power'].append(power_t)
            self.history['n_idle'].append(self.state.n_idle)
            self.history['n_busy'].append(self.state.total_busy_gpus())
            self.history['queue_size'].append(len(self.state.get_all_states()))
            waiting_total = sum(self.state.waiting.values())
            self.history['waiting_queue'].append(waiting_total)
            self.history['n_idle_per_class'].append(
                list(self.state.n_idle_per_class))
            
            if t % 10 == 0:
                idle_by_class = ", ".join(
                    f"{self.params.gpu_classes[g]['name']}:{self.state.n_idle_per_class[g]:.0f}"
                    for g in range(self.params.G)
                )
                print(f"t={t:3d}: Power={power_t/1000:6.1f} kW, "
                      f"Idle=[{idle_by_class}], "
                      f"Busy={self.state.total_busy_gpus():5.0f}, "
                      f"States={len(self.state.get_all_states())}")
        
        print()
        print("Simulation complete!")

        # ── Job arrival / miss summary (F3 enhanced) ─────────────────────
        missed          = self.dynamics.total_jobs_missed
        missed_by_mu    = self.dynamics.missed_by_mu
        missed_by_type  = self.dynamics.missed_by_type
        admitted        = self.dynamics.total_jobs_admitted
        completed       = self.dynamics.total_jobs_completed
        print()
        print("=" * 60)
        print("JOB SUMMARY  (F3 + F5)")
        print("=" * 60)
        print(f"  Total arrived      : {self.total_jobs_arrived:,.0f}")
        print(f"  Total admitted     : {admitted:,.0f}")
        print(f"  Total completed    : {completed:,.0f}")
        print(f"  Total missed       : {missed:,.0f}", end="")
        if self.total_jobs_arrived > 0:
            print(f"  ({100 * missed / self.total_jobs_arrived:.1f}%)")
        else:
            print()
        print()
        if missed > 0:
            print("  F3 — Miss breakdown by memory class:")
            for mu in sorted(missed_by_mu.keys()):
                pct_of_missed = 100 * missed_by_mu[mu] / missed
                pct_of_total  = 100 * missed_by_mu[mu] / max(self.total_jobs_arrived, 1)
                print(f"    µ={mu:>3d} GB : {missed_by_mu[mu]:>8,.0f}  "
                      f"({pct_of_missed:.1f}% of missed, {pct_of_total:.1f}% of arrived)")
            print()
            print("  F3 — Miss breakdown by job type:")
            for jtype in ['prefill', 'decode', 'fine_tuning', 'training']:
                count = missed_by_type.get(jtype, 0.0)
                pct_of_missed = 100 * count / missed if missed > 0 else 0.0
                pct_of_total  = 100 * count / max(self.total_jobs_arrived, 1)
                print(f"    {jtype:<12}: {count:>8,.0f}  "
                      f"({pct_of_missed:.1f}% of missed, {pct_of_total:.1f}% of arrived)")
        print()
        print("  F5 — Conservation check summary:")
        if self.conservation_violations == 0:
            print("    All checks PASSED (0 violations)")
        else:
            print(f"    WARNING: {self.conservation_violations} violation(s) detected "
                  f"— see [F5 VIOLATION] lines above")
        print("=" * 60)
        
    def plot_results(self):
        """Plot the simulation results"""
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Power consumption
        ax = axes[0]
        ax.plot(self.history['time'], 
                [p/1000 for p in self.history['power']], 
                label='Actual Power', linewidth=2)
        ax.axhline(self.params.P_target/1000, 
                   color='r', linestyle='--', 
                   label=f'Target ({self.params.P_target/1000:.0f} kW)', linewidth=2)
        ax.set_ylabel('Power (kW)', fontsize=12)
        ax.set_title('Power Consumption Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: GPU utilization
        ax = axes[1]
        ax.plot(self.history['time'], self.history['n_idle'], 
                label='Idle GPUs', linewidth=2)
        ax.plot(self.history['time'], self.history['n_busy'], 
                label='Busy GPUs', linewidth=2)
        ax.axhline(self.params.N_gpus, 
                   color='k', linestyle='--', alpha=0.5,
                   label=f'Total ({self.params.N_gpus})')
        ax.set_ylabel('Number of GPUs', fontsize=12)
        ax.set_title('GPU Utilization', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        #Waiting Queue
        ax = axes[2]
        ax.plot(self.history['time'], self.history['waiting_queue'], 
            linewidth=2, color='orange')
        ax.set_ylabel('Jobs Waiting', fontsize=12)
        ax.set_title('Waiting Queue Size', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.fill_between(self.history['time'], 0, self.history['waiting_queue'], 
                     alpha=0.3, color='orange')
        
        plt.tight_layout()
        import os
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation_results_LP.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {out_path}")
        
        return fig



    def save_run_metrics(self):
        """
        Save a JSON summary of this run keyed by lambda value.
        File: metrics_lambda_{lambda}.json in same directory as script.
        plot_section3() reads all such files to build the Section 3 figure.
        """
        import json, os
        import numpy as np

        lam   = self.params.lambda_arrival
        power = self.history['power']
        queue = self.history['waiting_queue']
        busy  = self.history['n_busy']
        missed         = self.dynamics.total_jobs_missed
        arrived        = self.total_jobs_arrived
        missed_by_type = dict(self.dynamics.missed_by_type)
        missed_by_mu   = {str(k): v for k, v in self.dynamics.missed_by_mu.items()}

        # Per-class mean busy GPUs
        # n_idle_per_class[t][g] -> mean idle per class -> busy = N_g - idle
        idle_arr = np.array(self.history['n_idle_per_class'])  # shape (T, G)
        mean_busy_per_class = {}
        for g, gc in enumerate(self.params.gpu_classes):
            mean_idle_g  = float(np.mean(idle_arr[:, g]))
            mean_busy_per_class[gc['name']] = float(gc['n_gpus'] - mean_idle_g)

        metrics = {
            'lambda':            lam,
            'miss_rate':         float(missed / max(arrived, 1)),
            'rms_error_kw':      float(np.sqrt(np.mean(
                                     [(p - self.params.P_target)**2 for p in power]
                                 )) / 1000),
            'mean_queue':        float(np.mean(queue)),
            'mean_busy_total':   float(np.mean(busy)),
            'mean_busy_per_class': mean_busy_per_class,
            'total_arrived':     int(arrived),
            'total_missed':      float(missed),
            'missed_by_type':    missed_by_type,
            'missed_by_mu':      missed_by_mu,
            'avg_power_kw':      float(np.mean(power) / 1000),
            'P_target_kw':       float(self.params.P_target / 1000),
        }

        script_dir = os.path.dirname(os.path.abspath(__file__))
        fname = os.path.join(script_dir, f'metrics_lambda_{lam:.1f}.json')
        with open(fname, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved -> {fname}")
        return fname


# ============================================================================
# SECTION 3 PLOTTER
# reads all metrics_lambda_*.json files in a folder and produces the
# 5-panel figure that report1 §3 asks for.
# ============================================================================

def compute_lambda_star(params):
    """Back-of-envelope lam* from report1 §1."""
    E_tau    = np.mean([jt['tau_r_mean'] for jt in params.job_types.values()])
    P_idle   = params.get_power(0)
    P_max    = params.get_power(params.K - 1)
    N_active = (params.P_target - params.N_gpus * P_idle) / (P_max - P_idle)
    lam_star = N_active / E_tau
    return lam_star, E_tau, N_active


def plot_section3(folder=None):
    """
    Load all metrics_lambda_*.json files from `folder` (defaults to script dir)
    and produce the 5-panel Section 3 figure from report1.

    Panels:
      1. Miss rate vs lam  +  analytical prediction max(0, 1 - lam*/lam)
      2. RMS tracking error vs lam
      3. Mean queue length vs lam
      4. Mean busy GPUs vs lam, split by memory class
      5. Miss rate decomposed by job type vs lam
    """
    import json, glob, os
    import numpy as np
    import matplotlib.pyplot as plt

    if folder is None:
        folder = os.path.dirname(os.path.abspath(__file__))

    # Load all metric files
    pattern = os.path.join(folder, 'metrics_lambda_*.json')
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"No metrics files found in {folder}")
        print("Run the simulation at several lam values first - each run saves a JSON file.")
        return

    data = []
    for fp in files:
        with open(fp) as f:
            data.append(json.load(f))
    data.sort(key=lambda x: x['lambda'])

    lambdas       = [d['lambda']        for d in data]
    miss_rates    = [d['miss_rate']     for d in data]
    rms_errors    = [d['rms_error_kw']  for d in data]
    mean_queues   = [d['mean_queue']    for d in data]

    # Analytical lam* - use reference params
    ref_params = GPUSystemParameters()
    lam_star, E_tau, N_active = compute_lambda_star(ref_params)
    miss_pred = [max(0.0, 1.0 - lam_star / lam) for lam in lambdas]

    # Per-class busy GPUs
    class_names = [gc['name'] for gc in ref_params.gpu_classes]
    class_colors = ['tab:blue', 'tab:orange', 'tab:red']
    busy_by_class = {name: [] for name in class_names}
    for d in data:
        for name in class_names:
            busy_by_class[name].append(d['mean_busy_per_class'].get(name, 0.0))

    # Per-type miss rates
    type_names   = ['prefill', 'decode', 'fine_tuning', 'training']
    type_colors  = ['tab:green', 'tab:blue', 'tab:orange', 'tab:red']
    miss_by_type = {jt: [] for jt in type_names}
    for d in data:
        arr = max(d['total_arrived'], 1)
        for jt in type_names:
            miss_by_type[jt].append(d['missed_by_type'].get(jt, 0.0) / arr)

    # ── Build figure ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        f"Section 3 - lambda-Sweep Results  |  Flat Target {ref_params.P_target/1000:.0f} kW  |  "
        "F1 fix: prefill tau_r=2, tau_s=1\n"
        f"lam* ≈ {lam_star:.1f} jobs/slot  (N_active={N_active:.0f}, E[tau]={E_tau:.2f})",
        fontsize=13, fontweight='bold'
    )
    kw = dict(marker='o', linewidth=2, markersize=7)

    # Panel 1 - Miss rate
    ax = axes[0, 0]
    ax.plot(lambdas, miss_rates, color='crimson', label='Empirical miss rate', **kw)
    ax.plot(lambdas, miss_pred,  color='black', linestyle='--', linewidth=2,
            label=f'Analytical: max(0, 1-lam*/lam)\nlam*={lam_star:.1f}')
    ax.axvline(lam_star, color='navy', linestyle=':', linewidth=1.5,
               label=f'lam* = {lam_star:.1f}')
    ax.set_xlabel('Arrival rate lambda (jobs/slot)', fontsize=11)
    ax.set_ylabel('Miss rate', fontsize=11)
    ax.set_title('1. Miss Rate vs lam', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 2 - RMS tracking error
    ax = axes[0, 1]
    ax.plot(lambdas, rms_errors, color='steelblue', label='RMS tracking error', **kw)
    ax.axvline(lam_star, color='navy', linestyle=':', linewidth=1.5,
               label=f'lam* = {lam_star:.1f}')
    ax.set_xlabel('Arrival rate lambda (jobs/slot)', fontsize=11)
    ax.set_ylabel('RMS error (kW)', fontsize=11)
    ax.set_title('2. RMS Tracking Error vs lam', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 3 - Mean queue length
    ax = axes[1, 0]
    ax.plot(lambdas, mean_queues, color='darkorange', label='Mean waiting queue', **kw)
    ax.axvline(lam_star, color='navy', linestyle=':', linewidth=1.5,
               label=f'lam* = {lam_star:.1f}')
    ax.set_xlabel('Arrival rate lambda (jobs/slot)', fontsize=11)
    ax.set_ylabel('Mean jobs waiting', fontsize=11)
    ax.set_title('3. Mean Queue Length vs lam', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 4 - Busy GPUs by memory class
    ax = axes[1, 1]
    for name, col in zip(class_names, class_colors):
        ax.plot(lambdas, busy_by_class[name], color=col,
                label=f'{name} busy', **kw)
    ax.axhline(N_active, color='black', linestyle='--', linewidth=1.5,
               label=f'N_active* = {N_active:.0f}')
    ax.axvline(lam_star, color='navy', linestyle=':', linewidth=1.5,
               label=f'lam* = {lam_star:.1f}')
    ax.set_xlabel('Arrival rate lambda (jobs/slot)', fontsize=11)
    ax.set_ylabel('Mean busy GPUs', fontsize=11)
    ax.set_title('4. GPU Utilisation vs lam (by memory class)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 5 - Miss rate by job type
    ax = axes[2, 0]
    for jt, col in zip(type_names, type_colors):
        ax.plot(lambdas, miss_by_type[jt], color=col, label=jt, **kw)
    ax.axvline(lam_star, color='navy', linestyle=':', linewidth=1.5,
               label=f'lam* = {lam_star:.1f}')
    ax.set_xlabel('Arrival rate lambda (jobs/slot)', fontsize=11)
    ax.set_ylabel('Miss rate by job type', fontsize=11)
    ax.set_title('5. Miss Rate by Job Type vs lam', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 6 - Summary table
    ax = axes[2, 1]
    ax.axis('off')
    col_labels = ['lam', 'Miss %', 'RMS kW', 'Avg Queue', 'Avg Power kW']
    rows = []
    for d in data:
        rows.append([
            f"{d['lambda']:.0f}",
            f"{d['miss_rate']*100:.1f}%",
            f"{d['rms_error_kw']:.1f}",
            f"{d['mean_queue']:.0f}",
            f"{d['avg_power_kw']:.1f}",
        ])
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    ax.set_title('Summary Table', fontsize=12, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(folder, 'section3_sweep.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Section 3 figure saved -> {out}")
    print(f"  Data points: lam = {lambdas}")
    print(f"  Analytical lam* = {lam_star:.2f}")
    return out


# ============================================================================
# STEP 8: RUN IT!
# ============================================================================

if __name__ == "__main__":
    params = GPUSystemParameters()
    sim    = GPUSimulator(params)
    sim.run()

    # Time-series plot for this run
    sim.plot_results()

    # Save metrics JSON for Section 3 sweep figure
    sim.save_run_metrics()

    # ── To build the Section 3 figure after running several lam values: ────
    # Uncomment the line below once you have metrics files for multiple lams.
    # plot_section3()  # reads all metrics_lambda_*.json in the script folder
    
    # Print summary statistics
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
    print(f"Target:         {params.P_target/1000:.2f} kW")
    print(f"Deviation:      {abs(avg_power - params.P_target)/1000:.2f} kW")
    
    avg_util = np.mean(sim.history['n_busy']) / params.N_gpus * 100
    print(f"\nAverage GPU utilization: {avg_util:.1f}%")
    print(f"Average idle GPUs: {np.mean(sim.history['n_idle']):.0f}")
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


# ============================================================================
# STEP 1: DEFINE THE SYSTEM PARAMETERS
# ============================================================================

class GPUSystemParameters:

    
    def __init__(self):
       
        self.N_gpus = 1000  # Total number of GPUs, for the time being is not binding
        
        # Freq Levels
        
        self.frequencies = {
            0: {'rate': 0.00, 'power': 50},   # Idle (leakage power only)
            1: {'rate': 0.25, 'power': 150},  # 25% speed
            2: {'rate': 0.50, 'power': 200},  # 50% speed  
            3: {'rate': 0.75, 'power': 250},  # 75% speed
            4: {'rate': 1.00, 'power': 300},  # 100% speed (max)
        }
        self.K = len(self.frequencies) 
        

        self.jobtypes = {
            "inference": {'rate': 0.00, 'power': 50},   # Idle (leakage power only)
            "training": {'rate': 0.25, 'power': 150},  # 25% speed
            "fine-tuning": {'rate': 0.50, 'power': 200},  # 50% speed  
        }
       
        self.lambda_arrival = 100.0  
        
        self.tau_r_mean = 10.0  
        self.tau_r_p = 1.0 / self.tau_r_mean  
        
        
        self.tau_s_mean = 5.0 
        self.tau_s_p = 1.0 / self.tau_s_mean  
        
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
        self.lambda_smooth = 50
        
    def get_rate(self, k):
        return self.frequencies[k]['rate']
    
    def get_power(self, k):
        return self.frequencies[k]['power']


# ============================================================================
# STEP 2: DEFINE THE STATE SPACE
# ============================================================================

class PopulationState:
  
    
    def __init__(self, params):
        self.params = params
        
    
        self.n = defaultdict(float)
        
        # Initially, all GPUs are idle
        self.n_idle = float(params.N_gpus)

        
        self.waiting = defaultdict(float)
        
        #returns count of active states
    def get_count(self, tau, s):
        return self.n.get((tau, s), 0.0)
    
    def set_count(self, tau, s, count):
        if count > 0:
            self.n[(tau, s)] = count
        else:
            # Remove zero entries to keep dict small
            self.n.pop((tau, s), None)
    
    def get_all_states(self):
        return list(self.n.keys())
    
    def total_busy_gpus(self):
        return sum(self.n.values())
    
    def total_gpus(self):
        return self.n_idle + self.total_busy_gpus()
    
    def get_waiting_count(self,tau,s):
        return self.waiting.get((tau,s),0.0)
    


# ============================================================================
# STEP 3: JOB ARRIVALS
# ============================================================================

class JobArrivalProcess:
    
    def __init__(self, params):
        self.params = params
        self.rng = np.random.default_rng(seed=42)  
        
    def generate_arrivals(self):
       
       
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
        n_ineq = 2 + 2 * n_vars_u
        A_ub = np.zeros((n_ineq, n_vars))
        b_ub = np.zeros(n_ineq)

        # Row 0: positive power deviation
        A_ub[0, :n_vars_u] =  p_row
        A_ub[0, idx_e]     = -1.0
        b_ub[0]            =  P_busy_target

        # Row 1: negative power deviation
        A_ub[1, :n_vars_u] = -p_row
        A_ub[1, idx_e]     = -1.0
        b_ub[1]            = -P_busy_target

        # Rows 2 ... 2+2*n_vars_u: smoothing constraints
        for col in range(n_vars_u):
            i_state = col // n_K
            j_freq  = col  % n_K
            tau, s  = free_states[i_state]
            k       = K_busy[j_freq]
            u_p     = self.u_prev.get((tau, s, k), 0.5)  # default 0.5 at start

            row_pos = 2 + 2 * col       #  u - u_prev - d <= 0
            row_neg = 2 + 2 * col + 1   # -u + u_prev - d <= 0

            A_ub[row_pos, col]        =  1.0   # u[col]
            A_ub[row_pos, idx_d+col]  = -1.0   # -d[col]
            b_ub[row_pos]             =  u_p   # <= u_prev

            A_ub[row_neg, col]        = -1.0   # -u[col]
            A_ub[row_neg, idx_d+col]  = -1.0   # -d[col]
            b_ub[row_neg]             = -u_p   # <= -u_prev
 
        #A_eq one row per free state
        #only assin one to the columns that such state for each row
        A_eq = np.zeros((n_free, n_vars))

        #each state must add up to 1 over all frequencies 
        #Ax = b, meaning all bs have to be 1
        b_eq = np.ones(n_free)
 
        for i in range(n_free):
            for j in range(n_K):
                col = i * n_K + j
                A_eq[i, col] = 1.0
        # e column stays 0 in A_eq ✓
 
        # ----------------------------------------------------------------
        # VARIABLE BOUNDS
        #   u variables: [0, 1]   (frequency fractions)
        #   e variable:  [0, ∞)   (absolute power deviation)
        #   d variables: [0, ∞)   (absolute u change, non-negative by construction)
        # ----------------------------------------------------------------
        bounds = []
        for col in range(n_vars_u):
            bounds.append((0.0, 1.0))    # u[i,k]
        bounds.append((0.0, None))        # e
        for col in range(n_vars_u):
            bounds.append((0.0, None))    # d[i,k]
 
        # ----------------------------------------------------------------
        # 10. SOLVE THE LP
        #    method='highs' is the default in scipy >= 1.7 and is fast,
        #    numerically stable, and handles degeneracy well.
        # ----------------------------------------------------------------
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
        
    def evolve(self, state, control, arrivals, current_power):
 
        from collections import defaultdict
    
        new_state = PopulationState(self.params)
        new_state.waiting = defaultdict(float, state.waiting)
    
        # new arrivals go to queue
        for (tau, s), count in arrivals.items():
            new_state.waiting[(tau, s)] += count
    
        
        # As we evolve active jobs, some will finish and free GPUs
        # This will DECREASE power
    
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
                r_k = self.params.get_rate(k)
                tau_next = tau - r_k
            
                # Will this job finish?
                if tau_next <= 0:
                    
                    P_k = self.params.get_power(k)
                    P_idle = self.params.get_power(0)
                    # Power lost per GPU
                    power_loss_per_gpu = P_k - P_idle
                    # Total power loss from these finishing jobs
                    predicted_power_loss += n_at_k * power_loss_per_gpu
    
       #add it to overall headroom
        power_gap = self.params.P_target - current_power
        adjusted_headroom = power_gap + predicted_power_loss
    
        # How many new jobs can we admit to use this headroom?
        # Estimate power per new job based on current control policy
        avg_power_per_new_job = self._estimate_avg_power_for_new_jobs(state, control)
    
        if avg_power_per_new_job > 0:
            max_new_gpus = max(0, adjusted_headroom / avg_power_per_new_job)
        else:
            max_new_gpus = 0
    
        # Optional: Cap at some reasonable maximum to avoid instability
        max_new_gpus = min(max_new_gpus, 500)  # Don't admit more than 500 at once
    
        # === ADMIT JOBS FROM WAITING QUEUE ===
        waiting_states = sorted(new_state.waiting.keys(), 
                           key=lambda ts: ts[0] + ts[1])  # Urgency
    
        gpus_assigned = 0
        new_state.n_idle = state.n_idle
    
        for (tau, s) in waiting_states:
            if gpus_assigned >= max_new_gpus:
                break
        
            waiting_count = new_state.waiting[(tau, s)]
            if waiting_count <= 0:
                continue
            #Can assign holds the number of GPUs to be assigned (added onto active jobs and removed from waiting)
            can_assign = min(waiting_count, max_new_gpus - gpus_assigned)
        
            # Move from waiting to active

            #remove jobs to be assigned from waiting queue
            new_state.waiting[(tau, s)] -= can_assign
            if new_state.waiting[(tau, s)] <= 0:
                new_state.waiting.pop((tau, s), None)
        
            
            new_state.set_count(tau, s, 
                          new_state.get_count(tau, s) + can_assign)
            new_state.n_idle -= can_assign
            gpus_assigned += can_assign
    
        #create new dictionary to hold updated slack values
        waiting_next = defaultdict(float)
    
        for (tau, s), count in new_state.waiting.items():
            if count <= 0:
                continue
        
        s_next = s - 1
        
        if s_next < 0:
            print(f"  Warning: {count:.0f} jobs missed deadline in queue!")
        else:
            waiting_next[(tau, s_next)] += count
    
        new_state.waiting = waiting_next
    
    # update state of active jobs
        for (tau, s) in state.get_all_states():
            n_current = state.get_count(tau, s)
        
            if n_current == 0:
                continue
        
            for k in range(self.params.K):
                u_k = control.get((tau, s, k), 0.0)
            
                if u_k == 0:
                    continue
            
                n_at_k = u_k * n_current
                r_k = self.params.get_rate(k)
            
                tau_next = tau - r_k
                s_next = s - (1 - r_k)
            
                if tau_next <= 0:
                    # Job finished!
                    new_state.n_idle += n_at_k
                else:
                    # Job continues
                    tau_next = round(tau_next * 4) / 4
                    s_next = round(s_next * 4) / 4
                    s_next = max(0, s_next)
                
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
        
        # History (for plotting)
        self.history = {
            'time': [],
            'power': [],
            'n_idle': [],
            'n_busy': [],
            'queue_size': [],
            'waiting_queue': [],
        }
        
    def run(self):
        """Run the simulation for T time steps"""
        
        print("Starting simulation...")
        print(f"Target power: {self.params.P_target/1000:.1f} kW")
        print(f"Initial idle GPUs: {self.state.n_idle:.0f}")
        print()

        #initializing power consumption
        #start assuming all GPUS are idle
        power_prev = self.params.get_power(0) * self.params.N_gpus
        
        for t in range(self.params.T_sim):
            #arrivals
            arrivals_t = self.arrivals.generate_arrivals()
            
            #choose new control policy based on new arrivals
            control_t = self.policy.compute_control_power(self.state, power_prev)

            #Evolve state including new arrivals and calculated frequencies 
            self.state = self.dynamics.evolve(self.state, control_t,arrivals_t,power_prev)

            #new power calc
            power_t = self.power_calc.compute_power(self.state, control_t)
            power_prev = power_t  # Save for next iteration
            
           
           
            
            # 4. Record history
            self.history['time'].append(t)
            self.history['power'].append(power_t)
            self.history['n_idle'].append(self.state.n_idle)
            self.history['n_busy'].append(self.state.total_busy_gpus())
            self.history['queue_size'].append(len(self.state.get_all_states()))
            waiting_total = sum(self.state.waiting.values())
            self.history['waiting_queue'].append(waiting_total)
            
            # 5. Print progress every 10 steps
            if t % 10 == 0:
                print(f"t={t:3d}: Power={power_t/1000:6.1f} kW, "
                      f"Idle={self.state.n_idle:5.0f}, "
                      f"Busy={self.state.total_busy_gpus():5.0f}, "
                      f"States={len(self.state.get_all_states())}")
        
        print()
        print("Simulation complete!")
        
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
        plt.savefig(r'C:/Users/lauso/OneDrive/Desktop/Cornell/AI flexible demand response/New model images/simulation_results_LP.png', dpi=150, bbox_inches='tight')
        print("Plot saved to simulation_results_LP.png")
        
        return fig


# ============================================================================
# STEP 8: RUN IT!
# ============================================================================

if __name__ == "__main__":
    # Create parameters
    params = GPUSystemParameters()
    
    # Create and run simulator
    sim = GPUSimulator(params)
    sim.run()
    
    # Plot results
    sim.plot_results()
    
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

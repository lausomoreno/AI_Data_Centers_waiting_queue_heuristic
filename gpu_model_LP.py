

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


# ============================================================================
# STEP 1: DEFINE THE SYSTEM PARAMETERS
# ============================================================================

class GPUSystemParameters:

    
    def __init__(self):
       
        self.N_gpus = 1000  # Total number of GPUs
        
        # Freq Levels
        
        self.frequencies = {
            0: {'rate': 0.00, 'power': 50},   # Idle (leakage power only)
            1: {'rate': 0.25, 'power': 150},  # 25% speed
            2: {'rate': 0.50, 'power': 200},  # 50% speed  
            3: {'rate': 0.75, 'power': 250},  # 75% speed
            4: {'rate': 1.00, 'power': 300},  # 100% speed (max)
        }
        self.K = len(self.frequencies) 
        
        # average poisson rate
        self.lambda_arrival = 80.0  #
        
       
       #Residual time and slack
        self.tau_r_mean = 10.0  
        self.tau_r_p = 1.0 / self.tau_r_mean  
        
        # Slack: geometric distribution  
        self.tau_s_mean = 5.0 
        self.tau_s_p = 1.0 / self.tau_s_mean  
        
        # --- Time Grid ---
        # For simplicity, we quantize time in slots of 1 second
        self.dt = 1.0  # Time slot duration (seconds)
        self.T_sim = 1000  # Number of time slots to simulate
        
        # Power Target 
        self.P_target = 150_000  #kW 
        
        # --- State Space Discretization ---
        # To keep things finite, we cap tau and s at some maximum
        self.tau_max = 50  # Max residual time we track
        self.s_max = 20    # Max slack we track
        
    def get_rate(self, k):
        """Get processing rate for frequency level k"""
        return self.frequencies[k]['rate']
    
    def get_power(self, k):
        """Get power consumption for frequency level k"""
        return self.frequencies[k]['power']


# ============================================================================
# STEP 2: DEFINE THE STATE SPACE
# ============================================================================

class PopulationState:
  
    
    def __init__(self, params):
        self.params = params
        
        # Population counts: n[tau, s] = number of GPUs in that state
        # We use a defaultdict so missing states automatically = 0
        self.n = defaultdict(float)
        
        # Initially, all GPUs are idle
        self.n_idle = float(params.N_gpus)


        #adding my waiting queue 
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
        self.rng = np.random.default_rng(seed=42)  # For reproducibility
        
    def generate_arrivals(self):
        """
        Generate arrivals for one time slot.
        
        Returns:
            arrivals: dict mapping (tau, s) -> number of arrivals
        """
        # Number of arrivals this slot (Poisson)
        n_arrivals = self.rng.poisson(self.params.lambda_arrival)
        
        arrivals = defaultdict(int)
        
        for _ in range(n_arrivals):
            # Draw tau_r from geometric distribution
            # numpy's geometric is 1-indexed, so subtract 1 for 0-indexing
            tau_r = self.rng.geometric(self.params.tau_r_p)
            tau_r = min(tau_r, self.params.tau_max)  # Cap at max
            
            # Draw tau_s from geometric distribution
            tau_s = self.rng.geometric(self.params.tau_s_p)
            tau_s = min(tau_s, self.params.s_max)  # Cap at max
            
            # Record arrival
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
 
        # Adjusted target: what power should the FREE GPUs produce?
        #readjusted power overhead, excluding idle and fixed 
        #to be used in b_ub
        P_busy_target = self.params.P_target - P_idle_total - P_fixed

        ##---------- LP variables----------------------------

        K_busy = [1, 2, 3, 4]  
        n_free  = len(free_states)
        n_K     = len(K_busy)
        n_vars_u = n_free * n_K
        idx_e    = n_vars_u      # Index of the slack variable e
        n_vars   = n_vars_u + 1  # Total LP variables
 
        # Map (tau, s, k) -> column index in x
        # State i gets columns [i*n_K, i*n_K + 1, i*n_K + 2, i*n_K + 3]
        # for k = 1, 2, 3, 4 respectively
        var_index = {}
        for i in range(n_free):
            tau, s = free_states[i]
            for j in range(n_K):
                k = K_busy[j]
                var_index[(tau, s, k)] = i * n_K + j

        c = np.zeros(n_vars)
        c[idx_e] = 1.0


        # ----------------------------------------------------------------
        # 6. POWER COEFFICIENT VECTOR
        #    p_row[col] = n[tau, s] * P_k  for column col = var_index[(tau,s,k)]
        #    This tells the LP how much power one unit of u[tau,s,k] contributes.
        # ----------------------------------------------------------------
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
        # 7. INEQUALITY CONSTRAINTS (A_ub @ x <= b_ub)
        #    We need:
        #        P_busy_actual - P_busy_target <= e
        #       -P_busy_actual + P_busy_target <= e
        #
        #    Row 1:  p_row @ u_vars  - e  <=  P_busy_target
        #    Row 2: -p_row @ u_vars  - e  <= -P_busy_target
        # ----------------------------------------------------------------
        A_ub = np.zeros((2, n_vars))
        b_ub = np.zeros(2)
 
        # Row 0: positive power deviation
        for col in range(n_vars_u):
            A_ub[0, col] =  p_row[col]
        A_ub[0, idx_e] = -1.0
        b_ub[0] = P_busy_target
 
        # Row 1: negative power deviation
        for col in range(n_vars_u):
            A_ub[1, col] = -p_row[col]
        A_ub[1, idx_e] = -1.0
        b_ub[1] = -P_busy_target
 
        # ----------------------------------------------------------------
        # 8. EQUALITY CONSTRAINTS (A_eq @ x == b_eq)
        #    Simplex constraint: for each free state i,
        #        u[tau_i, s_i, 1] + u[tau_i, s_i, 2] + u[tau_i, s_i, 3] + u[tau_i, s_i, 4] = 1
        #
        #    One equality row per free state.
        #    The slack variable e is NOT involved → coefficient stays 0.
        # ----------------------------------------------------------------
        A_eq = np.zeros((n_free, n_vars))
        b_eq = np.ones(n_free)
 
        for i in range(n_free):
            for j in range(n_K):
                col = i * n_K + j
                A_eq[i, col] = 1.0
        # e column stays 0 in A_eq ✓
 
        # ----------------------------------------------------------------
        # 9. VARIABLE BOUNDS
        #    u variables: [0, 1]  (fractions, by definition)
        #    e variable:  [0, ∞)  (absolute deviation, non-negative)
        # ----------------------------------------------------------------
        bounds = []
        for col in range(n_vars_u):
            bounds.append((0.0, 1.0))
        bounds.append((0.0, None))  # e: non-negative, no upper bound
 
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
 
        # ----------------------------------------------------------------
        # 11. EXTRACT SOLUTION
        #     If the LP succeeded, read off the u values.
        #     If it failed (infeasible or numerical issues), fall back to
        #     a uniform split. Failure should be rare because our LP is
        #     always feasible: any u satisfying the simplex constraints
        #     is valid, and e can always absorb any power error.
        # ----------------------------------------------------------------
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
 
        return u
 
 
# ============================================================================
# DIAGNOSTIC HELPER (optional, call manually for debugging)
# ============================================================================
 
def lp_diagnostics(state, params, u_lp, u_heuristic=None):
    """
    Print a comparison of the LP solution vs the heuristic.
    Call this during development to verify the LP is working.
 
    Example usage in your simulation loop:
        if t % 50 == 0:
            heuristic_u = old_policy.compute_control_power(state, power_prev)
            lp_diagnostics(state, params, control_t, heuristic_u)
    """
    from collections import defaultdict
 
    P_idle = params.get_power(0)
    P_idle_total = P_idle * state.n_idle
 
    def compute_busy_power(u):
        p = 0.0
        for (tau, s) in state.get_all_states():
            n = state.get_count(tau, s)
            for k in range(params.K):
                p += u.get((tau, s, k), 0.0) * n * params.get_power(k)
        return p
 
    P_lp = P_idle_total + compute_busy_power(u_lp)
    print(f"\n[LP Diagnostics]")
    print(f"  Target power  : {params.P_target/1000:.1f} kW")
    print(f"  LP power      : {P_lp/1000:.1f} kW  (error: {abs(P_lp - params.P_target)/1000:.2f} kW)")
 
    if u_heuristic is not None:
        P_h = P_idle_total + compute_busy_power(u_heuristic)
        print(f"  Heuristic pwr : {P_h/1000:.1f} kW  (error: {abs(P_h - params.P_target)/1000:.2f} kW)")
 
    # Show fraction of states using mixed (non-binary) u
    mixed = sum(
        1 for (tau, s) in state.get_all_states()
        for k in range(params.K)
        if 0 < u_lp.get((tau, s, k), 0.0) < 1.0
    )
    print(f"  Mixed (fractional) u entries: {mixed}  "
          f"(heuristic would have 0)")
    print()
# ============================================================================
# STEP 5: STATE TRANSITIONS (POPULATION DYNAMICS)
# ============================================================================

class PopulationDynamics:
    """
    Evolves the population from time t to t+1.
    
    This implements Equation 10 from the technical note.
    NOW WITH WAITING QUEUE AND ADMISSION CONTROL!
    """
    
    def __init__(self, params):
        self.params = params
        
    def evolve(self, state, control, arrivals, current_power):
 
        from collections import defaultdict
    
        new_state = PopulationState(self.params)
        new_state.waiting = defaultdict(float, state.waiting)
    
        # === NEW ARRIVALS GO TO WAITING QUEUE ===
        for (tau, s), count in arrivals.items():
            new_state.waiting[(tau, s)] += count
    
        # === PREDICT POWER CHANGE FROM JOBS FINISHING ===
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
                    # Yes! This GPU will go idle
                    # Current power from this GPU
                    P_k = self.params.get_power(k)
                    # After finishing, GPU goes idle
                    P_idle = self.params.get_power(0)
                    # Power lost per GPU
                    power_loss_per_gpu = P_k - P_idle
                    # Total power loss from these finishing jobs
                    predicted_power_loss += n_at_k * power_loss_per_gpu
    
        # === COMPUTE AVAILABLE POWER BUDGET ===
        # Current power situation
        power_gap = self.params.P_target - current_power
    
        # Account for jobs that will finish (power will decrease)
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
            #this can assign is by state, x number of jobs in given state are gonna end
            can_assign = min(waiting_count, max_new_gpus - gpus_assigned)
        
            # Move from waiting to active

            #This is job many jobs of this space I can assign 
            new_state.waiting[(tau, s)] -= can_assign
            if new_state.waiting[(tau, s)] <= 0:
                new_state.waiting.pop((tau, s), None)
        
            new_state.set_count(tau, s, 
                          new_state.get_count(tau, s) + can_assign)
            new_state.n_idle -= can_assign
            gpus_assigned += can_assign
    
            # === WAITING JOBS: Slack decreases ===
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
    
    # === ACTIVE JOBS: State transitions ===
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
            # 1. Generate arrivals for this time slot
            arrivals_t = self.arrivals.generate_arrivals()
            
            # 2. Compute current power
            #control tells me based on current state
            #what is my initial state????
            control_t = self.policy.compute_control_power(self.state, power_prev)

            #Evolve state
            self.state = self.dynamics.evolve(self.state, control_t,arrivals_t,power_prev)

            #new power
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

"""
SIMPLE GPU POPULATION MODEL - Teaching Implementation
======================================================

What this does:
- Models a datacenter with 1000 GPUs (all same memory class)
- Jobs arrive following a Poisson process
- Jobs have geometric slack distribution
- GPUs can run at 5 frequencies: 0%, 25%, 50%, 75%, 100%
- Goal: Keep power flat at 300 kW

Author: Teaching example for Laura
Date: March 2026
"""

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


# ============================================================================
# STEP 1: DEFINE THE SYSTEM PARAMETERS
# ============================================================================

class GPUSystemParameters:
    """
    All the constants that define our datacenter.
    Think of this as the "configuration file" for the simulation.
    """
    
    def __init__(self):
        # --- GPU Fleet ---
        self.N_gpus = 1000  # Total number of GPUs
        
        # --- Frequency Levels ---
        # We have 5 frequency options: idle, 25%, 50%, 75%, 100%
        self.frequencies = {
            0: {'rate': 0.00, 'power': 50},   # Idle (leakage power only)
            1: {'rate': 0.25, 'power': 150},  # 25% speed
            2: {'rate': 0.50, 'power': 200},  # 50% speed  
            3: {'rate': 0.75, 'power': 250},  # 75% speed
            4: {'rate': 1.00, 'power': 300},  # 100% speed (max)
        }
        self.K = len(self.frequencies)  # Number of frequency levels
        
        # --- Job Arrival Process ---
        self.lambda_arrival = 80.0  # Average jobs per time slot (Poisson rate)
        
        # --- Job Characteristics ---
        # Residual time: geometric distribution
        self.tau_r_mean = 10.0  # Average work time (slots at 100% speed)
        self.tau_r_p = 1.0 / self.tau_r_mean  # Geometric parameter
        
        # Slack: geometric distribution  
        self.tau_s_mean = 5.0  # Average slack time (slots)
        self.tau_s_p = 1.0 / self.tau_s_mean  # Geometric parameter
        
        # --- Time Grid ---
        # For simplicity, we quantize time in slots of 1 second
        self.dt = 1.0  # Time slot duration (seconds)
        self.T_sim = 100  # Number of time slots to simulate
        
        # --- Power Target ---
        self.P_target = 150_000  # Target power (300 kW in Watts)
        
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
    """
    Tracks the population of GPUs in each state (tau, s).
    
    Key data structure:
        n[tau, s] = number of GPUs in state (tau, s)
        n_idle = number of idle GPUs
    """
    
    def __init__(self, params):
        self.params = params
        
        # Population counts: n[tau, s] = number of GPUs in that state
        # We use a defaultdict so missing states automatically = 0
        self.n = defaultdict(float)
        
        # Initially, all GPUs are idle
        self.n_idle = float(params.N_gpus)


        #adding my waiting queue 
        self.waiting = defaultdict(float)
        
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
    """
    Generates new jobs according to a Poisson process.
    Each job gets random (tau_r, tau_s) from geometric distributions.
    """
    
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

class ControlPolicy:
    """
    Decides what fraction of GPUs in each state should run at each frequency.
    
    This is the "brain" of the system - the scheduler.
    """
    
    def __init__(self, params):
        self.params = params
        
    def compute_control(self, state, current_power):
        """
        Compute control u[tau, s, k] for all states.
        
        Args:
            state: PopulationState object
            current_power: current total power consumption (Watts)
            
        Returns:
            u: dict mapping (tau, s, k) -> fraction at frequency k
        """
        u = {}
        
        # --- SIMPLE POLICY: Throttle based on slack ---
        # 
        # Rule:
        #   - If s == 0 (no slack): MUST run at 100% (k=4)
        #   - If s > 0 (has slack): Choose frequency to flatten power
        #
        # For now, let's use a SUPER SIMPLE heuristic:
        #   - High slack (s >= 10): run at 25% (k=1)
        #   - Medium slack (5 <= s < 10): run at 50% (k=2)
        #   - Low slack (1 <= s < 5): run at 75% (k=3)
        #   - No slack (s == 0): run at 100% (k=4)
        
        for (tau, s) in state.get_all_states():
            if tau <= 0:
                # Job is done, shouldn't be here
                continue
                
            if s == 0:
                # No slack - MUST run at full speed
                u[(tau, s, 4)] = 1.0
                for k in range(4):
                    u[(tau, s, k)] = 0.0
                    
            elif s >= 10:
                # Lots of slack - throttle heavily
                u[(tau, s, 1)] = 1.0  # 100% at 25% speed
                for k in [0, 2, 3, 4]:
                    u[(tau, s, k)] = 0.0
                    
            elif s >= 5:
                # Medium slack - moderate throttling
                u[(tau, s, 2)] = 1.0  # 100% at 50% speed
                for k in [0, 1, 3, 4]:
                    u[(tau, s, k)] = 0.0
                    
            else:  # 1 <= s < 5
                # Low slack - light throttling
                u[(tau, s, 3)] = 1.0  # 100% at 75% speed
                for k in [0, 1, 2, 4]:
                    u[(tau, s, k)] = 0.0
        
        return u


    def compute_control_power(self, state, current_power):
        u = {}
        
        # Power-aware throttling
        power_gap = (current_power - self.params.P_target) / self.params.P_target
        
        for (tau, s) in state.get_all_states():
            if tau <= 0:
                continue
                
            if s == 0:
                u[(tau, s, 4)] = 1.0
                for k in range(4):
                    u[(tau, s, k)] = 0.0
            else:
                # Choose frequency based on power gap
                if power_gap > 0.2:  # Too much power
                    freq = 1  # Throttle to 25%
                elif power_gap > 0:
                    freq = 2  # Throttle to 50%
                elif power_gap > -0.2:
                    freq = 3  # 75%
                else:  # Too little power
                    freq = 4  # 100%
                
                u[(tau, s, freq)] = 1.0
                for k in range(5):
                    if k != freq:
                        u[(tau, s, k)] = 0.0
        
        return u
        


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
        """
        Evolve population one time step with waiting queue.
        
        Args:
            state: current PopulationState
            control: dict u[tau, s, k] from ControlPolicy
            arrivals: dict (tau, s) -> count from JobArrivalProcess
            current_power: current total power (Watts)
            
        Returns:
            new_state: PopulationState at t+1
        """
        new_state = PopulationState(self.params)
        
        # Copy waiting queue from previous state
        new_state.waiting = defaultdict(float, state.waiting)
        
        # === NEW ARRIVALS GO TO WAITING QUEUE ===
        for (tau, s), count in arrivals.items():
            new_state.waiting[(tau, s)] += count
        
        # === ADMISSION CONTROL: Assign from queue to GPUs ===
        # Compute power headroom
        power_headroom = self.params.P_target - current_power
        
        # How many GPUs can we assign?
        avg_power_per_job = 200  # Estimate (50% frequency)
        max_new_gpus = max(0, power_headroom / avg_power_per_job)
        max_new_gpus = min(max_new_gpus, state.n_idle)
        
        # Sort waiting jobs by urgency (lowest urgency value = most urgent)
        # urgency = tau + s (total time to deadline)
        waiting_states = sorted(new_state.waiting.keys(), 
                               key=lambda ts: ts[0] + ts[1])  # Ascending order
        
        gpus_assigned = 0
        new_state.n_idle = state.n_idle
        
        for (tau, s) in waiting_states:
            if gpus_assigned >= max_new_gpus:
                break
            
            waiting_count = new_state.waiting[(tau, s)]
            if waiting_count <= 0:
                continue
            
            can_assign = min(waiting_count, max_new_gpus - gpus_assigned)
            
            # Move from waiting to active
            new_state.waiting[(tau, s)] -= can_assign
            if new_state.waiting[(tau, s)] <= 0:
                new_state.waiting.pop((tau, s), None)
            
            new_state.set_count(tau, s, 
                              new_state.get_count(tau, s) + can_assign)
            new_state.n_idle -= can_assign
            gpus_assigned += can_assign
        
        # === WAITING JOBS: Slack decreases (no work progress) ===
        waiting_next = defaultdict(float)
        
        for (tau, s), count in new_state.waiting.items():
            if count <= 0:
                continue
            
            # While waiting: tau unchanged, s decreases by 1
            s_next = s - 1
            
            if s_next < 0:
                # Deadline missed!
                print(f"  Warning: {count:.0f} jobs with state ({tau},{s}) missed deadline!")
            else:
                # Move to next waiting state
                waiting_next[(tau, s_next)] += count
        
        new_state.waiting = waiting_next
        
        # === ACTIVE JOBS: Normal state transitions ===
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



# ============================================================================
# STEP 6: POWER CALCULATION
# ============================================================================

class PowerCalculator:
    """
    Computes total power consumption given state and control.
    
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
        self.policy = ControlPolicy(params)
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
        }
        
    def run(self):
        """Run the simulation for T time steps"""
        
        print("Starting simulation...")
        print(f"Target power: {self.params.P_target/1000:.1f} kW")
        print(f"Initial idle GPUs: {self.state.n_idle:.0f}")
        print()

        #initializing power consumption
        power_prev = self.params.get_power(0) * self.params.N_gpus
        
        for t in range(self.params.T_sim):
            # 1. Generate arrivals for this time slot
            arrivals_t = self.arrivals.generate_arrivals()
            
            # 2. Compute current power
            # (We need this before deciding control for next step)
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
                   label='Target (300 kW)', linewidth=2)
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
        
        # Plot 3: Number of distinct states
        ax = axes[2]
        ax.plot(self.history['time'], self.history['queue_size'], 
                linewidth=2, color='purple')
        ax.set_xlabel('Time (slots)', fontsize=12)
        ax.set_ylabel('Number of States', fontsize=12)
        ax.set_title('State Space Size (Distinct (τ, s) pairs)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(r'C:/Users/lauso/OneDrive/Desktop/Cornell/AI flexible demand response/New model images/simulation_results.png', dpi=150, bbox_inches='tight')
        print("Plot saved to simulation_results.png")
        
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

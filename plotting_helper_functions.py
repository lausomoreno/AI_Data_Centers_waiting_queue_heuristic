def plot_state_heatmap(state, params):
    import numpy as np
    import matplotlib.pyplot as plt

    grid = np.zeros((params.tau_max + 1, params.s_max + 1))

    for (tau, s), count in state.n.items():
        tau_i = int(round(tau))
        s_i = int(round(s))
        if tau_i <= params.tau_max and s_i <= params.s_max:
            grid[tau_i, s_i] += count

    plt.figure(figsize=(10, 6))
    plt.imshow(grid.T, origin='lower', aspect='auto')
    plt.colorbar(label="Number of GPUs")

    plt.xlabel("Runtime (tau)")
    plt.ylabel("Slack (s)")
    plt.title("State Space Occupancy (n[tau, s])")

    plt.show()


def plot_runtime_distribution(state):
    import numpy as np
    import matplotlib.pyplot as plt

    tau_vals = []
    weights = []

    for (tau, s), count in state.n.items():
        tau_vals.append(tau)
        weights.append(count)

    tau_vals = np.array(tau_vals)
    weights = np.array(weights)

    plt.figure(figsize=(8, 5))
    plt.hist(tau_vals, bins=30, weights=weights, density=True)

    plt.xlabel("Runtime (tau)")
    plt.ylabel("Density")
    plt.title("Runtime Distribution")

    plt.show()



def plot_slack_distribution(state):
    import numpy as np
    import matplotlib.pyplot as plt

    s_vals = []
    weights = []

    for (tau, s), count in state.n.items():
        s_vals.append(s)
        weights.append(count)

    s_vals = np.array(s_vals)
    weights = np.array(weights)

    plt.figure(figsize=(8, 5))
    plt.hist(s_vals, bins=20, weights=weights, density=True)

    plt.xlabel("Slack (s)")
    plt.ylabel("Density")
    plt.title("Slack Distribution")

    plt.show()




plot_state_heatmap(sim.state, params)
plot_runtime_distribution(sim.state)
plot_slack_distribution(sim.state)

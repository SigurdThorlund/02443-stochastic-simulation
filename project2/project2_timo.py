import numpy as np
import matplotlib.pyplot as plt

# Function to simulate the stochastic SIR model
def simulate_sir_model(population_size, initial_infected, infection_rate, recovery_rate, time_steps):
    # Initialize compartments: S (Susceptible), I (Infected), R (Recovered)
    S = population_size - initial_infected
    I = initial_infected
    R = 0

    # Lists to store the population counts at each time step
    susceptible_count = [S]
    infected_count = [I]
    recovered_count = [R]

    # Simulate the model for the specified number of time steps
    for t in range(time_steps):
        # Calculate the rates of infection and recovery
        infection_probability = infection_rate * I / population_size
        recovery_probability = recovery_rate

        # Determine the number of new infections and recoveries
        new_infections = np.random.binomial(S, infection_probability)
        new_recoveries = np.random.binomial(I, recovery_probability)

        # Update the compartment counts
        S -= new_infections
        I += new_infections - new_recoveries
        R += new_recoveries

        # Append the current population counts to the lists
        susceptible_count.append(S)
        infected_count.append(I)
        recovered_count.append(R)

    return susceptible_count, infected_count, recovered_count

# Set the parameters for the simulation
population_size = 1000
initial_infected = 10
infection_rate = 0.2
recovery_rate = 0.1
time_steps = 100

# Run the simulation
susceptible, infected, recovered = simulate_sir_model(population_size, initial_infected, infection_rate, recovery_rate, time_steps)

# Plot the results
time = np.arange(time_steps + 1)
plt.plot(time, susceptible, label='Susceptible')
plt.plot(time, infected, label='Infected')
plt.plot(time, recovered, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population Count')
plt.title('Stochastic SIR Model')
plt.legend()
plt.show()

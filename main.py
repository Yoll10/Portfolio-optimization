from Data import get_data
from MCsimulation import run_simulation
import matplotlib.pyplot as plt

N_SIMULATIONS = 10000 

tickers, returns, cov = get_data()

results, weights, best_idx = run_simulation(tickers, returns, cov, num_portfolios=N_SIMULATIONS)

portfolio_data = []
for i in range(len(tickers)):
    ticker = tickers[i]
    w = weights[best_idx][i] #w for weights but name was already taken so just w
    portfolio_data.append((ticker, w))

print("\n" + "="*35)
print(f"{'ACTION':<12} | {'ALLOCATION':>10}")
print("="*35)

for ticker, w in portfolio_data:
    if w > 0.01:  
        print(f"{ticker:<12} | {w:>10.2%}")


print(f"\nEspected yield : {results[0, best_idx]:.2%}")
print(f"Volatility : {results[1, best_idx]:.2%}")

plt.show()
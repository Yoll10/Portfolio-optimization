import numpy as np
import matplotlib.pyplot as plt

def run_simulation(all_tickers, annual_returns, cov_matrix, num_portfolios):
    results = np.zeros((3, num_portfolios))
    all_weights = []
    risk_free_rate = 0.02 

    for i in range(num_portfolios):
        weights = np.random.random(len(all_tickers))
        weights /= np.sum(weights) 
        all_weights.append(weights)
        
        portfolio_return = np.dot(weights, annual_returns)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std_dev
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev 

    max_sharpe_idx = np.argmax(results[2])
    min_vol_idx = np.argmin(results[1])

    plt.figure(2)
    plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', s=10, alpha=0.2)
    plt.colorbar(label='Sharpe ratio')
    plt.scatter(results[1,max_sharpe_idx], results[0,max_sharpe_idx], color='red', marker='*', s=50, label='Optimal')
    plt.scatter(results[1,min_vol_idx], results[0,min_vol_idx], color='yellow', marker='*', s=50, label='Min Risque')
    plt.title(f'Markowitz Frontier : {num_portfolios} Simulations')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.legend()


    return results, all_weights, max_sharpe_idx
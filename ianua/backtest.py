import pandas as pd
import numpy as np

import quantstats as qs


def run_backtest(
    data_dict, rebalance_period, comission_rate, from_date, to_date="2100-01-01"
):
    prices = data_dict["close"].fillna(0)
    weights = data_dict["weights"].fillna(0)
    prices = prices.reindex(weights.index)

    indexes = prices.resample(rebalance_period).last()[from_date:to_date].index

    total_ret = None

    old_weight = weights.iloc[0] * 0

    for i in range(0, len(indexes) - 1):
        returns = prices[indexes[i] : indexes[i + 1]][1:]
        returns = returns / returns.iloc[0]

        weight = weights[indexes[i] : indexes[i + 1]].iloc[1]
        if np.abs(weight.values).sum() == 0:
            continue
        turnover = np.abs(weight.values - old_weight.values)
        turnover = turnover.sum()
        comission = turnover * comission_rate

        batch_ret = 1 + (returns * weight - weight)
        mask = ~np.isnan(returns.iloc[0])
        tickers = [weight.index[i] for i in range(len(mask)) if mask[i]]
        batch_ret = batch_ret[tickers].mean(1)
        total_ret = (
            batch_ret
            if total_ret is None
            else pd.concat(
                (total_ret, total_ret[-1] * batch_ret * (1 - comission)), axis=0
            )
        )
        old_weight = weight

    return total_ret


def run_backtest_with_report(output_file, benchmark=None, *args, **kwargs):
    returns = run_backtest(*args, **kwargs)
    qs.report.html(returns, output=output_file, benchmark=benchmark)
    return returns

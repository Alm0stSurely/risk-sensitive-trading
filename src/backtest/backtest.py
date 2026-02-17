"""
Backtesting framework for the trading agent.
Tests strategy performance on historical data.
"""

import json
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.fetch_market_data import fetch_historical_data
from data.indicators import calculate_all_indicators, get_latest_indicators
from portfolio.portfolio import Portfolio
from llm.trading_agent import TradingAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtesting engine for paper trading strategies.
    Simulates trading decisions on historical data.
    """
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 10000.0,
        tickers: Optional[List[str]] = None,
        rebalance_frequency: str = "daily"  # daily, weekly
    ):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.initial_capital = initial_capital
        self.tickers = tickers or ["SPY", "QQQ", "GLD"]
        self.rebalance_frequency = rebalance_frequency
        
        self.portfolio = None
        self.results = []
        
    def fetch_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for backtest period."""
        period = f"{(self.end_date - self.start_date).days + 60}d"
        logger.info(f"Fetching data for {len(self.tickers)} tickers, period: {period}")
        
        data = fetch_historical_data(self.tickers, period=period)
        
        # Filter to backtest date range
        filtered_data = {}
        for ticker, df in data.items():
            df = df[(df.index >= self.start_date.strftime("%Y-%m-%d")) & 
                    (df.index <= self.end_date.strftime("%Y-%m-%d"))]
            if not df.empty:
                filtered_data[ticker] = df
        
        return filtered_data
    
    def calculate_indicators_for_date(
        self,
        data: Dict[str, pd.DataFrame],
        current_date: datetime
    ) -> Dict:
        """Calculate indicators for all assets up to current date."""
        market_data = {}
        
        for ticker, df in data.items():
            # Get data up to current date
            mask = df.index <= current_date.strftime("%Y-%m-%d %H:%M:%S")
            hist = df[mask]
            
            if len(hist) < 20:
                continue
            
            # Calculate indicators
            df_with_ind = calculate_all_indicators(hist.copy())
            
            market_data[ticker] = {
                "dataframe": df_with_ind,
                "latest": get_latest_indicators(df_with_ind),
                "total_return": (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1 if len(hist) > 1 else 0
            }
        
        return market_data
    
    def run_backtest(
        self,
        use_llm: bool = False,
        strategy: str = "buy_and_hold"  # buy_and_hold, equal_weight, llm
    ) -> Dict:
        """
        Run the backtest simulation.
        
        Args:
            use_llm: Whether to use LLM for decisions (slower)
            strategy: Trading strategy to use
        
        Returns:
            Backtest results dictionary
        """
        logger.info(f"Starting backtest from {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Strategy: {strategy}, Use LLM: {use_llm}")
        
        # Fetch data
        data = self.fetch_historical_data()
        if not data:
            logger.error("No data fetched")
            return {}
        
        # Initialize portfolio
        self.portfolio = Portfolio(
            state_file=f"backtest_{strategy}_{self.start_date.strftime('%Y%m%d')}.json",
            data_dir="data/backtest"
        )
        
        # Get trading dates
        trading_dates = self._get_trading_dates(data)
        logger.info(f"Backtesting over {len(trading_dates)} trading days")
        
        # Initialize LLM agent if needed
        agent = TradingAgent() if use_llm else None
        
        # Run simulation
        for i, current_date in enumerate(trading_dates):
            if i % 20 == 0:
                logger.info(f"Processing day {i+1}/{len(trading_dates)}: {current_date.date()}")
            
            # Get prices for current date
            current_prices = self._get_prices_for_date(data, current_date)
            
            # Update portfolio with current prices
            self.portfolio.update_prices(current_prices)
            
            # Check if rebalance day
            should_rebalance = self._should_rebalance(i, current_date)
            
            if should_rebalance:
                if strategy == "llm" and use_llm:
                    self._execute_llm_strategy(data, current_date, agent, current_prices)
                elif strategy == "equal_weight":
                    self._execute_equal_weight_strategy(current_prices)
                # buy_and_hold: do nothing after initial purchase
            
            # Record daily result
            self._record_daily_result(current_date, current_prices)
        
        # Calculate final metrics
        metrics = self._calculate_metrics()
        
        return {
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "end_date": self.end_date.strftime("%Y-%m-%d"),
            "strategy": strategy,
            "initial_capital": self.initial_capital,
            "final_value": self.portfolio.total_value,
            "total_return": metrics["total_return"],
            "annualized_return": metrics["annualized_return"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "volatility": metrics["volatility"],
            "num_trades": len(self.portfolio.trades),
            "daily_results": self.results
        }
    
    def _get_trading_dates(self, data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """Get list of trading dates from data."""
        # Use the first ticker's dates
        first_ticker = list(data.keys())[0]
        dates = pd.to_datetime(data[first_ticker].index).tolist()
        return [d for d in dates if self.start_date <= d <= self.end_date]
    
    def _get_prices_for_date(
        self,
        data: Dict[str, pd.DataFrame],
        date: datetime
    ) -> Dict[str, float]:
        """Get closing prices for all tickers on given date."""
        prices = {}
        date_str = date.strftime("%Y-%m-%d")
        
        for ticker, df in data.items():
            # Find row for this date
            mask = df.index.strftime("%Y-%m-%d") == date_str
            rows = df[mask]
            if not rows.empty:
                prices[ticker] = float(rows['Close'].iloc[-1])
        
        return prices
    
    def _should_rebalance(self, day_index: int, current_date: datetime) -> bool:
        """Determine if we should rebalance on this day."""
        if self.rebalance_frequency == "daily":
            return True
        elif self.rebalance_frequency == "weekly":
            return day_index % 5 == 0
        return False
    
    def _execute_llm_strategy(
        self,
        data: Dict[str, pd.DataFrame],
        current_date: datetime,
        agent: TradingAgent,
        current_prices: Dict[str, float]
    ):
        """Execute LLM-based trading strategy."""
        # Calculate indicators
        market_data = self.calculate_indicators_for_date(data, current_date)
        
        # Get portfolio summary
        portfolio_summary = self.portfolio.get_summary()
        
        # Get LLM decision
        decision = agent.get_trading_decision(market_data, portfolio_summary)
        
        # Execute actions
        for action in decision.get("actions", []):
            ticker = action.get("ticker")
            action_type = action.get("action")
            pct = action.get("pct", 0)
            
            if ticker not in current_prices:
                continue
            
            price = current_prices[ticker]
            
            if action_type == "buy":
                self.portfolio.buy(ticker, pct, price)
            elif action_type == "sell":
                self.portfolio.sell(ticker, price)
    
    def _execute_equal_weight_strategy(self, current_prices: Dict[str, float]):
        """Execute equal weight buy-and-hold strategy."""
        # Only execute once at start
        if self.portfolio.positions:
            return
        
        # Buy equal weight in all available tickers
        n_tickers = len(current_prices)
        if n_tickers == 0:
            return
        
        pct_per_ticker = 90.0 / n_tickers  # 90% invested, 10% cash buffer
        
        for ticker, price in current_prices.items():
            self.portfolio.buy(ticker, pct_per_ticker, price)
    
    def _record_daily_result(self, date: datetime, prices: Dict[str, float]):
        """Record daily portfolio state."""
        summary = self.portfolio.get_summary()
        
        self.results.append({
            "date": date.strftime("%Y-%m-%d"),
            "total_value": summary["total_value"],
            "cash": summary["cash"],
            "positions_value": summary["positions_value"],
            "total_return_pct": summary["total_return_pct"],
            "num_positions": summary["num_positions"]
        })
    
    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.results:
            return {}
        
        values = [r["total_value"] for r in self.results]
        returns = [(values[i] - values[i-1]) / values[i-1] 
                   for i in range(1, len(values))]
        
        # Total return
        total_return = (values[-1] / self.initial_capital) - 1
        
        # Annualized return
        days = len(self.results)
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252) if returns else 0
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = ((annualized_return - risk_free_rate) / volatility 
                       if volatility > 0 else 0)
        
        # Max drawdown
        peak = self.initial_capital
        max_drawdown = 0
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }


def run_comparison_backtest(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    tickers: Optional[List[str]] = None
) -> Dict:
    """
    Run backtest comparing multiple strategies.
    
    Returns:
        Comparison results
    """
    tickers = tickers or ["SPY", "QQQ", "GLD"]
    
    strategies = ["buy_and_hold", "equal_weight"]
    results = {}
    
    for strategy in strategies:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running backtest: {strategy}")
        logger.info(f"{'='*60}")
        
        engine = BacktestEngine(
            start_date=start_date,
            end_date=end_date,
            tickers=tickers,
            rebalance_frequency="daily"
        )
        
        result = engine.run_backtest(use_llm=False, strategy=strategy)
        results[strategy] = result
        
        if result:
            logger.info(f"\nResults for {strategy}:")
            logger.info(f"  Total Return: {result['total_return']*100:.2f}%")
            logger.info(f"  Annualized Return: {result['annualized_return']*100:.2f}%")
            logger.info(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            logger.info(f"  Max Drawdown: {result['max_drawdown']*100:.2f}%")
            logger.info(f"  Volatility: {result['volatility']*100:.2f}%")
            logger.info(f"  Number of Trades: {result['num_trades']}")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Run comparison backtest
    results = run_comparison_backtest(
        start_date="2024-06-01",
        end_date="2024-12-31",
        tickers=["SPY", "QQQ", "GLD"]
    )
    
    # Save results
    output_file = "results/backtest_comparison.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_file}")

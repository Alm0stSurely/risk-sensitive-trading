#!/usr/bin/env python3
"""
LLM Decision Quality Analyzer.

Analyzes the quality of LLM trading decisions by comparing predictions
with actual market outcomes. Tracks hit rate, risk-adjusted returns,
and behavioral biases in LLM decisions.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / ".."))

from data.fetch_market_data import fetch_historical_data


class DecisionAnalyzer:
    """Analyze quality of LLM trading decisions."""
    
    def __init__(self, results_dir: str = "results/daily"):
        self.results_dir = Path(results_dir)
        self.analysis = {
            "total_decisions": 0,
            "buy_decisions": 0,
            "sell_decisions": 0,
            "hold_decisions": 0,
            "successful_buys": 0,  # Price went up after buy
            "successful_sells": 0,  # Price went down after sell
            "avg_return_after_buy": 0.0,
            "avg_return_after_sell": 0.0,
            "timing_score": 0.0,  # How well does LLM time entries/exits
            "risk_score": 0.0,  # Does LLM respect risk limits
            "behavioral_biases": {
                "loss_aversion_adherence": 0.0,  # Does LLM cut losses quickly
                "profit_taking_timing": 0.0,  # Does LLM let winners run
                "overtrading_tendency": 0.0,  # Too many trades?
            }
        }
    
    def load_decisions(self, days: int = 30) -> List[Dict]:
        """Load historical decisions from daily results."""
        decisions = []
        
        if not self.results_dir.exists():
            return decisions
        
        # Get files from last N days
        files = sorted(self.results_dir.glob("*.json"))[-days:]
        
        for file in files:
            try:
                with open(file) as f:
                    data = json.load(f)
                    if "decision" in data and "executed_trades" in data:
                        decisions.append({
                            "date": data.get("date", file.stem),
                            "timestamp": data.get("timestamp"),
                            "actions": data["decision"].get("actions", []),
                            "trades": data.get("executed_trades", []),
                            "reasoning": data["decision"].get("reasoning", "")
                        })
            except Exception as e:
                print(f"Warning: Could not load {file}: {e}")
                continue
        
        return decisions
    
    def analyze_outcomes(self, decisions: List[Dict], 
                        forward_days: int = 5) -> Dict:
        """
        Analyze outcomes of decisions vs forward market performance.
        
        Args:
            decisions: List of decision records
            forward_days: How many days forward to check performance
        """
        outcomes = {
            "buys": [],
            "sells": [],
            "holds": [],
        }
        
        for decision in decisions:
            date = decision["date"]
            
            for trade in decision.get("trades", []):
                ticker = trade["ticker"]
                action = trade["action"]
                price = trade.get("price", 0)
                
                if price == 0:
                    continue
                
                # Calculate forward return
                forward_return = self._get_forward_return(
                    ticker, date, price, forward_days
                )
                
                record = {
                    "ticker": ticker,
                    "date": date,
                    "entry_price": price,
                    f"return_{forward_days}d": forward_return,
                    "success": False
                }
                
                if action == "buy":
                    record["success"] = forward_return > 0
                    outcomes["buys"].append(record)
                elif action == "sell":
                    record["success"] = forward_return < 0
                    outcomes["sells"].append(record)
        
        return self._calculate_metrics(outcomes)
    
    def _get_forward_return(self, ticker: str, date: str, 
                           entry_price: float, days: int) -> float:
        """Calculate forward return for a decision."""
        try:
            # Fetch data from date onwards
            from data.fetch_market_data import fetch_historical_data
            
            data = fetch_historical_data([ticker], period=f"{days+5}d")
            
            if ticker not in data or data[ticker].empty:
                return 0.0
            
            df = data[ticker]
            
            # Find entry date index
            df.index = pd.to_datetime(df.index)
            entry_date = pd.to_datetime(date)
            
            mask = df.index >= entry_date
            if not mask.any():
                return 0.0
            
            future_data = df[mask]
            if len(future_data) < days:
                return 0.0
            
            exit_price = future_data["Close"].iloc[days - 1]
            return (exit_price - entry_price) / entry_price
            
        except Exception as e:
            return 0.0
    
    def _calculate_metrics(self, outcomes: Dict) -> Dict:
        """Calculate aggregate metrics from outcomes."""
        metrics = {
            "buy_accuracy": 0.0,
            "sell_accuracy": 0.0,
            "avg_forward_return_buy": 0.0,
            "avg_forward_return_sell": 0.0,
            "sharpe_of_decisions": 0.0,
            "total_decisions": 0,
            "win_rate": 0.0,
        }
        
        # Buy metrics
        if outcomes["buys"]:
            buy_returns = [r["return_5d"] for r in outcomes["buys"]]
            buy_successes = sum(1 for r in outcomes["buys"] if r["success"])
            
            metrics["buy_accuracy"] = buy_successes / len(outcomes["buys"])
            metrics["avg_forward_return_buy"] = np.mean(buy_returns)
        
        # Sell metrics
        if outcomes["sells"]:
            sell_returns = [r["return_5d"] for r in outcomes["sells"]]
            sell_successes = sum(1 for r in outcomes["sells"] if r["success"])
            
            metrics["sell_accuracy"] = sell_successes / len(outcomes["sells"])
            # For sells, negative return is good (price went down)
            metrics["avg_forward_return_sell"] = -np.mean(sell_returns)
        
        # Overall metrics
        all_decisions = outcomes["buys"] + outcomes["sells"]
        if all_decisions:
            metrics["total_decisions"] = len(all_decisions)
            wins = sum(1 for d in all_decisions if d["success"])
            metrics["win_rate"] = wins / len(all_decisions)
            
            # Pseudo-Sharpe of decisions
            returns = [r["return_5d"] for r in all_decisions]
            if len(returns) > 1 and np.std(returns) > 0:
                metrics["sharpe_of_decisions"] = np.mean(returns) / np.std(returns)
        
        return metrics
    
    def analyze_behavioral_patterns(self, decisions: List[Dict]) -> Dict:
        """Analyze behavioral patterns in LLM decisions."""
        patterns = {
            "herding_tendency": 0.0,  # Does LLM follow recent trends
            "loss_aversion_score": 0.0,  # Quick to cut losses?
            "overconfidence_check": 0.0,  # Does it overtrade?
            "diversification_score": 0.0,  # Portfolio concentration
        }
        
        if not decisions:
            return patterns
        
        # Count unique tickers traded
        all_tickers = set()
        total_trades = 0
        
        for decision in decisions:
            for trade in decision.get("trades", []):
                all_tickers.add(trade["ticker"])
                total_trades += 1
        
        # Overconfidence: too many trades per day
        avg_trades_per_day = total_trades / len(decisions) if decisions else 0
        patterns["overconfidence_check"] = 1.0 if avg_trades_per_day <= 3 else 0.5
        
        # Diversification: trades spread across many assets
        patterns["diversification_score"] = min(len(all_tickers) / 10, 1.0)
        
        return patterns
    
    def generate_report(self, days: int = 30) -> str:
        """Generate comprehensive analysis report."""
        decisions = self.load_decisions(days)
        
        if not decisions:
            return "No decision data available for analysis."
        
        outcomes = self.analyze_outcomes(decisions)
        behavioral = self.analyze_behavioral_patterns(decisions)
        
        report = f"""
{'='*70}
LLM DECISION QUALITY ANALYSIS
Analysis Period: Last {len(decisions)} trading days
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

PERFORMANCE METRICS
-------------------
Total Decisions Analyzed: {outcomes['total_decisions']}
Overall Win Rate: {outcomes['win_rate']*100:.1f}%

Buy Decisions:
  - Accuracy: {outcomes['buy_accuracy']*100:.1f}%
  - Avg 5D Return: {outcomes['avg_forward_return_buy']*100:+.2f}%

Sell Decisions:
  - Accuracy: {outcomes['sell_accuracy']*100:.1f}%
  - Avg 5D Return (avoided): {outcomes['avg_forward_return_sell']*100:+.2f}%

Decision Sharpe Ratio: {outcomes['sharpe_of_decisions']:.3f}

BEHAVIORAL ANALYSIS
-------------------
Overconfidence Check: {behavioral['overconfidence_check']:.1f}/1.0
  (Lower trading frequency = better discipline)

Diversification Score: {behavioral['diversification_score']:.1f}/1.0
  (Number of unique assets traded)

ASSESSMENT
----------
"""
        
        # Add qualitative assessment
        if outcomes['win_rate'] > 0.55:
            report += "✓ LLM shows predictive skill (win rate > 55%)\n"
        elif outcomes['win_rate'] > 0.45:
            report += "~ LLM performance is near random (45-55% win rate)\n"
        else:
            report += "⚠ LLM underperforming (win rate < 45%)\n"
        
        if outcomes['sharpe_of_decisions'] > 0.5:
            report += "✓ Positive risk-adjusted returns from decisions\n"
        elif outcomes['sharpe_of_decisions'] < 0:
            report += "⚠ Negative risk-adjusted returns - review strategy\n"
        
        report += f"\n{'='*70}\n"
        
        return report


def main():
    """Run decision analysis."""
    analyzer = DecisionAnalyzer()
    report = analyzer.generate_report(days=30)
    print(report)
    
    # Save report
    output_dir = Path("results/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"decision_analysis_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":
    main()

#!/repos/almost-surely-profitable/.venv/bin/python3
"""
Intraday monitoring script.
Checks for significant price movements and alerts if thresholds are breached.
Called every 2 hours during market hours (8h-20h UTC) by external cron.

Improvements:
- Dynamic indices from universe.json
- Bollinger Bands breakout detection
- Configurable thresholds
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data.fetch_market_data import fetch_current_prices, get_tickers_by_category
from data.indicators import analyze_market_data
from portfolio.portfolio import Portfolio


# Default alert thresholds (can be overridden via config)
DEFAULT_THRESHOLDS = {
    'position_movement_pct': 2.0,      # Alert if portfolio position moves > 2%
    'index_movement_pct': 3.0,         # Alert if index moves > 3%
    'portfolio_drawdown_pct': 1.5,     # Alert if portfolio draws down > 1.5%
    'bollinger_breakout_pct': 2.0,     # Alert if price breaks BB by > 2%
}


def load_config() -> Dict:
    """Load monitor configuration from config file if exists."""
    config_path = Path("config/monitor.json")
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def get_indices_to_monitor() -> List[str]:
    """
    Get benchmark indices to monitor from universe config.
    Returns ETF category tickers (typically major indices).
    """
    # Primary benchmarks: ETFs representing major indices
    etf_tickers = get_tickers_by_category('etf')
    
    # Also include commodities as macro indicators
    commodity_tickers = get_tickers_by_category('commodity')
    
    # Combine - limit to most liquid/important ones
    key_tickers = []
    
    # Key equity indices (always monitor these)
    priority_etfs = ['SPY', 'QQQ', '^FCHI', 'FEZ']
    for t in priority_etfs:
        if t in etf_tickers:
            key_tickers.append(t)
    
    # Add remaining ETFs
    for t in etf_tickers:
        if t not in key_tickers:
            key_tickers.append(t)
    
    # Add key commodities as macro indicators
    priority_commodities = ['GLD', 'PDBC', 'USO']
    for t in priority_commodities:
        if t in commodity_tickers and t not in key_tickers:
            key_tickers.append(t)
    
    return key_tickers


def load_previous_close(portfolio: Portfolio) -> Dict[str, float]:
    """
    Load previous closing prices from portfolio state or market data.
    """
    state_file = Path("data/market_state.json")
    
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            return state.get('previous_close', {})
        except:
            pass
    
    # If no saved state, use average price of positions as reference
    return {ticker: pos.avg_price for ticker, pos in portfolio.positions.items()}


def save_market_state(prices: Dict[str, float]):
    """Save current prices as reference for next check."""
    state_file = Path("data/market_state.json")
    state_file.parent.mkdir(exist_ok=True)
    
    state = {
        'timestamp': datetime.now().isoformat(),
        'previous_close': prices
    }
    
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def check_bollinger_breakouts(
    market_analysis: Dict,
    current_prices: Dict[str, float],
    threshold_pct: float
) -> List[Dict]:
    """
    Check for Bollinger Bands breakouts.
    
    Args:
        market_analysis: Analysis data with Bollinger Bands
        current_prices: Current price snapshot
        threshold_pct: Minimum breakout percentage to trigger alert
    
    Returns:
        List of breakout alerts
    """
    alerts = []
    
    for ticker, analysis in market_analysis.get('assets', {}).items():
        current_price = current_prices.get(ticker)
        if not current_price:
            continue
        
        latest = analysis.get('latest', {})
        bb_upper = latest.get('bb_upper')
        bb_lower = latest.get('bb_lower')
        
        if bb_upper is None or bb_lower is None:
            continue
        
        # Check upper breakout
        if current_price > bb_upper:
            breakout_pct = ((current_price - bb_upper) / bb_upper) * 100
            if breakout_pct >= threshold_pct:
                alerts.append({
                    'type': 'bollinger_breakout_upper',
                    'ticker': ticker,
                    'severity': 'medium',
                    'current_price': current_price,
                    'bb_upper': bb_upper,
                    'bb_lower': bb_lower,
                    'breakout_pct': breakout_pct,
                    'message': f'{ticker} broke above upper Bollinger Band by {breakout_pct:.2f}%'
                })
        
        # Check lower breakout
        elif current_price < bb_lower:
            breakout_pct = ((bb_lower - current_price) / bb_lower) * 100
            if breakout_pct >= threshold_pct:
                alerts.append({
                    'type': 'bollinger_breakout_lower',
                    'ticker': ticker,
                    'severity': 'medium',
                    'current_price': current_price,
                    'bb_upper': bb_upper,
                    'bb_lower': bb_lower,
                    'breakout_pct': breakout_pct,
                    'message': f'{ticker} broke below lower Bollinger Band by {breakout_pct:.2f}%'
                })
    
    return alerts


def check_movements(
    current_prices: Dict[str, float],
    reference_prices: Dict[str, float],
    portfolio: Portfolio,
    indices: List[str],
    thresholds: Dict
) -> List[Dict]:
    """
    Check for significant price movements.
    
    Returns:
        List of alerts
    """
    alerts = []
    
    # Check portfolio positions
    for ticker, position in portfolio.positions.items():
        current_price = current_prices.get(ticker)
        if not current_price:
            continue
        
        # Use position average price as reference
        reference_price = position.avg_price
        
        if reference_price > 0:
            movement_pct = ((current_price - reference_price) / reference_price) * 100
            
            if abs(movement_pct) >= thresholds['position_movement_pct']:
                alerts.append({
                    'type': 'position_movement',
                    'ticker': ticker,
                    'severity': 'high' if abs(movement_pct) > 5 else 'medium',
                    'current_price': current_price,
                    'reference_price': reference_price,
                    'movement_pct': movement_pct,
                    'position_size': position.market_value,
                    'unrealized_pnl': position.unrealized_pnl
                })
    
    # Check indices/benchmarks
    for index in indices:
        current_price = current_prices.get(index)
        reference_price = reference_prices.get(index)
        
        if current_price and reference_price:
            movement_pct = ((current_price - reference_price) / reference_price) * 100
            
            if abs(movement_pct) >= thresholds['index_movement_pct']:
                alerts.append({
                    'type': 'index_movement',
                    'ticker': index,
                    'severity': 'high',
                    'current_price': current_price,
                    'reference_price': reference_price,
                    'movement_pct': movement_pct
                })
    
    # Check portfolio drawdown
    if portfolio.positions:
        total_cost = sum(pos.cost_basis for pos in portfolio.positions.values())
        total_current = sum(
            current_prices.get(ticker, pos.current_price) * pos.quantity
            for ticker, pos in portfolio.positions.items()
        )
        
        if total_cost > 0:
            drawdown_pct = ((total_current - total_cost) / total_cost) * 100
            
            if drawdown_pct <= -thresholds['portfolio_drawdown_pct']:
                alerts.append({
                    'type': 'portfolio_drawdown',
                    'ticker': 'PORTFOLIO',
                    'severity': 'critical',
                    'current_value': total_current,
                    'cost_basis': total_cost,
                    'drawdown_pct': drawdown_pct
                })
    
    return alerts


def run_monitor():
    """Run the monitoring check."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting intraday monitor...")
    
    # Load configuration
    config = load_config()
    thresholds = config.get('thresholds', DEFAULT_THRESHOLDS)
    
    # Load portfolio
    portfolio = Portfolio(data_dir="data")
    
    # Get dynamic indices from universe config
    indices = get_indices_to_monitor()
    print(f"Monitoring {len(indices)} benchmark indices: {', '.join(indices[:5])}...")
    
    # Get tickers to monitor (positions + indices)
    tickers_to_monitor = list(portfolio.positions.keys()) + indices
    
    if not tickers_to_monitor:
        print("No positions to monitor.")
        return [], 0
    
    # Fetch current prices
    print(f"Fetching prices for {len(tickers_to_monitor)} tickers...")
    current_prices = fetch_current_prices(tickers_to_monitor)
    
    # Fetch market data for Bollinger analysis
    print("Fetching market data for technical analysis...")
    from data.fetch_market_data import fetch_historical_data
    market_data = fetch_historical_data(tickers_to_monitor, period="10d")
    market_analysis = analyze_market_data(market_data)
    
    # Load reference prices
    reference_prices = load_previous_close(portfolio)
    
    # Check for movements
    alerts = check_movements(
        current_prices, reference_prices, portfolio, indices, thresholds
    )
    
    # Check for Bollinger Bands breakouts
    bb_alerts = check_bollinger_breakouts(
        market_analysis, current_prices, thresholds.get('bollinger_breakout_pct', 2.0)
    )
    alerts.extend(bb_alerts)
    
    # Save current prices for next check
    save_market_state(current_prices)
    
    # Output results
    if alerts:
        print(f"\n⚠️  {len(alerts)} ALERT(S) DETECTED:\n")
        
        for alert in alerts:
            print(f"Type: {alert['type'].upper()}")
            print(f"Ticker: {alert['ticker']}")
            print(f"Severity: {alert['severity'].upper()}")
            
            if alert['type'] == 'position_movement':
                print(f"Movement: {alert['movement_pct']:+.2f}%")
                print(f"Price: €{alert['current_price']:.2f} (ref: €{alert['reference_price']:.2f})")
                print(f"Position P&L: €{alert['unrealized_pnl']:+.2f}")
            elif alert['type'] == 'index_movement':
                print(f"Movement: {alert['movement_pct']:+.2f}%")
                print(f"Price: €{alert['current_price']:.2f}")
            elif alert['type'] == 'portfolio_drawdown':
                print(f"Drawdown: {alert['drawdown_pct']:.2f}%")
                print(f"Value: €{alert['current_value']:.2f} (cost: €{alert['cost_basis']:.2f})")
            elif 'bollinger' in alert['type']:
                print(f"Breakout: {alert['breakout_pct']:.2f}%")
                print(f"Price: €{alert['current_price']:.2f}")
                print(f"BB Upper: €{alert['bb_upper']:.2f}, BB Lower: €{alert['bb_lower']:.2f}")
            
            print("-" * 50)
        
        # Output JSON for external processing
        output = {
            'timestamp': datetime.now().isoformat(),
            'alert_count': len(alerts),
            'alerts': alerts,
            'portfolio_value': portfolio.total_value,
            'indices_monitored': indices
        }
        print("\nJSON_OUTPUT:")
        print(json.dumps(output, indent=2))
        
        return alerts, 1  # Exit code 1 = alert triggered
    else:
        print("✓ No significant movements detected.")
        print(f"Portfolio Value: €{portfolio.total_value:.2f}")
        
        return [], 0  # Exit code 0 = normal


if __name__ == "__main__":
    try:
        alerts, exit_code = run_monitor()
        sys.exit(exit_code)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)  # Exit code 2 = error

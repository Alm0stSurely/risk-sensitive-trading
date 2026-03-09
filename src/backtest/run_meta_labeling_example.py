#!/usr/bin/env python3
"""
Example script demonstrating Meta-Labeling usage.

This script shows how to:
1. Generate synthetic primary signals
2. Simulate triple-barrier outcomes
3. Train a meta-labeling model
4. Filter and size new signals
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtest import (
    PrimarySignal,
    SignalType,
    MetaLabeler,
    MetaLabelingConfig,
    create_meta_labels_from_triple_barrier
)


def generate_synthetic_data(n_days: int = 500) -> pd.DataFrame:
    """Generate synthetic OHLCV data with indicators."""
    np.random.seed(42)
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='D')
    
    # Random walk for price
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_days)),
        'high': prices * (1 + abs(np.random.normal(0, 0.015, n_days))),
        'low': prices * (1 - abs(np.random.normal(0, 0.015, n_days))),
        'close': prices,
        'volume': np.random.randint(1_000_000, 10_000_000, n_days)
    }, index=dates)
    
    # Add technical indicators
    df['rsi'] = calculate_rsi(df['close'])
    df['sma20'] = df['close'].rolling(20).mean()
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    
    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Simple RSI calculation."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def generate_primary_signals(
    price_data: pd.DataFrame,
    n_signals: int = 200
) -> list:
    """Generate synthetic primary model signals."""
    np.random.seed(123)
    
    # Random timestamps within data range
    timestamps = np.random.choice(price_data.index[50:], n_signals)
    
    signals = []
    for ts in timestamps:
        # Random signal with some bias toward momentum
        rsi_at_ts = price_data.loc[ts, 'rsi'] if 'rsi' in price_data.columns else 50
        
        # Mean-reversion primary model: buy when RSI low, sell when high
        if rsi_at_ts < 35:
            signal_type = SignalType.BUY
        elif rsi_at_ts > 65:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD if np.random.random() > 0.5 else SignalType.BUY
        
        signals.append(PrimarySignal(
            timestamp=ts,
            ticker='SYNTH',
            signal=signal_type,
            confidence=np.random.uniform(0.5, 0.9)
        ))
    
    return sorted(signals, key=lambda x: x.timestamp)


def simulate_outcomes(
    signals: list,
    price_data: pd.DataFrame,
    win_rate: float = 0.55
) -> list:
    """
    Simulate triple-barrier outcomes.
    
    Success (1): Upper barrier touched (profit)
    Failure (0): Lower barrier or vertical barrier
    """
    np.random.seed(456)
    outcomes = []
    
    for signal in signals:
        # Get price context
        ts = signal.timestamp
        mask = price_data.index <= ts
        available = price_data[mask]
        
        if len(available) < 20:
            outcomes.append(0)
            continue
        
        recent = available.tail(20)
        volatility = recent['close'].pct_change().std()
        
        # Simulate outcome based on signal quality and volatility
        # Better signals in low volatility = higher success rate
        if signal.signal == SignalType.BUY:
            # Success more likely if RSI was truly oversold
            rsi = recent['rsi'].iloc[-1] if 'rsi' in recent.columns else 50
            quality = 1 - (rsi / 100)  # Lower RSI = better quality
        else:
            quality = 0.5
        
        # Volatility penalizes success
        vol_penalty = min(volatility * 10, 0.2)
        
        # Calculate success probability
        p_success = win_rate * quality - vol_penalty
        p_success = np.clip(p_success, 0.1, 0.9)
        
        outcome = 1 if np.random.random() < p_success else 0
        outcomes.append(outcome)
    
    return outcomes


def main():
    print("=" * 60)
    print("Meta-Labeling Demonstration")
    print("=" * 60)
    
    # Step 1: Generate synthetic data
    print("\n1. Generating synthetic price data...")
    price_data = generate_synthetic_data(n_days=500)
    print(f"   Generated {len(price_data)} days of data")
    print(f"   Price range: ${price_data['close'].min():.2f} - ${price_data['close'].max():.2f}")
    
    # Step 2: Generate primary signals
    print("\n2. Generating primary model signals...")
    all_signals = generate_primary_signals(price_data, n_signals=200)
    buy_signals = [s for s in all_signals if s.signal == SignalType.BUY]
    sell_signals = [s for s in all_signals if s.signal == SignalType.SELL]
    print(f"   Total signals: {len(all_signals)}")
    print(f"   Buy signals: {len(buy_signals)}")
    print(f"   Sell signals: {len(sell_signals)}")
    
    # Step 3: Split into train/test
    split_idx = int(len(all_signals) * 0.7)
    train_signals = all_signals[:split_idx]
    test_signals = all_signals[split_idx:]
    
    print(f"\n3. Split: {len(train_signals)} train, {len(test_signals)} test")
    
    # Step 4: Simulate outcomes
    print("\n4. Simulating triple-barrier outcomes...")
    train_outcomes = simulate_outcomes(train_signals, price_data, win_rate=0.55)
    
    success_rate = sum(train_outcomes) / len(train_outcomes)
    print(f"   Training success rate: {success_rate:.1%}")
    
    # Step 5: Train meta-labeler
    print("\n5. Training meta-labeling model...")
    config = MetaLabelingConfig(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        min_probability=0.55,  # Only take trades with >55% predicted success
        max_position_pct=0.20,
        kelly_fraction=0.5
    )
    
    labeler = MetaLabeler(config)
    labeler.fit(train_signals, price_data, train_outcomes)
    
    print(f"\n   Model trained!")
    print(f"   Accuracy: {labeler.metrics.get('accuracy', 0):.2%}")
    print(f"   Precision: {labeler.metrics.get('precision', 0):.2%}")
    print(f"   Recall: {labeler.metrics.get('recall', 0):.2%}")
    print(f"   ROC-AUC: {labeler.metrics.get('roc_auc', 0):.2%}")
    
    # Step 6: Predict on test signals
    print("\n6. Predicting on test signals...")
    predictions = labeler.predict(test_signals, price_data)
    
    # Show probability distribution
    probas = [p.predicted_proba for p in predictions if p.predicted_proba is not None]
    print(f"   Mean predicted probability: {np.mean(probas):.2%}")
    print(f"   Probability range: {min(probas):.2%} - {max(probas):.2%}")
    
    # Step 7: Size positions
    print("\n7. Calculating position sizes...")
    sized = labeler.size_positions(predictions, avg_win_loss_ratio=1.5)
    
    # Filter trades
    filtered = labeler.filter_signals(sized)
    
    total_signals = len(sized)
    accepted = len(filtered)
    rejected = total_signals - accepted
    
    print(f"   Total signals: {total_signals}")
    print(f"   Accepted (≥{config.min_probability:.0%} prob): {accepted}")
    print(f"   Rejected: {rejected} ({rejected/total_signals:.1%})")
    
    # Show position size distribution
    sizes = [s.position_size for s in sized if s.position_size and s.position_size > 0]
    if sizes:
        print(f"\n   Position sizes:")
        print(f"   - Min: {min(sizes):.1%}")
        print(f"   - Max: {max(sizes):.1%}")
        print(f"   - Mean: {np.mean(sizes):.1%}")
    
    # Step 8: Show example trades
    print("\n8. Example filtered trades:")
    for label in filtered[:5]:
        print(f"   {label.signal.timestamp.strftime('%Y-%m-%d')} | "
              f"{label.signal.signal.name:4} | "
              f"Prob: {label.predicted_proba:.1%} | "
              f"Size: {label.position_size:.1%}")
    
    print("\n" + "=" * 60)
    print("Key Insight:")
    print(f"Meta-labeling filtered out {rejected}/{total_signals} signals")
    print(f"({rejected/total_signals:.1%}) that likely would have failed.")
    print("This reduces false positives without manual intervention.")
    print("=" * 60)


if __name__ == "__main__":
    main()

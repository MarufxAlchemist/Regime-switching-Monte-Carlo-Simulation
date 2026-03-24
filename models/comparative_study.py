from models import backtest_engine as be
from models import validation_tests as vt

def run_backtest(model_type):
    """Runs backtest for given model type and returns results in a dictionary"""
    
    df = be.run_backtest(model_type)

    # Calculate metrics using the DataFrame
    avg_var = df['predicted_var_95'].mean()
    avg_loss = df['realized_loss'].mean()
    violation_rate = df['violation'].sum() / len(df)
    
    kupiec_test = vt.kupiec_test(df['predicted_var_95'], df['realized_loss'])
    christoffersen_test = vt.christoffersen_test(df['violation'])

    result = {
        "avg_var": avg_var,
        "avg_loss": avg_loss,
        "violation_rate": violation_rate,
        "kupiec_test": kupiec_test,
        "christoffersen_test": christoffersen_test,
    }
    
    return result

def main():
    baseline = run_backtest('baseline')
    regime = run_backtest('regime')
    full = run_backtest('full')

    results = {
        "baseline": baseline,
        "regime": regime,
        "full": full,
    }
    
    return results

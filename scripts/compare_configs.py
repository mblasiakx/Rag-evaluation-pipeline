import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'results')
METRICS = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'completeness']

def load_results() -> pd.DataFrame:
    frames = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith('_results.csv'):
            continue
        config_name = fname.replace('_results.csv', '')
        df = pd.read_csv(os.path.join(RESULTS_DIR, fname))
        df['config'] = config_name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No *_results.csv files found in {RESULTS_DIR}")
    return pd.concat(frames, ignore_index=True)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    available = [m for m in METRICS if m in df.columns]
    summary = df.groupby('config')[available].mean().round(4)
    summary['latency_ms'] = df.groupby('config')['latency_ms'].mean().round(0)
    summary['n_questions'] = df.groupby('config').size()
    return summary


def rank_configs(summary: pd.DataFrame) -> pd.Series:
    score_cols = [m for m in METRICS if m in summary.columns]
    return summary[score_cols].mean(axis=1).sort_values(ascending=False)


def print_report(summary: pd.DataFrame, ranking: pd.Series) -> None:
    print("\n=== Metric Averages per Config ===")
    print(summary.to_string())

    print("\n=== Overall Ranking (mean of all metrics) ===")
    for i, (config, score) in enumerate(ranking.items(), 1):
        print(f"  {i}. {config}: {score:.4f}")

    best = ranking.index[0]
    print(f"\nBest config: {best}")

    print("\n=== Per-metric Winner ===")
    score_cols = [m for m in METRICS if m in summary.columns]
    for metric in score_cols:
        winner = summary[metric].idxmax()
        value = summary[metric].max()
        print(f"  {metric}: {winner} ({value:.4f})")


if __name__ == '__main__':
    df = load_results()
    summary = build_summary(df)
    ranking = rank_configs(summary)
    print_report(summary, ranking)

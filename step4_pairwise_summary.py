"""Summarize paired human visual preferences; these are not retrieval accuracy scores."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description='Summarize pairwise A/B visual preferences')
    p.add_argument('--query-id', required=True)
    p.add_argument('--annotator', default='judge01')
    p.add_argument('--infile', type=Path, default=Path('outputs/pairwise_preferences.csv'))
    a = p.parse_args()
    if not a.infile.is_file():
        p.error(f'Missing input file: {a.infile}. Run step4_pairwise_rating.py first.')
    key = 'gallery:' + a.query_id
    votes = {}
    with a.infile.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row.get('query_key') == key and row.get('annotator') == a.annotator:
                votes[(row['left_image_id'], row['right_image_id'])] = row['choice']
    if not votes:
        p.error(f'No votes for {key} by {a.annotator}')
    counts = Counter(votes.values())
    print(f'Query: {key} | Annotator: {a.annotator} | Rated pairs: {len(votes)}')
    print('Choices:', ' | '.join(f'{k}={counts[k]}' for k in ('A','B','tie','neither')))
    results = defaultdict(lambda: [0, 0, 0])
    for (left, right), choice in votes.items():
        if choice == 'A':
            results[left][0] += 1; results[right][1] += 1
        elif choice == 'B':
            results[right][0] += 1; results[left][1] += 1
        elif choice == 'tie':
            results[left][2] += 1; results[right][2] += 1
    print('\nCandidate vote counts (not a complete ranking):')
    print(f'{"Image":<10}{"Wins":>7}{"Losses":>9}{"Ties":>7}')
    for iid, (win, loss, tie) in sorted(results.items(), key=lambda t:(-t[1][0], t[1][1], t[0])):
        print(f'{iid:<10}{win:>7}{loss:>9}{tie:>7}')
    print('\nNOTE: Pair coverage may be unequal. This table is descriptive only;')
    print('do not treat the win totals as ground-truth NDCG or evidence that one method is best.')


if __name__ == '__main__':
    main()

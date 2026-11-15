"""Reproduce the small manually annotated audit; never print call content."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification import classify_calls
from src.data_processing import preprocess_calls


def main():
    labels = json.loads(Path('tests/fixtures/manual_labels.json').read_text(encoding='utf-8'))
    df, error = preprocess_calls('data/calls_week_anon.xlsx')
    if error:
        raise RuntimeError(error)
    results = classify_calls(df.loc[[item['row_index'] for item in labels]])
    report = {'sample_size': len(labels), 'annotation': 'Single model reviewer, manual reading; not independent human validation',
              'usage': 'Development sample used to refine rules; not a held-out accuracy estimate',
              'selection': '16 user-containing transcripts, random_state=42; 8 meeting-keyword transcripts, random_state=7',
              'metrics': {}}
    for field in ['has_relevance_signal', 'next_step_proposed', 'next_step_accepted']:
        tp = fp = fn = tn = 0
        errors = []
        for label in labels:
            row = results.loc[label['row_index']]
            digest = hashlib.sha256(df.loc[label['row_index'], 'raw_transcript'].encode()).hexdigest()
            if digest != label['sha256']:
                raise ValueError('Dataset changed at row ' + str(label['row_index']))
            actual, expected = bool(row[field]), label[field]
            tp += actual and expected
            fp += actual and not expected
            fn += not actual and expected
            tn += not actual and not expected
            if actual != expected:
                errors.append(label['row_index'])
        report['metrics'][field] = dict(tp=tp, fp=fp, fn=fn, tn=tn,
            precision=round(tp/(tp+fp), 4) if tp+fp else None,
            recall=round(tp/(tp+fn), 4) if tp+fn else None, error_rows=errors)
    Path('docs/classification_evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

"""ERRANT (primary) and NUS M2 scores for a list of corrections.

ERRANT uses compare() from greco_setup/evaluate_errant.py. The bundled M2 scorer
is Python 2; like greco_setup/evaluate_pilot.py, temporary copies are converted
with lib2to3 (Python <= 3.12) and the original scripts stay untouched.
"""

from pathlib import Path
import re
import subprocess
import sys
import tempfile

from greco_setup.evaluate_errant import compare
from training.common import ROOT, write_lines


def m2_scorer(directory):
    """Write a Python 3 copy of m2scorer/scripts into directory; return the script path."""
    from lib2to3.refactor import RefactoringTool, get_fixers_from_package
    converter = RefactoringTool(get_fixers_from_package('lib2to3.fixes'))
    for name in ('util.py', 'levenshtein.py', 'm2scorer.py'):
        text = str(converter.refactor_string((ROOT / 'm2scorer/scripts' / name).read_text(), name))
        text = text.replace("puffer.decode('utf8')", 'puffer').replace('line.decode("utf8")', 'line')
        Path(directory, name).write_text(text)
    return Path(directory, 'm2scorer.py')


def score(src_path, hyps, gold_m2, m2=True):
    """{'errant': {'correction': {...}, 'detection': {...}}, 'm2': {precision, recall, f0_5}}"""
    with tempfile.TemporaryDirectory() as tmp:
        result = {'errant': compare(str(src_path), hyps, str(gold_m2), tmp)}
        if m2:
            pred = Path(tmp, 'm2.pred')
            write_lines(pred, hyps)
            output = subprocess.check_output([sys.executable, str(m2_scorer(tmp)), str(pred), str(gold_m2)], text=True)
            values = [float(v) for v in re.findall(r':\s*([0-9.]+)', output)]
            if len(values) != 3:
                raise ValueError('unexpected M2 scorer output: ' + output)
            result['m2'] = dict(zip(['precision', 'recall', 'f0_5'], values))
    return result


def m2_subset(gold_m2, n, out):
    """Write the first n blocks of an M2 file to out."""
    blocks = Path(gold_m2).read_text(encoding='utf-8').strip().split('\n\n')[:n]
    Path(out).write_text('\n\n'.join(blocks) + '\n\n', encoding='utf-8')
    return out

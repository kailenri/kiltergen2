import json
import pytest


def test_plot_cycle_saves_tmp(tmp_path):
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    from viz import plot_sequence_cycle

    with open('sample_climbs.json') as f:
        data = json.load(f)
    climb = data['results'][0]
    holds = climb['best_sequence']['holds']
    seq = climb['best_sequence']['sequence']
    out = tmp_path / 'out.png'
    plot_sequence_cycle(holds, seq, title='test', output_path=str(out), show=False)
    assert out.exists()

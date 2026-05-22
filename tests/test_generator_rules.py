import json
import math

from sequence_generator import ClimbSequenceGenerator
from config import MAX_FOOT_REACH


def load_sample():
    with open('sample_climbs.json') as f:
        data = json.load(f)
    return data['results'][0]


def test_hand_moves_on_hand_holds():
    climb = load_sample()
    holds = climb['best_sequence']['holds']
    gen = ClimbSequenceGenerator(holds)
    res = gen.generate_sequences(beam_width=3)
    assert res.get('status') == 'success'
    for seqobj in res.get('all_sequences', []):
        seq = seqobj['sequence']
        for m in seq:
            if m['limb'] in ('RH', 'LH'):
                role = gen.hold_dict.get(m['hold'], {}).get('role_id')
                assert role in {12, 13, 14}


def test_finish_on_final_move():
    climb = load_sample()
    holds = climb['best_sequence']['holds']
    gen = ClimbSequenceGenerator(holds)
    res = gen.generate_sequences(beam_width=3)
    assert res.get('status') == 'success'
    for seqobj in res.get('all_sequences', []):
        seq = seqobj['sequence']
        last = seq[-1]
        role = gen.hold_dict.get(last['hold'], {}).get('role_id')
        assert role == 14


def test_foot_reach_distance():
    """Each foot transition must respect MAX_FOOT_REACH from the foot's prior position."""
    climb = load_sample()
    holds = climb['best_sequence']['holds']
    gen = ClimbSequenceGenerator(holds)
    res = gen.generate_sequences(beam_width=3)
    assert res.get('status') == 'success'
    for seqobj in res.get('all_sequences', []):
        seq = seqobj['sequence']
        foot_pos = {'RF': None, 'LF': None}
        for m in seq:
            if m['limb'] not in ('RF', 'LF'):
                continue
            hold = gen.hold_dict.get(m['hold'])
            assert hold is not None
            new_xy = (hold['x'], hold['y'])
            prev = foot_pos[m['limb']]
            if prev is not None:
                dist = math.hypot(new_xy[0] - prev[0], new_xy[1] - prev[1])
                assert dist <= MAX_FOOT_REACH + 1e-6, (
                    f"foot move {m['limb']} {prev}->{new_xy} dist={dist} > {MAX_FOOT_REACH}"
                )
            foot_pos[m['limb']] = new_xy


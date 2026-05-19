import json
from sequence_generator import ClimbSequenceGenerator


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

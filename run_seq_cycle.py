import json
from sequence_generator import ClimbSequenceGenerator
from viz import plot_sequence_cycle

DATA_FILE = 'sample_climbs.json'
CLIMB_ID = '006f95cbc91545afa69ff5ff4a737f45'

with open(DATA_FILE) as f:
    data = json.load(f)

climb = None
for r in data.get('results', []):
    if r.get('id') == CLIMB_ID:
        climb = r
        break

if not climb:
    raise SystemExit('Climb not found')

holds = climb.get('best_sequence', {}).get('holds', [])

print(f'Using {len(holds)} holds')

# instantiate generator
gen = ClimbSequenceGenerator(holds)

print('Generating sequences...')
res = gen.generate_sequences(beam_width=6)

if res.get('status') != 'success':
    print('No sequences generated:', res)
    raise SystemExit(1)

best = res['best_sequence']['sequence']
print('Best sequence length:', len(best))

# The generator sequence entries are dicts with 'hold' (hole_id) and 'limb'
plot_sequence_cycle(holds, best, title=climb.get('name','Climb'), show=True)

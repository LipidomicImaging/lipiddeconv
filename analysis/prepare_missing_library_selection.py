"""Prepare outcome-blind omission IDs from existing V57/V51 design assets only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def prepare(root):
    parent = root / 'results/v57_spectral_spatial_identity_confidence_benchmark'
    geometry = root / 'results/v51_competition_structure'
    design = read(parent / 'design.json')
    scientific = design['scientific']
    canonical = json.dumps(scientific,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    assert hashlib.sha256(canonical.encode()).hexdigest() == design['design_fingerprint']
    assert sha(geometry/'candidate_geometry.csv') == scientific['v51_stage0_hashes']['v51_candidate_geometry']
    pair_sources = [v for k,v in scientific['source_hashes'].items() if k.replace('\\','/').endswith('/v51_competition_structure/pair_geometry.csv')]
    assert len(pair_sources) == 1 and sha(geometry/'pair_geometry.csv') == pair_sources[0]
    units = [u for u in scientific['matched_units'] if int(u['block']) <= 5]
    assert len(units) == 125
    truth = {int(u['HOLD_index']):u for u in units}
    names = {u['HOLD_lipid_name'] for u in units}
    candidates = {int(r['candidate_index']):r for r in rows(geometry/'candidate_geometry.csv')}
    assert len(candidates) == 391 and set(candidates) == set(range(391))
    abundance = {int(r['unit_id']):float(r['abundance_multiplier']) for r in scientific['abundance_multipliers']}
    neighbors = {i:[] for i in truth}
    for r in rows(geometry/'pair_geometry.csv'):
        i,j = int(r['i']),int(r['j'])
        for a,b in ((i,j),(j,i)):
            if a in truth and candidates[b]['lipid_name'] not in names:
                neighbors[a].append((float(r['fragment_cosine']),candidates[b]['lipid_name'],candidates[b]['candidate_id'],b,float(r['full_cosine'])))
    ranked = []
    for i,u in truth.items():
        assert candidates[i]['lipid_name'] == u['HOLD_lipid_name']
        assert sum(c['lipid_name'] == u['HOLD_lipid_name'] for c in candidates.values()) == 1
        best = sorted(neighbors[i],key=lambda n:(-n[0],n[1],n[2]))[0]
        ranked.append(dict(candidate_index=i,candidate_id=candidates[i]['candidate_id'],lipid_name=u['HOLD_lipid_name'],
                           lipid_class=candidates[i]['lipid_class'],unit_id=int(u['pair_id']),
                           abundance_multiplier=abundance[int(u['pair_id'])],
                           max_nontruth_fragment_cosine=best[0],neighbor_lipid_name=best[1],neighbor_candidate_id=best[2],
                           neighbor_candidate_index=best[3],neighbor_full_cosine=best[4],
                           cone_isolation=float(candidates[i]['cone_isolation'])))
    hard = sorted(ranked,key=lambda r:(-r['max_nontruth_fragment_cosine'],r['lipid_name']))[:5]
    used = {r['candidate_index'] for r in hard}
    easy = []
    for h in hard:
        eligible = [r for r in ranked if r['candidate_index'] not in used and r['abundance_multiplier'] == h['abundance_multiplier']]
        assert eligible, 'NO_EXACT_ABUNDANCE_MATCH'
        choice = min(eligible,key=lambda r:(r['lipid_class'] != h['lipid_class'],r['max_nontruth_fragment_cosine'],r['lipid_name']))
        easy.append({**choice,'matched_hard_identity':h['lipid_name'],'same_class':choice['lipid_class']==h['lipid_class']})
        used.add(choice['candidate_index'])
    arms = {}
    for label,selection in [('close_neighbor',hard),('relatively_isolated',easy)]:
        omitted_names = {r['lipid_name'] for r in selection}
        removed = [i for i,c in candidates.items() if c['lipid_name'] in omitted_names]
        kept = sorted(set(candidates)-set(removed))
        assert len(removed)==5 and all(r['neighbor_candidate_index'] in kept for r in selection)
        arms[label] = dict(omitted=selection,removed_original_indices=sorted(removed),reduced_to_original=kept,N_solver=len(kept))
    manifest = dict(status='SELECTION_PREPARED_NOT_EXECUTION_FROZEN',parent_fingerprint=design['design_fingerprint'],
                    datasets=[f'HOLD_R{r}_K125' for r in range(1,4)],K=125,omitted_truth_count=5,
                    planned_new_fits_if_controls_verified=6,arms=arms,
                    source_hashes={str(p.relative_to(root)):sha(p) for p in [parent/'design.json',geometry/'candidate_geometry.csv',geometry/'pair_geometry.csv']},
                    implementation_sha256=sha(Path(__file__)),
                    selection_uses_learned_outcomes=False,
                    matching_order='hard descending cosine/name; control same abundance, prefer same class, ascending cosine/name; greedy without replacement',
                    pending=['B/X_true reconstruction and byte identity','full control provenance and current production implementation equivalence','isolated solver adapter and validation'])
    raw=json.dumps(manifest,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    manifest['selection_fingerprint']=hashlib.sha256(raw.encode()).hexdigest()
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=prepare(args.root.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if args.output.exists():
        assert read(args.output)==result,'EXISTING_SELECTION_CHANGED'
    else:
        args.output.write_text(encoded,encoding='utf-8')
    print(json.dumps({'status':result['status'],'selection_fingerprint':result['selection_fingerprint']}))

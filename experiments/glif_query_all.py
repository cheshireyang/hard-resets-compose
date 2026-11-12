"""List every Allen Cell Types specimen that has official GLIF models (templates 1-5), with model ids and explained
variance; writes results/allen_glif_models_raw_all.json.  usage: python3 experiments/glif_query_all.py"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allen_fetch import query
tmpl = json.load(open("results/allen_glif_template_ids.json"))
rows, start = [], 0
while True:
    page = query("model::NeuronalModel,rma::criteria,[neuronal_model_template_id$in%s],rma::include,neuronal_model_template,"
                 "well_known_files,neuronal_model_runs,specimen(donor),rma::options[num_rows$eq2000][start_row$eq%d]"
                 % (",".join(map(str, tmpl)), start))
    rows += page; start += len(page)
    print(start, flush=True)
    if len(page) < 2000: break
json.dump(rows, open("results/allen_glif_models_raw_all.json", "w"))
sp = {}
for r in rows:
    d = ((r.get("specimen") or {}).get("donor") or {})
    sp.setdefault(r["specimen_id"], d.get("organism_id"))
print(len(rows), "models,", len(sp), "specimens; organisms:", {o: sum(1 for v in sp.values() if v == o) for o in set(sp.values())})

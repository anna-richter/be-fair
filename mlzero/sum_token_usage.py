import json
from pathlib import Path

MLZERO_DIR = Path("/Users/arichter/Documents/GitHub/be-fair/mlzero")

totals = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_tokens": 0,
}

per_run = []
missing = []

for entry in sorted(MLZERO_DIR.iterdir()):
    if not entry.is_dir():
        continue
    if not entry.name[:1].isdigit():
        continue

    token_file = entry / "token_usage.json"
    if not token_file.exists():
        missing.append(entry.name)
        continue

    with token_file.open() as f:
        data = json.load(f)

    run_total = data.get("total", {})
    per_run.append((entry.name, run_total))
    for key in totals:
        totals[key] += run_total.get(key, 0)

print(f"{'run':<25} {'input':>15} {'output':>12} {'total':>15}")
print("-" * 70)
for name, t in per_run:
    print(
        f"{name:<25} "
        f"{t.get('total_input_tokens', 0):>15,} "
        f"{t.get('total_output_tokens', 0):>12,} "
        f"{t.get('total_tokens', 0):>15,}"
    )
print("-" * 70)
print(
    f"{'SUM':<25} "
    f"{totals['total_input_tokens']:>15,} "
    f"{totals['total_output_tokens']:>12,} "
    f"{totals['total_tokens']:>15,}"
)
print(f"\nProcessed {len(per_run)} runs.")
if missing:
    print(f"Missing token_usage.json in: {missing}")


"""
run                                 input       output           total
----------------------------------------------------------------------
28-basic_prompt                 2,751,229       48,298       2,799,527
29-basic_prompt                 2,251,625       44,004       2,295,629
30-basic_prompt                 2,869,282       50,504       2,919,786
31-basic_prompt                 3,244,130       61,217       3,305,347
32-basic_prompt                 3,194,228       59,133       3,253,361
33-basic_prompt                 3,239,965       64,382       3,304,347
34-basic_prompt                 2,974,715       52,582       3,027,297
35-addition_1                   2,804,476       49,582       2,854,058
36-addition_1                   3,325,756       66,469       3,392,225
37-addition_1                   2,818,133       47,293       2,865,426
38-addition_1                   3,148,167       58,758       3,206,925
39-addition_1                   3,150,439       55,448       3,205,887
40-addition_1                   2,911,603       58,844       2,970,447
41-addition_1                   3,046,961       56,136       3,103,097
42-addition_2                   3,019,969       54,803       3,074,772
43-addition_2                   3,003,100       51,760       3,054,860
44-addition_2                   2,711,737       50,333       2,762,070
45-addition_2                   3,391,038       58,204       3,449,242
46-addition_2                   3,252,434       61,599       3,314,033
47-addition_2                   3,066,564       49,694       3,116,258
48-addition_2                   2,974,815       55,592       3,030,407
49-addition_3                   2,935,126       64,979       3,000,105
50-addition_3                   3,285,168       57,291       3,342,459
51-addition_3                   2,575,320       46,827       2,622,147
52-addition_3                   3,261,019       57,661       3,318,680
53-addition_3                   3,264,659       59,901       3,324,560
54-addition_3                   2,671,647       50,577       2,722,224
55-addition_3                   3,280,819       68,207       3,349,026
----------------------------------------------------------------------
SUM                            84,424,124    1,560,078      85,984,202
"""
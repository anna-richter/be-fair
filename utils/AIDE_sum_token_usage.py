import re
from collections import defaultdict
from pathlib import Path

AIDE_LOGS_DIR = Path("/Users/arichter/Documents/GitHub/be-fair/aide/logs")

PATTERN = re.compile(
    r"OpenAI API call completed - (?P<model>\S+) - [\d.]+s - "
    r"(?P<total>\d+) tokens \(in: (?P<in_tok>\d+), out: (?P<out_tok>\d+)\)"
)

grand_totals = {"total_input_tokens": 0, "total_output_tokens": 0, "total_tokens": 0}
per_run = []
per_model = defaultdict(lambda: {"in": 0, "out": 0, "total": 0, "calls": 0})
missing = []

for entry in sorted(AIDE_LOGS_DIR.iterdir()):
    if not entry.is_dir() or not entry.name[:1].isdigit():
        continue

    log_file = entry / "aide.log"
    if not log_file.exists():
        missing.append(entry.name)
        continue

    run_total = {"in": 0, "out": 0, "total": 0, "calls": 0}

    with log_file.open(errors="replace") as f:
        for line in f:
            m = PATTERN.search(line)
            if not m:
                continue
            in_tok = int(m["in_tok"])
            out_tok = int(m["out_tok"])
            total = int(m["total"])
            model = m["model"]

            run_total["in"] += in_tok
            run_total["out"] += out_tok
            run_total["total"] += total
            run_total["calls"] += 1

            per_model[model]["in"] += in_tok
            per_model[model]["out"] += out_tok
            per_model[model]["total"] += total
            per_model[model]["calls"] += 1

    per_run.append((entry.name, run_total))
    grand_totals["total_input_tokens"] += run_total["in"]
    grand_totals["total_output_tokens"] += run_total["out"]
    grand_totals["total_tokens"] += run_total["total"]

print(f"{'run':<22} {'calls':>6} {'input':>15} {'output':>15} {'total':>15}")
print("-" * 78)
for name, t in per_run:
    print(f"{name:<22} {t['calls']:>6} {t['in']:>15,} {t['out']:>15,} {t['total']:>15,}")
print("-" * 78)
print(
    f"{'SUM':<22} {sum(t['calls'] for _, t in per_run):>6} "
    f"{grand_totals['total_input_tokens']:>15,} "
    f"{grand_totals['total_output_tokens']:>15,} "
    f"{grand_totals['total_tokens']:>15,}"
)

print("\nPer-model breakdown:")
for model, t in sorted(per_model.items()):
    print(
        f"  {model:<30} calls={t['calls']:>5}  "
        f"in={t['in']:>12,}  out={t['out']:>12,}  total={t['total']:>12,}"
    )

print(f"\nProcessed {len(per_run)} runs.")
if missing:
    print(f"Missing aide.log in: {missing}")



"""

run                     calls           input          output           total
------------------------------------------------------------------------------
0-basic_prompt             41         138,539          62,842         201,381
1-basic_prompt             41         150,465          65,261         215,726
10-addition_1              42         173,712          70,051         243,763
11-addition_1              41         181,485          68,274         249,759
12-addition_1              41         144,328          65,299         209,627
13-addition_1              41         129,946          61,361         191,307
14-addition_2              41         159,705          65,888         225,593
15-addition_2              41         150,705          72,505         223,210
16-addition_2              41         149,188          70,211         219,399
17-addition_2              41         160,889          71,548         232,437
18-addition_2              41         161,008          65,457         226,465
19-addition_2              41         158,728          74,047         232,775
2-basic_prompt             41         141,949          66,666         208,615
20-addition_2              41         134,909          65,536         200,445
21-addition_3              41         156,481          67,017         223,498
22-addition_3              41         146,159          74,725         220,884
23-addition_3              41         142,818          72,201         215,019
24-addition_3              41         158,399          69,148         227,547
25-addition_3              41         158,204          68,879         227,083
26-addition_3              41         167,771          75,208         242,979
27-addition_3              42         153,298          66,403         219,701
3-basic_prompt             41         133,274          65,157         198,431
4-basic_prompt             41         148,448          68,815         217,263
5-basic_prompt             41         131,381          63,165         194,546
6-basic_prompt             41         182,196          64,755         246,951
7-addition_1               41         141,740          63,256         204,996
8-addition_1               41         153,085          65,016         218,101
9-addition_1               41         150,301          63,313         213,614
------------------------------------------------------------------------------
SUM                      1150       4,259,111       1,892,004       6,151,115

Per-model breakdown:
  gpt-4.1-2025-04-14             calls=   28  in=     920,323  out=      45,174  total=     965,497
  gpt-4.1-mini-2025-04-14        calls=  560  in=   1,353,480  out=      57,405  total=   1,410,885
  o4-mini-2025-04-16             calls=  562  in=   1,985,308  out=   1,789,425  total=   3,774,733

Processed 28 runs.

"""
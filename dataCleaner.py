
import re

input_file = "dataset_100M_clean.txt"
output_file = "dataset_100M_clean2.txt"

seen = set()

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:

        line = line.strip()

        if len(line) < 30:
            continue

        # remove star separators
        if re.fullmatch(r'[\*\s]{3,}', line):
            continue

        # remove bracket formatting
        if re.fullmatch(r'\[_.*_\]', line):
            continue

        # remove duplicates
        if line in seen:
            continue

        seen.add(line)

        fout.write(line + "\n")

print("Cleaning finished.")
                        
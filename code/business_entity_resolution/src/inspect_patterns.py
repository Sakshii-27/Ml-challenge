import csv

def inspect_patterns():
    # Load 20 matches from ground truth
    matches_sample = []
    with open("dataset/train/train_ground_truth.tsv", 'r', encoding='utf-8') as f:
        f.readline()
        for i, line in enumerate(f):
            parts = line.rstrip('\n').split('\t')
            s1_id = parts[0]
            if len(parts) > 1 and parts[1].strip():
                ms = parts[1].split(',')
                matches_sample.append((s1_id, ms))
            if len(matches_sample) >= 30:
                break

    needed_ids = set()
    s1_ids = set()
    for s1_id, ms in matches_sample:
        s1_ids.add(s1_id)
        for m in ms:
            needed_ids.add(m)

    s1_records = {}
    with open("dataset/train/train_source1.tsv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            if row[0] in s1_ids:
                s1_records[row[0]] = row

    other_records = {}
    for src_file in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        with open(src_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            for row in reader:
                if row[0] in needed_ids:
                    other_records[row[0]] = row
                    
    print("=== SAMPLE GROUND TRUTH MATCHES INSPECTION ===\n")
    for s1_id, ms in matches_sample[:10]:
        s1_row = s1_records.get(s1_id)
        if not s1_row: continue
        print(f"==================================================")
        print(f"SOURCE 1 [{s1_row[3]}]: ID={s1_id}")
        print(f"  Name:    {s1_row[1]}")
        print(f"  Address: {s1_row[2]}")
        print(f"MATCHES ({len(ms)}):")
        for m in ms:
            m_row = other_records.get(m)
            if m_row:
                print(f"  --> [{m_row[0]} ({m_row[3]})]")
                print(f"      Name:    {m_row[1]}")
                print(f"      Address: {m_row[2]}")
            else:
                print(f"  --> [{m}] (not found in early sample)")
        print()

if __name__ == "__main__":
    inspect_patterns()

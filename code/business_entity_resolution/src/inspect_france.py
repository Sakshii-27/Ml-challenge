"""
Samples France test records from S1 and retrieves their candidate matches from S2 and S3.
Prints raw name & address side-by-side with generated blocking keys and candidate matches
to manually inspect France normalization, diacritic stripping, legal suffix removal, and address keys.
"""
import csv
import collections
from blocking_keys import extract_name_keys, extract_address_keys

def inspect_france_samples(num_samples=35):
    print("=== INSPECTING FRANCE TEST RECORDS AND CANDIDATE KEYS ===")
    
    # 1. Read S1 France sample
    s1_france = []
    with open("dataset/test/test_source1.tsv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row[3] == "France":
                s1_france.append(row)
                if len(s1_france) >= num_samples:
                    break

    # 2. Build index of a slice of S2 and S3 France records
    s23_france = {}
    france_inverted_index = collections.defaultdict(list)
    
    for filename in ["dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv"]:
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for i, row in enumerate(reader):
                if row[3] == "France":
                    cid = row[0]
                    s23_france[cid] = row
                    keys = set(extract_name_keys(row[1]) + extract_address_keys(row[2]))
                    for k in keys:
                        if len(france_inverted_index[k]) < 500:
                            france_inverted_index[k].append(cid)
                if len(s23_france) >= 150000:
                    break

    print(f"Loaded {len(s1_france)} France S1 samples.")
    print(f"Loaded {len(s23_france)} France S2/S3 records into test index.\n")

    for i, s1_row in enumerate(s1_france[:25], 1):
        s1_id = s1_row[0]
        s1_name = s1_row[1]
        s1_addr = s1_row[2]
        
        n_keys = extract_name_keys(s1_name)
        a_keys = extract_address_keys(s1_addr)
        all_keys = set(n_keys + a_keys)
        
        hits = collections.Counter()
        for k in all_keys:
            for cid in france_inverted_index.get(k, []):
                hits[cid] += 1
                
        top_matches = hits.most_common(3)
        
        print(f"--------------------------------------------------------------------------------")
        print(f"SAMPLE #{i} [France S1 ID: {s1_id}]")
        print(f"  Raw Name:      {s1_name}")
        print(f"  Raw Address:   {s1_addr}")
        print(f"  Name Keys:     {n_keys}")
        print(f"  Address Keys:  {a_keys}")
        print(f"  Retrieved Top Candidates from S2/S3 ({len(hits)} total hits):")
        if not top_matches:
            print("    (No candidate match found in 150k test slice)")
        for cid, score in top_matches:
            c_row = s23_france[cid]
            c_n_keys = extract_name_keys(c_row[1])
            c_a_keys = extract_address_keys(c_row[2])
            overlap = all_keys.intersection(set(c_n_keys + c_a_keys))
            print(f"    -> [{cid}] (Shared Keys: {score}) Overlap: {list(overlap)}")
            print(f"       Cand Name:    {c_row[1]}")
            print(f"       Cand Address: {c_row[2]}")
        print()

if __name__ == "__main__":
    inspect_france_samples()

"""
Analyzes missed pairs from blocking to identify failure modes:
1. Missing address vs missing name
2. Extreme transliteration mismatch
3. Typos in single-word names
4. Missing blocking key collisions
"""
import csv
import collections
from blocking_harness import create_val_split, load_ground_truth
from blocking_keys import extract_name_keys, extract_address_keys

def diagnose_misses(sample_size=30):
    val_ids = create_val_split(val_s1_size=5000)
    ground_truth = load_ground_truth(val_ids)
    
    # Load sample S1 records
    s1_dict = {}
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row[0] in val_ids:
                s1_dict[row[0]] = row
                
    # Load candidate predictions from our inverted index logic for a small subset
    # To quickly inspect, let's find pairs where neither name keys nor address keys overlap
    print("Finding true matches that share 0 keys with S1...")
    
    # Let's inspect 50 missed pairs by streaming S2/S3
    needed_s23 = set()
    s1_to_check = list(val_ids)[:500]
    for s1 in s1_to_check:
        for m in ground_truth.get(s1, []):
            needed_s23.add(m)
            
    s23_dict = {}
    for fname in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        with open(fname, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                if row[0] in needed_s23:
                    s23_dict[row[0]] = row

    missed_count = 0
    print(f"\nAnalyzing true pairs for {len(s1_to_check)} S1 entities...")
    for s1_id in s1_to_check:
        s1_row = s1_dict[s1_id]
        s1_name_k = set(extract_name_keys(s1_row[1]))
        s1_addr_k = set(extract_address_keys(s1_row[2]))
        s1_all_k = s1_name_k.union(s1_addr_k)
        
        true_ms = ground_truth.get(s1_id, set())
        for m in true_ms:
            m_row = s23_dict.get(m)
            if not m_row:
                continue
            m_name_k = set(extract_name_keys(m_row[1]))
            m_addr_k = set(extract_address_keys(m_row[2]))
            m_all_k = m_name_k.union(m_addr_k)
            
            overlap = s1_all_k.intersection(m_all_k)
            if not overlap:
                missed_count += 1
                if missed_count <= 10:
                    print(f"\n--- MISSED MATCH #{missed_count} [{s1_row[3]}] ---")
                    print(f"S1 ({s1_id}):")
                    print(f"  Name:    {s1_row[1]}")
                    print(f"  Address: {s1_row[2]}")
                    print(f"  Keys:    {sorted(s1_all_k)}")
                    print(f"Match ({m}):")
                    print(f"  Name:    {m_row[1]}")
                    print(f"  Address: {m_row[2]}")
                    print(f"  Keys:    {sorted(m_all_k)}")

    print(f"\nTotal missed pairs with 0 key overlap: {missed_count}")

if __name__ == "__main__":
    diagnose_misses()

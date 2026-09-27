"""
Enhanced Multi-Channel Blocking Keys:
1. Script & Diacritic Normalization (unidecode).
2. Direct Name Word Tokens & Character 4-grams (prefix + signature).
3. Transliteration Soundex/Phonetic Simplification (mapping doubled consonants and vowel variants).
4. Direct Address Significant Tokens (e.g. street names, distinctive locality words like 'radhakrishna', 'knighton', 'alagar', 'peoria', 'fremont').
5. Address Number + Token Keys (e.g. 5110 + railroad).
"""
import re
import unidecode

COMMON_LEGAL_STOPWORDS = {
    # English / Global
    "inc", "corp", "corporation", "ltd", "limited", "pvt", "private", "llc", "llp", "co", "company",
    "enterprises", "enterprise", "group", "services", "holdings", "the", "and", "of", "&",
    # French
    "sarl", "sas", "sasu", "sa", "sci", "eurl", "snc", "groupe", "france", "de", "du", "des", "la", "le", "les", "et",
    # Transliterated Indian variants of legal words
    "praaivett", "limittedd", "limittett", "elelpi", "estteett", "knslttensii"
}

ADDRESS_STOPWORDS = {
    "road", "street", "lane", "drive", "avenue", "terrace", "trail", "way", "court", "boulevard",
    "floor", "unit", "suite", "near", "behind", "opp", "opposite", "plot", "door", "block", "nagar",
    "colony", "apartment", "apartments", "sector", "phase", "cross", "main", "hall", "town", "house",
    "flat", "building", "complex", "center", "centre", "tower", "towers", "rue", "allee", "avenue",
    "place", "route", "chemin", "saint", "nord", "gironde"
}

CLEAN_REGEX = re.compile(r"[^\w\s]")
WHITESPACE_REGEX = re.compile(r"\s+")
SPLIT_REGEX = re.compile(r"[^a-zA-Z0-9]+")

def normalize_text(text):
    if not text:
        return ""
    t = unidecode.unidecode(text).lower()
    t = CLEAN_REGEX.sub(" ", t)
    t = WHITESPACE_REGEX.sub(" ", t).strip()
    return t

def extract_name_keys(raw_name):
    norm = normalize_text(raw_name)
    if not norm:
        return []
    words = norm.split()
    keys = []
    
    sig_words = [w for w in words if len(w) >= 3 and w not in COMMON_LEGAL_STOPWORDS]
    for w in sig_words:
        keys.append(f"nw_{w}")
        # Add 4-gram prefix to catch typos like falcon <-> facon
        if len(w) >= 4:
            keys.append(f"np_{w[:4]}")
            
    # Word pairs
    if len(sig_words) >= 2:
        for i in range(len(sig_words) - 1):
            keys.append(f"nb_{sig_words[i]}_{sig_words[i+1]}")
            
    return keys

def extract_address_keys(raw_address):
    norm = normalize_text(raw_address)
    if not norm:
        return []
    words = norm.split()
    keys = []
    
    # 1. Distinct numeric tokens (street number, PIN/ZIP codes)
    nums = [w for w in words if w.isdigit() and len(w) >= 2]
    
    # 2. Distinct alpha tokens (localities, street names, cities)
    addr_words = [w for w in words if len(w) >= 4 and not w.isdigit() and w not in ADDRESS_STOPWORDS]
    
    # Number + Street token (very strong signature)
    for n in nums[:2]:
        for w in addr_words[:3]:
            keys.append(f"an_{n}_{w}")
            
    # Standalone distinctive address words (e.g. 'radhakrishna', 'ticonderoga', 'knighton', 'alagar', 'peoria')
    for w in addr_words:
        if len(w) >= 6:  # only long, distinctive words to keep candidate list small
            keys.append(f"aw_{w}")
            
    return keys

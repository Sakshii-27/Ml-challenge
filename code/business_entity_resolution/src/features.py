"""
Pairwise Feature Extraction Engine for Entity Resolution.
Extracts high-signal, robust string similarity, token overlap, numeric alignment,
and blocking signals for candidate pairs (S1, S2/S3).
Designed for fast vectorization with rapidfuzz and zero-dependency string utilities.
"""
import re
import math
import unidecode
from rapidfuzz.distance import Levenshtein, JaroWinkler
from rapidfuzz import fuzz

COMMON_LEGAL_STOPWORDS = {
    "inc", "corp", "corporation", "ltd", "limited", "pvt", "private", "llc", "llp", "co", "company",
    "enterprises", "enterprise", "group", "services", "holdings", "the", "and", "of", "&",
    "sarl", "sas", "sasu", "sa", "sci", "eurl", "snc", "groupe", "france", "de", "du", "des", "la", "le", "les", "et",
    "praaivett", "limittedd", "limittett", "elelpi", "estteett", "knslttensii"
}

DIGIT_REGEX = re.compile(r"\b\d+\b")
TOKEN_REGEX = re.compile(r"[a-z0-9]+")

def clean_and_normalize(text):
    if not text:
        return ""
    t = unidecode.unidecode(text).lower()
    return " ".join(TOKEN_REGEX.findall(t))

def extract_tokens(clean_str):
    return set(clean_str.split())

def extract_significant_tokens(tokens):
    return {t for t in tokens if len(t) >= 3 and t not in COMMON_LEGAL_STOPWORDS}

def extract_pairwise_features(s1_name_raw, s1_addr_raw, s1_country,
                              cand_id, cand_name_raw, cand_addr_raw, cand_country,
                              shared_key_count=1):
    """
    Computes a 1D numerical feature vector for the pair (s1, cand).
    """
    # 1. Cleaned and normalized text
    s1_name = clean_and_normalize(s1_name_raw)
    cand_name = clean_and_normalize(cand_name_raw)
    
    s1_addr = clean_and_normalize(s1_addr_raw)
    cand_addr = clean_and_normalize(cand_addr_raw)
    
    # 2. Name Features
    jw_name = JaroWinkler.similarity(s1_name, cand_name) if s1_name and cand_name else 0.0
    lev_name = Levenshtein.normalized_similarity(s1_name, cand_name) if s1_name and cand_name else 0.0
    token_sort_name = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
    token_set_name = fuzz.token_set_ratio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
    
    # Name tokens & significant brand words
    s1_ntokens = extract_tokens(s1_name)
    c_ntokens = extract_tokens(cand_name)
    n_common_tokens = len(s1_ntokens.intersection(c_ntokens))
    jaccard_name = (n_common_tokens / len(s1_ntokens.union(c_ntokens))) if (s1_ntokens or c_ntokens) else 0.0
    
    s1_sig_ntokens = extract_significant_tokens(s1_ntokens)
    c_sig_ntokens = extract_significant_tokens(c_ntokens)
    sig_common_tokens = len(s1_sig_ntokens.intersection(c_sig_ntokens))
    sig_jaccard_name = (sig_common_tokens / len(s1_sig_ntokens.union(c_sig_ntokens))) if (s1_sig_ntokens or c_sig_ntokens) else 0.0
    
    # First token match (primary company brand / keyword)
    s1_first_token = s1_name.split()[0] if s1_name else ""
    c_first_token = cand_name.split()[0] if cand_name else ""
    first_token_match = 1.0 if (s1_first_token and s1_first_token == c_first_token) else 0.0
    first_token_jw = JaroWinkler.similarity(s1_first_token, c_first_token) if (s1_first_token and c_first_token) else 0.0
    
    # Length diff
    len_diff_name = abs(len(s1_name) - len(cand_name))
    len_ratio_name = (min(len(s1_name), len(cand_name)) / max(len(s1_name), len(cand_name))) if (s1_name and cand_name) else 0.0
    
    # 3. Address Features
    addr_missing = 1.0 if not cand_addr.strip() else 0.0
    if addr_missing:
        jw_addr = 0.0
        lev_addr = 0.0
        token_sort_addr = 0.0
        jaccard_addr = 0.0
        digit_overlap = 0.0
        digit_match_exact = 0.0
    else:
        jw_addr = JaroWinkler.similarity(s1_addr, cand_addr)
        lev_addr = Levenshtein.normalized_similarity(s1_addr, cand_addr)
        token_sort_addr = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
        
        s1_atokens = extract_tokens(s1_addr)
        c_atokens = extract_tokens(cand_addr)
        a_common = len(s1_atokens.intersection(c_atokens))
        jaccard_addr = (a_common / len(s1_atokens.union(c_atokens))) if (s1_atokens or c_atokens) else 0.0
        
        # Digit / postal code / street number overlap
        s1_digits = set(DIGIT_REGEX.findall(s1_addr))
        c_digits = set(DIGIT_REGEX.findall(cand_addr))
        d_common = len(s1_digits.intersection(c_digits))
        digit_overlap = float(d_common)
        digit_match_exact = 1.0 if (s1_digits and c_digits and s1_digits == c_digits) else 0.0

    # 4. Source & Blocking metadata features
    is_source2 = 1.0 if cand_id.startswith("S2-") else 0.0
    is_source3 = 1.0 if cand_id.startswith("S3-") else 0.0
    shared_keys = float(shared_key_count)
    log_shared_keys = math.log1p(shared_key_count)

    # 5. Composite interaction terms
    name_addr_geom = math.sqrt(max(0.0, jw_name) * max(0.0, jw_addr)) if not addr_missing else jw_name

    features = [
        jw_name,
        lev_name,
        token_sort_name,
        token_set_name,
        jaccard_name,
        sig_jaccard_name,
        first_token_match,
        first_token_jw,
        len_diff_name,
        len_ratio_name,
        addr_missing,
        jw_addr,
        lev_addr,
        token_sort_addr,
        jaccard_addr,
        digit_overlap,
        digit_match_exact,
        is_source2,
        is_source3,
        shared_keys,
        log_shared_keys,
        name_addr_geom
    ]
    return features

FEATURE_NAMES = [
    "jw_name", "lev_name", "token_sort_name", "token_set_name", "jaccard_name",
    "sig_jaccard_name", "first_token_match", "first_token_jw", "len_diff_name", "len_ratio_name",
    "addr_missing", "jw_addr", "lev_addr", "token_sort_addr", "jaccard_addr",
    "digit_overlap", "digit_match_exact", "is_source2", "is_source3", "shared_keys",
    "log_shared_keys", "name_addr_geom"
]

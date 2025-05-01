import hashlib
import numpy as np
import random
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.special import gammaln
import time
from scipy.stats import norm


class CountMinSketch:
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.table = [[0] * width for _ in range(depth)]
        self.hash_seeds = [random.randint(0, 2**32 - 1) for _ in range(depth)]

    def _hash(self, item, seed):
        return int(hashlib.md5((str(seed) + str(item)).encode("utf-8")).hexdigest(), 16) % self.width

    def add(self, item):
        for i in range(self.depth):
            index = self._hash(item, self.hash_seeds[i])
            self.table[i][index] += 1

    def count(self, item):
        min_count = float("inf")
        for i in range(self.depth):
            index = self._hash(item, self.hash_seeds[i])
            min_count = min(min_count, self.table[i][index])
        return min_count


class FairCountMinColumnBased:
    def __init__(self, width, depth, group_buckets):
        self.width = width
        self.depth = depth
        self.group_buckets = group_buckets
        self.table = [[0] * width for _ in range(depth)]
        self.hash_seeds = [random.randint(0, 2**32 - 1) for _ in range(depth)]

    def _hash(self, item, seed):
        return int(hashlib.md5((str(seed) + str(item)).encode("utf-8")).hexdigest(), 16)

    def add(self, item, group):
        for i in range(self.depth):
            bucket_range = self.group_buckets[i][group]
            bucket = self._hash(item, self.hash_seeds[i]) % (bucket_range[1] - bucket_range[0]) + bucket_range[0]
            self.table[i][bucket] += 1

    def count(self, item, group):
        min_count = float("inf")
        for i in range(self.depth):
            bucket_range = self.group_buckets[i][group]
            bucket = self._hash(item, self.hash_seeds[i]) % (bucket_range[1] - bucket_range[0]) + bucket_range[0]
            min_count = min(min_count, self.table[i][bucket])
        return min_count


class FairCountMinRowBased:
    def __init__(self, width, depth_l, depth_h):
        self.width = width
        self.depth_l = depth_l
        self.depth_h = depth_h
        self.table_l = [[0] * width for _ in range(depth_l)]
        self.table_h = [[0] * width for _ in range(depth_h)]
        self.hash_seeds_l = [random.randint(0, 2**32 - 1) for _ in range(depth_l)]
        self.hash_seeds_h = [random.randint(0, 2**32 - 1) for _ in range(depth_h)]

    def _hash(self, x, seed):
        return int(hashlib.md5((str(seed) + str(x)).encode("utf-8")).hexdigest(), 16) % self.width

    def add(self, item, group):
        if group == "l":
            for i in range(self.depth_l):
                index = self._hash(item, self.hash_seeds_l[i])
                self.table_l[i][index] += 1
        elif group == "h":
            for i in range(self.depth_h):
                index = self._hash(item, self.hash_seeds_h[i])
                self.table_h[i][index] += 1

    def count(self, item, group):
        if group == "l":
            return min(self.table_l[i][self._hash(item, self.hash_seeds_l[i])] for i in range(self.depth_l))
        elif group == "h":
            return min(self.table_h[i][self._hash(item, self.hash_seeds_h[i])] for i in range(self.depth_h))


def calculate_sum(n, w, d):
    p = 1 / w
    q = 1 - p
    i = np.arange(n + 1)
    log_binom = gammaln(n + 1) - gammaln(i + 1) - gammaln(n - i + 1)
    log_pmf = log_binom + i * np.log(p) + (n - i) * np.log(q)
    binom_pmf = np.exp(log_pmf)
    tail_probs = np.cumsum(binom_pmf[::-1])[::-1][1:]
    powered = np.power(tail_probs, d)
    return np.sum(powered)


def find_w_l(n_l, n_h, d, w):
    def f(w_l):
        sum_w_l = calculate_sum(n_l, w_l, d)
        sum_w_h = calculate_sum(n_h, w - w_l, d)
        diff = abs(sum_w_l - sum_w_h)
        return diff

    low, high = 1, w - 1
    while high - low > 3:
        mid1 = low + (high - low) // 3
        mid2 = high - (high - low) // 3
        if f(mid1) < f(mid2):
            high = mid2
        else:
            low = mid1
    min_diff = float("inf")
    best_w_l = None
    for w_l in range(low, high + 1):
        diff = f(w_l)
        if diff < min_diff:
            min_diff = diff
            best_w_l = w_l
    return best_w_l


def find_d_l(n_l, n_h, d, w):
    def f(d_l):
        sum_d_l = calculate_sum(n_l, w, d_l)
        sum_d_h = calculate_sum(n_h, w, d - d_l)
        diff = abs(sum_d_l - sum_d_h)
        return diff

    low, high = 1, d - 1
    while high - low > 3:
        mid1 = low + (high - low) // 3
        mid2 = high - (high - low) // 3
        if f(mid1) < f(mid2):
            high = mid2
        else:
            low = mid1

    min_diff = float("inf")
    best_d_l = None
    for d_l in range(low, high + 1):
        diff = f(d_l)
        if diff < min_diff:
            min_diff = diff
            best_d_l = d_l
    return best_d_l


def calculate_errors(true_counts, sketch, group=None):
    return np.mean(
        [
            true_counts[element] / sketch.count(element, group)
            if group
            else true_counts[element] / sketch.count(element)
            for element in true_counts
        ]
    ), np.sum(
        [
            sketch.count(element, group) - true_counts[element]
            if group
            else sketch.count(element) - true_counts[element]
            for element in true_counts
        ]
    )


def read_census():
    data = pd.read_csv("data/USCensus1990.csv")
    data["group"] = data["iSex"].apply(lambda x: "l" if x == 1 else "h")
    stream = [(f"{row['iYearsch']}{row['dIndustry']}{row['iSex']}", row["group"]) for _, row in data.iterrows()]
    n_l = data[data["group"] == "l"].groupby(["iYearsch", "dIndustry", "iSex"]).ngroups
    n = data.groupby(["iYearsch", "dIndustry", "iSex"]).ngroups
    return stream, n, n_l


def run_fairness_census(w, d):
    stream, n, n_l = read_census()
    n_h = n - n_l
    w_l = find_w_l(n_l, n_h, d, w)
    d_l = find_d_l(n_l, n_h, d, w)

    true_counts_h = defaultdict(int)
    true_counts_l = defaultdict(int)
    N = len(stream)
    N_l = 0
    for element, group in stream:
        (true_counts_l if group == "l" else true_counts_h)[element] += 1
        if group == "l":
            N_l += 1
    N_h = N - N_l
    cms = CountMinSketch(w, d)
    fcm_c = FairCountMinColumnBased(w, d, [{"l": (0, w_l), "h": (w_l, w)} for _ in range(d)])
    fcm_r = FairCountMinRowBased(w, d_l, d - d_l)

    for element, group in stream:
        cms.add(element)
        fcm_c.add(element, group)
        fcm_r.add(element, group)

    mean_approx_factor_cms_l, total_add_error_cms_l = calculate_errors(true_counts_l, cms)
    mean_approx_factor_cms_h, total_add_error_cms_h = calculate_errors(true_counts_h, cms)
    mean_approx_factor_fcm_c_l, total_add_error_fcm_c_l = calculate_errors(true_counts_l, fcm_c, "l")
    mean_approx_factor_fcm_c_h, total_add_error_fcm_c_h = calculate_errors(true_counts_h, fcm_c, "h")
    mean_approx_factor_fcm_r_l, total_add_error_fcm_r_l = calculate_errors(true_counts_l, fcm_r, "l")
    mean_approx_factor_fcm_r_h, total_add_error_fcm_r_h = calculate_errors(true_counts_h, fcm_r, "h")

    print(
        "======================== Census, ",
        f"n={n}, n_l={n_l}, n_h={n_h}, N={N}, N_l={N_l}, N_h={N_h}, "
        f"w={w}, w_l={w_l}, w_h={w - w_l}, d={d}, d_l={d_l}, d_h={d - d_l}",
        "========================",
    )
    print(
        f"CountMinSketch (Group l) - Mean Approximation Factor: {mean_approx_factor_cms_l}, Additive Error: {total_add_error_cms_l}"
    )
    print(
        f"CountMinSketch (Group h) - Mean Approximation Factor: {mean_approx_factor_cms_h}, Additive Error: {total_add_error_cms_h}"
    )
    print(
        f"FairCountMinColumnBased (Group l) - Mean Approximation Factor: {mean_approx_factor_fcm_c_l}, Additive Error: {total_add_error_fcm_c_l}"
    )
    print(
        f"FairCountMinColumnBased (Group h) - Mean Approximation Factor: {mean_approx_factor_fcm_c_h}, Additive Error: {total_add_error_fcm_c_h}"
    )
    print(
        f"FairCountMinRowBased (Group l) - Mean Approximation Factor: {mean_approx_factor_fcm_r_l}, Additive Error: {total_add_error_fcm_r_l}"
    )
    print(
        f"FairCountMinRowBased (Group h) - Mean Approximation Factor: {mean_approx_factor_fcm_r_h}, Additive Error: {total_add_error_fcm_r_h}"
    )
    print("============================================")

    return (
        mean_approx_factor_cms_l,
        total_add_error_cms_l,
        mean_approx_factor_cms_h,
        total_add_error_cms_h,
        mean_approx_factor_fcm_c_l,
        total_add_error_fcm_c_l,
        mean_approx_factor_fcm_c_h,
        total_add_error_fcm_c_h,
        mean_approx_factor_fcm_r_l,
        total_add_error_fcm_r_l,
        mean_approx_factor_fcm_r_h,
        total_add_error_fcm_r_h,
    )


def run_efficiency_census(w, d):
    stream, n, n_l = read_census()
    n_h = n - n_l
    fcm_c_construction_time = 0
    fcm_r_construction_time = 0
    cms_construction_time = 0

    w_l = find_w_l(n_l, n_h, d, w)
    start_time = time.time()
    fcm_c = FairCountMinColumnBased(w, d, [{"l": (0, w_l), "h": (w_l, w)} for _ in range(d)])
    for element, group in stream:
        fcm_c.add(element, group)
    end_time = time.time()
    fcm_c_construction_time = end_time - start_time

    d_l = find_d_l(n_l, n_h, d, w)
    start_time = time.time()
    fcm_r = FairCountMinRowBased(w, d_l, d - d_l)
    for element, group in stream:
        fcm_r.add(element, group)
    end_time = time.time()
    fcm_r_construction_time = end_time - start_time

    start_time = time.time()
    cms = CountMinSketch(w, d)
    for element, group in stream:
        cms.add(element)
    end_time = time.time()
    cms_construction_time = end_time - start_time

    cms_query_time = []
    for element, group in stream[:1000]:
        start_time = time.time()
        cms.count(element)
        end_time = time.time()
        cms_query_time.append(end_time - start_time)

    fcm_c_query_time = []
    for element, group in stream[:1000]:
        start_time = time.time()
        fcm_c.count(element, group)
        end_time = time.time()
        fcm_c_query_time.append(end_time - start_time)

    fcm_r_query_time = []
    for element, group in stream[:1000]:
        start_time = time.time()
        fcm_r.count(element, group)
        end_time = time.time()
        fcm_r_query_time.append(end_time - start_time)

    cms_query_time = np.mean(cms_query_time)
    fcm_c_query_time = np.mean(fcm_c_query_time)
    fcm_r_query_time = np.mean(fcm_r_query_time)

    print("CountMinSketch Construction Time:", cms_construction_time)
    print("FairCountMinColumnBased Construction Time:", fcm_c_construction_time)
    print("FairCountMinRowBased Construction Time:", fcm_r_construction_time)
    print("CountMinSketch Query Time:", cms_query_time)
    print("FairCountMinColumnBased Query Time:", fcm_c_query_time)
    print("FairCountMinRowBased Query Time:", fcm_r_query_time)
    print("============================================")

    return (
        cms_construction_time,
        fcm_c_construction_time,
        fcm_r_construction_time,
        cms_query_time,
        fcm_c_query_time,
        fcm_r_query_time,
    )


def read_google_books(cutoff):
    df = pd.read_csv(
        "data/googlebooks-eng-all-2gram-20120701-a_",
        sep="\t",
        names=["ngram", "year", "match_count", "volume_count"],
        dtype={"ngram": str, "year": int, "match_count": int, "volume_count": int},
    ).sample(random_state=42, frac=0.05)

    df_grouped = df.groupby("ngram").size().reset_index(name="count")
    df_grouped["group"] = df_grouped["count"].apply(lambda x: "l" if x < cutoff else "h")
    grouped_data_dict = df_grouped.set_index("ngram")["group"].to_dict()

    df["group"] = df["ngram"].map(grouped_data_dict).fillna("unknown")

    stream = list(zip(df["ngram"], df["group"]))
    n = len(df_grouped)
    n_l = (df_grouped["group"] == "l").sum()

    return stream, n, n_l


def run_fairness_google_books(w, d, cutoff):
    stream, n, n_l = read_google_books(cutoff)
    n_h = n - n_l
    w_l = find_w_l(n_l, n_h, d, w)
    d_l = find_d_l(n_l, n_h, d, w)

    true_counts_h = defaultdict(int)
    true_counts_l = defaultdict(int)
    N = len(stream)
    N_l = 0
    for element, group in stream:
        (true_counts_l if group == "l" else true_counts_h)[element] += 1
        if group == "l":
            N_l += 1
    N_h = N - N_l
    cms = CountMinSketch(w, d)
    fcm_c = FairCountMinColumnBased(w, d, [{"l": (0, w_l), "h": (w_l, w)} for _ in range(d)])
    fcm_r = FairCountMinRowBased(w, d_l, d - d_l)

    for element, group in stream:
        cms.add(element)
        fcm_c.add(element, group)
        fcm_r.add(element, group)

    mean_approx_factor_cms_l, total_add_error_cms_l = calculate_errors(true_counts_l, cms)
    mean_approx_factor_cms_h, total_add_error_cms_h = calculate_errors(true_counts_h, cms)
    mean_approx_factor_fcm_c_l, total_add_error_fcm_c_l = calculate_errors(true_counts_l, fcm_c, "l")
    mean_approx_factor_fcm_c_h, total_add_error_fcm_c_h = calculate_errors(true_counts_h, fcm_c, "h")
    mean_approx_factor_fcm_r_l, total_add_error_fcm_r_l = calculate_errors(true_counts_l, fcm_r, "l")
    mean_approx_factor_fcm_r_h, total_add_error_fcm_r_h = calculate_errors(true_counts_h, fcm_r, "h")

    print(
        "======================== Google Books, ",
        f"n={n}, n_l={n_l}, n_h={n_h}, N={N}, N_l={N_l}, N_h={N_h}, "
        f"w={w}, w_l={w_l}, w_h={w - w_l}, d={d}, d_l={d_l}, d_h={d - d_l}",
        "========================",
    )
    print(
        f"CountMinSketch (Group l) - Mean Approximation Factor: {mean_approx_factor_cms_l}, Additive Error: {total_add_error_cms_l}"
    )
    print(
        f"CountMinSketch (Group h) - Mean Approximation Factor: {mean_approx_factor_cms_h}, Additive Error: {total_add_error_cms_h}"
    )
    print(
        f"FairCountMinColumnBased (Group l) - Mean Approximation Factor: {mean_approx_factor_fcm_c_l}, Additive Error: {total_add_error_fcm_c_l}"
    )
    print(
        f"FairCountMinColumnBased (Group h) - Mean Approximation Factor: {mean_approx_factor_fcm_c_h}, Additive Error: {total_add_error_fcm_c_h}"
    )
    print(
        f"FairCountMinRowBased (Group l) - Mean Approximation Factor: {mean_approx_factor_fcm_r_l}, Additive Error: {total_add_error_fcm_r_l}"
    )
    print(
        f"FairCountMinRowBased (Group h) - Mean Approximation Factor: {mean_approx_factor_fcm_r_h}, Additive Error: {total_add_error_fcm_r_h}"
    )
    print("============================================")

    return (
        mean_approx_factor_cms_l,
        total_add_error_cms_l,
        mean_approx_factor_cms_h,
        total_add_error_cms_h,
        mean_approx_factor_fcm_c_l,
        total_add_error_fcm_c_l,
        mean_approx_factor_fcm_c_h,
        total_add_error_fcm_c_h,
        mean_approx_factor_fcm_r_l,
        total_add_error_fcm_r_l,
        mean_approx_factor_fcm_r_h,
        total_add_error_fcm_r_h,
        n_l,
    )


def run_efficiency_google_books(w, d, cutoff):
    stream, n, n_l = read_google_books(cutoff)
    n_h = n - n_l
    fcm_c_construction_time = 0
    fcm_r_construction_time = 0
    cms_construction_time = 0

    w_l = find_w_l(n_l, n_h, d, w)
    start_time = time.time()
    fcm_c = FairCountMinColumnBased(w, d, [{"l": (0, w_l), "h": (w_l, w)} for _ in range(d)])
    for element, group in stream:
        fcm_c.add(element, group)
    end_time = time.time()
    fcm_c_construction_time = end_time - start_time

    d_l = find_d_l(n_l, n_h, d, w)
    start_time = time.time()
    fcm_r = FairCountMinRowBased(w, d_l, d - d_l)
    for element, group in stream:
        fcm_r.add(element, group)
    end_time = time.time()
    fcm_r_construction_time = end_time - start_time

    start_time = time.time()
    cms = CountMinSketch(w, d)
    for element, group in stream:
        cms.add(element)
    end_time = time.time()
    cms_construction_time = end_time - start_time

    cms_query_time = []
    for element, group in stream[:1000]:
        start_time = time.time()
        cms.count(element)
        end_time = time.time()
        cms_query_time.append(end_time - start_time)

    fcm_c_query_time = []
    for element, group in stream[:1000]:
        start_time = time.time()
        fcm_c.count(element, group)
        end_time = time.time()
        fcm_c_query_time.append(end_time - start_time)

    fcm_r_query_time = []
    for element, group in stream[:1000]:
        start_time = time.time()
        fcm_r.count(element, group)
        end_time = time.time()
        fcm_r_query_time.append(end_time - start_time)

    cms_query_time = np.mean(cms_query_time)
    fcm_c_query_time = np.mean(fcm_c_query_time)
    fcm_r_query_time = np.mean(fcm_r_query_time)

    print("CountMinSketch Construction Time:", cms_construction_time)
    print("FairCountMinColumnBased Construction Time:", fcm_c_construction_time)
    print("FairCountMinRowBased Construction Time:", fcm_r_construction_time)
    print("CountMinSketch Query Time:", cms_query_time)
    print("FairCountMinColumnBased Query Time:", fcm_c_query_time)
    print("FairCountMinRowBased Query Time:", fcm_r_query_time)
    print("============================================")

    return (
        cms_construction_time,
        fcm_c_construction_time,
        fcm_r_construction_time,
        cms_query_time,
        fcm_c_query_time,
        fcm_r_query_time,
        n_l,
    )

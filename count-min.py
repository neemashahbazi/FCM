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


# def calculate_sum(n, w, d):
#     p = 1 / w
#     q = 1 - p
#     i = np.arange(n + 1)
#     log_binom = gammaln(n + 1) - gammaln(i + 1) - gammaln(n - i + 1)
#     log_pmf = log_binom + i * np.log(p) + (n - i) * np.log(q)
#     binom_pmf = np.exp(log_pmf)
#     tail_probs = np.cumsum(binom_pmf[::-1])[::-1][1:]
#     powered = np.power(tail_probs, d)
#     return np.sum(powered)


def calculate_sum(n, w, d):
    p = 1 / w
    mu = n * p
    sigma = (n * p * (1 - p)) ** 0.5
    start = max(1, int(mu - 6 * sigma))
    end = min(n, int(mu + 6 * sigma))
    c_vals = np.arange(start, end + 1)
    z_scores = (c_vals - 0.5 - mu) / sigma
    tail_probs = norm.sf(z_scores)
    powered = np.power(tail_probs, d)
    return powered.sum()


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


# cms_approx_factor_diffs = []
# fcm_c_approx_factor_diffs = []
# fcm_r_approx_factor_diffs = []
# cms_add_error_totals = []
# fcm_c_add_error_totals = []
# fcm_r_add_error_totals = []
# cms_approx_factor_l = []
# fcm_c_approx_factor_l = []
# fcm_r_approx_factor_l = []
# cms_add_error_l = []
# fcm_c_add_error_l = []
# fcm_r_add_error_l = []
# cms_approx_factor_h = []
# fcm_c_approx_factor_h = []
# fcm_r_approx_factor_h = []
# cms_add_error_h = []
# fcm_c_add_error_h = []
# fcm_r_add_error_h = []
# cms_approx_factor_diffs_std = []
# fcm_c_approx_factor_diffs_std = []
# fcm_r_approx_factor_diffs_std = []
# cms_add_error_totals_std = []
# fcm_c_add_error_totals_std = []
# fcm_r_add_error_totals_std = []
# n_l_label = []


# for cutoff in [i for i in range(2, 13)]:
#     cms_approx_factor_diffs_run = []
#     fcm_c_approx_factor_diffs_run = []
#     fcm_r_approx_factor_diffs_run = []
#     cms_add_error_totals_run = []
#     fcm_c_add_error_totals_run = []
#     fcm_r_add_error_totals_run = []
#     cms_approx_factor_l_run = []
#     fcm_c_approx_factor_l_run = []
#     fcm_r_approx_factor_l_run = []
#     cms_add_error_l_run = []
#     fcm_c_add_error_l_run = []
#     fcm_r_add_error_l_run = []
#     cms_approx_factor_h_run = []
#     fcm_c_approx_factor_h_run = []
#     fcm_r_approx_factor_h_run = []
#     cms_add_error_h_run = []
#     fcm_c_add_error_h_run = []
#     fcm_r_add_error_h_run = []

#     for _ in range(5):
#         (
#             mean_approx_factor_cms_l,
#             total_add_error_cms_l,
#             mean_approx_factor_cms_h,
#             total_add_error_cms_h,
#             mean_approx_factor_fcm_c_l,
#             total_add_error_fcm_c_l,
#             mean_approx_factor_fcm_c_h,
#             total_add_error_fcm_c_h,
#             mean_approx_factor_fcm_r_l,
#             total_add_error_fcm_r_l,
#             mean_approx_factor_fcm_r_h,
#             total_add_error_fcm_r_h,
#             n_l,
#         ) = run_fairness_google_books(
#             w=65536,
#             d=5,
#             cutoff=cutoff,
#         )
#         cms_approx_factor_diffs_run.append(mean_approx_factor_cms_l - mean_approx_factor_cms_h)
#         fcm_c_approx_factor_diffs_run.append(mean_approx_factor_fcm_c_l - mean_approx_factor_fcm_c_h)
#         fcm_r_approx_factor_diffs_run.append(mean_approx_factor_fcm_r_l - mean_approx_factor_fcm_r_h)
#         cms_add_error_totals_run.append(total_add_error_cms_h + total_add_error_cms_l)
#         fcm_c_add_error_totals_run.append(total_add_error_fcm_c_h + total_add_error_fcm_c_l)
#         fcm_r_add_error_totals_run.append(total_add_error_fcm_r_h + total_add_error_fcm_r_l)
#         cms_approx_factor_l_run.append(mean_approx_factor_cms_l)
#         cms_approx_factor_h_run.append(mean_approx_factor_cms_h)
#         cms_add_error_l_run.append(total_add_error_cms_l)
#         cms_add_error_h_run.append(total_add_error_cms_h)
#         fcm_c_approx_factor_l_run.append(mean_approx_factor_fcm_c_l)
#         fcm_c_approx_factor_h_run.append(mean_approx_factor_fcm_c_h)
#         fcm_c_add_error_l_run.append(total_add_error_fcm_c_l)
#         fcm_c_add_error_h_run.append(total_add_error_fcm_c_h)
#         fcm_r_approx_factor_l_run.append(mean_approx_factor_fcm_r_l)
#         fcm_r_approx_factor_h_run.append(mean_approx_factor_fcm_r_h)
#         fcm_r_add_error_l_run.append(total_add_error_fcm_r_l)
#         fcm_r_add_error_h_run.append(total_add_error_fcm_r_h)
#         n_l_label.append(int(n_l))

#     cms_approx_factor_diffs_std.append(np.std(cms_approx_factor_diffs_run))
#     fcm_c_approx_factor_diffs_std.append(np.std(fcm_c_approx_factor_diffs_run))
#     fcm_r_approx_factor_diffs_std.append(np.std(fcm_r_approx_factor_diffs_run))
#     cms_add_error_totals_std.append(np.std(cms_add_error_totals_run))
#     fcm_c_add_error_totals_std.append(np.std(fcm_c_add_error_totals_run))
#     fcm_r_add_error_totals_std.append(np.std(fcm_r_add_error_totals_run))

#     cms_approx_factor_diffs.append(np.mean(cms_approx_factor_diffs_run))
#     fcm_c_approx_factor_diffs.append(np.mean(fcm_c_approx_factor_diffs_run))
#     fcm_r_approx_factor_diffs.append(np.mean(fcm_r_approx_factor_diffs_run))
#     cms_add_error_totals.append(np.mean(cms_add_error_totals_run))
#     fcm_c_add_error_totals.append(np.mean(fcm_c_add_error_totals_run))
#     fcm_r_add_error_totals.append(np.mean(fcm_r_add_error_totals_run))
#     cms_approx_factor_l.append(np.mean(cms_approx_factor_l_run))
#     cms_approx_factor_h.append(np.mean(cms_approx_factor_h_run))
#     cms_add_error_l.append(np.mean(cms_add_error_l_run))
#     cms_add_error_h.append(np.mean(cms_add_error_h_run))
#     fcm_c_approx_factor_l.append(np.mean(fcm_c_approx_factor_l_run))
#     fcm_c_approx_factor_h.append(np.mean(fcm_c_approx_factor_h_run))
#     fcm_c_add_error_l.append(np.mean(fcm_c_add_error_l_run))
#     fcm_c_add_error_h.append(np.mean(fcm_c_add_error_h_run))
#     fcm_r_approx_factor_l.append(np.mean(fcm_r_approx_factor_l_run))
#     fcm_r_approx_factor_h.append(np.mean(fcm_r_approx_factor_h_run))
#     fcm_r_add_error_l.append(np.mean(fcm_r_add_error_l_run))
#     fcm_r_add_error_h.append(np.mean(fcm_r_add_error_h_run))

# results = {
#     "cms_approx_factor_diffs": cms_approx_factor_diffs,
#     "fcm_c_approx_factor_diffs": fcm_c_approx_factor_diffs,
#     "fcm_r_approx_factor_diffs": fcm_r_approx_factor_diffs,
#     "cms_add_error_totals": cms_add_error_totals,
#     "fcm_c_add_error_totals": fcm_c_add_error_totals,
#     "fcm_r_add_error_totals": fcm_r_add_error_totals,
#     "cms_approx_factor_l": cms_approx_factor_l,
#     "fcm_c_approx_factor_l": fcm_c_approx_factor_l,
#     "fcm_r_approx_factor_l": fcm_r_approx_factor_l,
#     "cms_add_error_l": cms_add_error_l,
#     "fcm_c_add_error_l": fcm_c_add_error_l,
#     "fcm_r_add_error_l": fcm_r_add_error_l,
#     "cms_approx_factor_h": cms_approx_factor_h,
#     "fcm_c_approx_factor_h": fcm_c_approx_factor_h,
#     "fcm_r_approx_factor_h": fcm_r_approx_factor_h,
#     "cms_add_error_h": cms_add_error_h,
#     "fcm_c_add_error_h": fcm_c_add_error_h,
#     "fcm_r_add_error_h": fcm_r_add_error_h,
#     "cms_approx_factor_diffs_std": cms_approx_factor_diffs_std,
#     "fcm_c_approx_factor_diffs_std": fcm_c_approx_factor_diffs_std,
#     "fcm_r_approx_factor_diffs_std": fcm_r_approx_factor_diffs_std,
#     "cms_add_error_totals_std": cms_add_error_totals_std,
#     "fcm_c_add_error_totals_std": fcm_c_add_error_totals_std,
#     "fcm_r_add_error_totals_std": fcm_r_add_error_totals_std,
#     "n_l_label": [n_l_label[5 * i] for i in range(11)]
# }

# with open("results_google_books_varying_n_l.json", "w") as f:
#     json.dump(results, f, indent=4)

with open("results_google_books_varying_n_l.json", "r") as f:
    results = json.load(f)

n_l = range(1, 12)

cms_approx_factor_diffs = results["cms_approx_factor_diffs"]
fcm_c_approx_factor_diffs = results["fcm_c_approx_factor_diffs"]
fcm_r_approx_factor_diffs = results["fcm_r_approx_factor_diffs"]
cms_add_error_totals = results["cms_add_error_totals"]
fcm_c_add_error_totals = results["fcm_c_add_error_totals"]
fcm_r_add_error_totals = results["fcm_r_add_error_totals"]
cms_approx_factor_l = results["cms_approx_factor_l"]
fcm_c_approx_factor_l = results["fcm_c_approx_factor_l"]
fcm_r_approx_factor_l = results["fcm_r_approx_factor_l"]
cms_add_error_l = results["cms_add_error_l"]
fcm_c_add_error_l = results["fcm_c_add_error_l"]
fcm_r_add_error_l = results["fcm_r_add_error_l"]
cms_approx_factor_h = results["cms_approx_factor_h"]
fcm_c_approx_factor_h = results["fcm_c_approx_factor_h"]
fcm_r_approx_factor_h = results["fcm_r_approx_factor_h"]
cms_add_error_h = results["cms_add_error_h"]
fcm_c_add_error_h = results["fcm_c_add_error_h"]
fcm_r_add_error_h = results["fcm_r_add_error_h"]
cms_approx_factor_diffs_std = results["cms_approx_factor_diffs_std"]
fcm_c_approx_factor_diffs_std = results["fcm_c_approx_factor_diffs_std"]
fcm_r_approx_factor_diffs_std = results["fcm_r_approx_factor_diffs_std"]
cms_add_error_totals_std = results["cms_add_error_totals_std"]
fcm_c_add_error_totals_std = results["fcm_c_add_error_totals_std"]
fcm_r_add_error_totals_std = results["fcm_r_add_error_totals_std"]
n_l_label = results["n_l_label"]
n_l_label = [round(x / 10**5, 1) for x in n_l_label]

plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.errorbar(
    n_l,
    fcm_c_approx_factor_diffs,
    yerr=fcm_c_approx_factor_diffs_std,
    label="Fair-Count-Min",
    marker="s",
    markersize=10,
    color="tab:orange",
    linewidth=3,
)
plt.errorbar(
    n_l,
    cms_approx_factor_diffs,
    yerr=cms_approx_factor_diffs_std,
    label="Count-Min Baseline",
    marker="o",
    markersize=10,
    color="tab:blue",
    linewidth=3,
)
plt.errorbar(
    n_l,
    fcm_r_approx_factor_diffs,
    fcm_r_approx_factor_diffs_std,
    label="Row-Partitioning Baseline",
    marker="^",
    markersize=10,
    color="tab:green",
    linewidth=3,
)
# plt.xscale("log")
plt.xticks(ticks=n_l, labels=n_l_label)
plt.xlabel("$n_{l}$ ×100000")
plt.ylabel("Mean Approx. Fact. Diff. (l - h)")
plt.legend()
# plt.title("Mean Approximation Factor Difference for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/google_books/mean_approx_factor_difference_plot_varying_n_l.png", bbox_inches="tight")


plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.errorbar(
    n_l,
    cms_add_error_totals,
    yerr=cms_add_error_totals_std,
    label="Count-Min Baseline",
    marker="o",
    markersize=10,
    linewidth=3,
)
plt.errorbar(
    n_l,
    fcm_c_add_error_totals,
    yerr=fcm_c_add_error_totals_std,
    label="Fair-Count-Min",
    marker="s",
    markersize=10,
    linewidth=3,
)
# plt.errorbar(
#     n_l,
#     fcm_r_add_error_totals,
#     yerr=fcm_r_add_error_totals_std,
#     label="Row-Partitioning Baseline",
#     marker="^",
#     markersize=10,
#     linewidth=3,
# )
# plt.xscale("log")
plt.xticks(ticks=n_l, labels=n_l_label)
plt.xlabel("$n_{l}$ ×100000")
plt.ylim(0, 7 * 10**7)
plt.ylabel("Total Additive Error")
plt.legend()
# plt.title("Price of Fairness w.r.t. Additive Error for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/google_books/price_of_fairness_plot_varying_n_l.png", bbox_inches="tight")


plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(n_l, cms_approx_factor_l, label="Count-Min Baseline (l)", marker="o", markersize=10, linewidth=3)
plt.plot(n_l, cms_approx_factor_h, label="Count-Min Baseline (h)", marker="o", markersize=10, linewidth=3)
plt.plot(n_l, fcm_c_approx_factor_l, label="Fair-Count-Min (l)", marker="s", markersize=10, linewidth=3)
plt.plot(n_l, fcm_c_approx_factor_h, label="Fair-Count-Min (h)", marker="s", markersize=10, linewidth=3)
plt.plot(n_l, fcm_r_approx_factor_l, label="Row-Partitioning Baseline (l)", marker="^", markersize=10, linewidth=3)
plt.plot(n_l, fcm_r_approx_factor_h, label="Row-Partitioning Baseline (h)", marker="^", markersize=10, linewidth=3)
# plt.xscale("log")
plt.xticks(ticks=n_l, labels=n_l_label)
plt.xlabel("$n_{l}$ ×100000")
plt.ylabel("Mean Approx. Fact. Abs. Values")
plt.legend()
# plt.title("Mean Approximation Factor for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/google_books/mean_approx_factor_absolute_values_plot_varying_n_l.png", bbox_inches="tight")


plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(n_l, cms_add_error_l, label="Count-Min Baseline (l)", marker="o", markersize=10, linewidth=3)
plt.plot(n_l, cms_add_error_h, label="Count-Min Baseline (h)", marker="o", markersize=10, linewidth=3)
plt.plot(n_l, fcm_c_add_error_l, label="Fair-Count-Min (l)", marker="s", markersize=10, linewidth=3)
plt.plot(n_l, fcm_c_add_error_h, label="Fair-Count-Min (h)", marker="s", markersize=10, linewidth=3)
# plt.plot(n_l, fcm_r_add_error_l, label="Row-Partitioning Baseline (l)", marker="^", markersize=10, linewidth=3)
# plt.plot(n_l, fcm_r_add_error_h, label="Row-Partitioning Baseline (h)", marker="^", markersize=10, linewidth=3)
# plt.xscale("log")
plt.xticks(ticks=n_l, labels=n_l_label)
plt.xlabel("$n_{l}$ ×100000")
plt.ylabel("Additive Absolute Error")
plt.legend()
# plt.title("Additive Error for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/google_books/additive_absolute_error_plot_varying_n_l.png", bbox_inches="tight")


# cms_approx_factor_diffs = []
# fcm_c_approx_factor_diffs = []
# fcm_r_approx_factor_diffs = []
# cms_add_error_totals = []
# fcm_c_add_error_totals = []
# fcm_r_add_error_totals = []
# cms_approx_factor_l = []
# fcm_c_approx_factor_l = []
# fcm_r_approx_factor_l = []
# cms_add_error_l = []
# fcm_c_add_error_l = []
# fcm_r_add_error_l = []
# cms_approx_factor_h = []
# fcm_c_approx_factor_h = []
# fcm_r_approx_factor_h = []
# cms_add_error_h = []
# fcm_c_add_error_h = []
# fcm_r_add_error_h = []

# for w in [2**i for i in range(10, 21)]:
#     cms_approx_factor_diffs_run = []
#     fcm_c_approx_factor_diffs_run = []
#     fcm_r_approx_factor_diffs_run = []
#     cms_add_error_totals_run = []
#     fcm_c_add_error_totals_run = []
#     fcm_r_add_error_totals_run = []
#     cms_approx_factor_l_run = []
#     fcm_c_approx_factor_l_run = []
#     fcm_r_approx_factor_l_run = []
#     cms_add_error_l_run = []
#     fcm_c_add_error_l_run = []
#     fcm_r_add_error_l_run = []
#     cms_approx_factor_h_run = []
#     fcm_c_approx_factor_h_run = []
#     fcm_r_approx_factor_h_run = []
#     cms_add_error_h_run = []
#     fcm_c_add_error_h_run = []
#     fcm_r_add_error_h_run = []

#     for _ in range(1):
#         (
#             mean_approx_factor_cms_l,
#             total_add_error_cms_l,
#             mean_approx_factor_cms_h,
#             total_add_error_cms_h,
#             mean_approx_factor_fcm_c_l,
#             total_add_error_fcm_c_l,
#             mean_approx_factor_fcm_c_h,
#             total_add_error_fcm_c_h,
#             mean_approx_factor_fcm_r_l,
#             total_add_error_fcm_r_l,
#             mean_approx_factor_fcm_r_h,
#             total_add_error_fcm_r_h,
#             n_l
#         ) = run_fairness_google_books(
#             w=w,
#             d=5,
#             cutoff=10,
#         )
#         cms_approx_factor_diffs_run.append(mean_approx_factor_cms_l - mean_approx_factor_cms_h)
#         fcm_c_approx_factor_diffs_run.append(mean_approx_factor_fcm_c_l - mean_approx_factor_fcm_c_h)
#         fcm_r_approx_factor_diffs_run.append(mean_approx_factor_fcm_r_l - mean_approx_factor_fcm_r_h)
#         cms_add_error_totals_run.append(total_add_error_cms_h + total_add_error_cms_l)
#         fcm_c_add_error_totals_run.append(total_add_error_fcm_c_h + total_add_error_fcm_c_l)
#         fcm_r_add_error_totals_run.append(total_add_error_fcm_r_h + total_add_error_fcm_r_l)
#         cms_approx_factor_l_run.append(mean_approx_factor_cms_l)
#         cms_approx_factor_h_run.append(mean_approx_factor_cms_h)
#         cms_add_error_l_run.append(total_add_error_cms_l)
#         cms_add_error_h_run.append(total_add_error_cms_h)
#         fcm_c_approx_factor_l_run.append(mean_approx_factor_fcm_c_l)
#         fcm_c_approx_factor_h_run.append(mean_approx_factor_fcm_c_h)
#         fcm_c_add_error_l_run.append(total_add_error_fcm_c_l)
#         fcm_c_add_error_h_run.append(total_add_error_fcm_c_h)
#         fcm_r_approx_factor_l_run.append(mean_approx_factor_fcm_r_l)
#         fcm_r_approx_factor_h_run.append(mean_approx_factor_fcm_r_h)
#         fcm_r_add_error_l_run.append(total_add_error_fcm_r_l)
#         fcm_r_add_error_h_run.append(total_add_error_fcm_r_h)

#     cms_approx_factor_diffs.append(np.mean(cms_approx_factor_diffs_run))
#     fcm_c_approx_factor_diffs.append(np.mean(fcm_c_approx_factor_diffs_run))
#     fcm_r_approx_factor_diffs.append(np.mean(fcm_r_approx_factor_diffs_run))
#     cms_add_error_totals.append(np.mean(cms_add_error_totals_run))
#     fcm_c_add_error_totals.append(np.mean(fcm_c_add_error_totals_run))
#     fcm_r_add_error_totals.append(np.mean(fcm_r_add_error_totals_run))
#     cms_approx_factor_l.append(np.mean(cms_approx_factor_l_run))
#     cms_approx_factor_h.append(np.mean(cms_approx_factor_h_run))
#     cms_add_error_l.append(np.mean(cms_add_error_l_run))
#     cms_add_error_h.append(np.mean(cms_add_error_h_run))
#     fcm_c_approx_factor_l.append(np.mean(fcm_c_approx_factor_l_run))
#     fcm_c_approx_factor_h.append(np.mean(fcm_c_approx_factor_h_run))
#     fcm_c_add_error_l.append(np.mean(fcm_c_add_error_l_run))
#     fcm_c_add_error_h.append(np.mean(fcm_c_add_error_h_run))
#     fcm_r_approx_factor_l.append(np.mean(fcm_r_approx_factor_l_run))
#     fcm_r_approx_factor_h.append(np.mean(fcm_r_approx_factor_h_run))
#     fcm_r_add_error_l.append(np.mean(fcm_r_add_error_l_run))
#     fcm_r_add_error_h.append(np.mean(fcm_r_add_error_h_run))

# results = {
#     "cms_approx_factor_diffs": cms_approx_factor_diffs,
#     "fcm_c_approx_factor_diffs": fcm_c_approx_factor_diffs,
#     "fcm_r_approx_factor_diffs": fcm_r_approx_factor_diffs,
#     "cms_add_error_totals": cms_add_error_totals,
#     "fcm_c_add_error_totals": fcm_c_add_error_totals,
#     "fcm_r_add_error_totals": fcm_r_add_error_totals,
#     "cms_approx_factor_l": cms_approx_factor_l,
#     "fcm_c_approx_factor_l": fcm_c_approx_factor_l,
#     "fcm_r_approx_factor_l": fcm_r_approx_factor_l,
#     "cms_add_error_l": cms_add_error_l,
#     "fcm_c_add_error_l": fcm_c_add_error_l,
#     "fcm_r_add_error_l": fcm_r_add_error_l,
#     "cms_approx_factor_h": cms_approx_factor_h,
#     "fcm_c_approx_factor_h": fcm_c_approx_factor_h,
#     "fcm_r_approx_factor_h": fcm_r_approx_factor_h,
#     "cms_add_error_h": cms_add_error_h,
#     "fcm_c_add_error_h": fcm_c_add_error_h,
#     "fcm_r_add_error_h": fcm_r_add_error_h,
# }

# with open("results_google_books_varying_w.json", "w") as f:
#     json.dump(results, f, indent=4)

with open("results_google_books_varying_w.json", "r") as f:
    results = json.load(f)

cms_approx_factor_diffs = results["cms_approx_factor_diffs"]
fcm_c_approx_factor_diffs = results["fcm_c_approx_factor_diffs"]
fcm_r_approx_factor_diffs = results["fcm_r_approx_factor_diffs"]
cms_add_error_totals = results["cms_add_error_totals"]
fcm_c_add_error_totals = results["fcm_c_add_error_totals"]
fcm_r_add_error_totals = results["fcm_r_add_error_totals"]
cms_approx_factor_l = results["cms_approx_factor_l"]
fcm_c_approx_factor_l = results["fcm_c_approx_factor_l"]
fcm_r_approx_factor_l = results["fcm_r_approx_factor_l"]
cms_add_error_l = results["cms_add_error_l"]
fcm_c_add_error_l = results["fcm_c_add_error_l"]
fcm_r_add_error_l = results["fcm_r_add_error_l"]
cms_approx_factor_h = results["cms_approx_factor_h"]
fcm_c_approx_factor_h = results["fcm_c_approx_factor_h"]
fcm_r_approx_factor_h = results["fcm_r_approx_factor_h"]
cms_add_error_h = results["cms_add_error_h"]
fcm_c_add_error_h = results["fcm_c_add_error_h"]
fcm_r_add_error_h = results["fcm_r_add_error_h"]

plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(1, 12), cms_approx_factor_diffs, label="Count-Min Baseline", marker="o", markersize=10, linewidth=3)
plt.plot(range(1, 12), fcm_c_approx_factor_diffs, label="Fair-Count-Min", marker="s", markersize=10, linewidth=3)
plt.plot(
    range(1, 12), fcm_r_approx_factor_diffs, label="Row-Partitioning Baseline", marker="^", markersize=10, linewidth=3
)
plt.xticks(ticks=range(1, 12), labels=[r"$2^{{{}}}$".format(i) for i in range(10, 21)])
plt.xlabel("w")
plt.ylabel("Mean Approx. Fact. Diff. (l - h)")
plt.legend()
# plt.title("Mean Approximation Factor Differences for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/google_books/mean_approx_factor_difference_plot_varying_w.png", bbox_inches="tight")

plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(1, 12), cms_add_error_totals, label="Count-Min Baseline", marker="o", markersize=12, linewidth=6)
plt.plot(range(1, 12), fcm_c_add_error_totals, label="Fair-Count-Min", marker="s", markersize=8, linewidth=2)
# plt.plot(range(1, 12), fcm_r_add_error_totals, label="Row-Partitioning Baseline", marker="^", markersize=10, linewidth=3)
plt.xticks(ticks=range(1, 12), labels=[r"$2^{{{}}}$".format(i) for i in range(10, 21)])
plt.xlabel("w")
plt.ylabel("Total Additive Error")
plt.legend()
# plt.title("Price of Fairness w.r.t. Additive Error for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/google_books/price_of_fairness_plot_varying_w.png", bbox_inches="tight")


plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(1, 12), cms_approx_factor_l, label="Count-Min Baseline (l)", marker="o", markersize=10, linewidth=3)
plt.plot(range(1, 12), cms_approx_factor_h, label="Count-Min Baseline (h)", marker="o", markersize=10, linewidth=3)
plt.plot(range(1, 12), fcm_c_approx_factor_l, label="Fair-Count-Min (l)", marker="s", markersize=10, linewidth=3)
plt.plot(range(1, 12), fcm_c_approx_factor_h, label="Fair-Count-Min (h)", marker="s", markersize=10, linewidth=3)
plt.plot(
    range(1, 12), fcm_r_approx_factor_l, label="Row-Partitioning Baseline (l)", marker="^", markersize=10, linewidth=3
)
plt.plot(
    range(1, 12), fcm_r_approx_factor_h, label="Row-Partitioning Baseline (h)", marker="^", markersize=10, linewidth=3
)
plt.xticks(ticks=range(1, 12), labels=[r"$2^{{{}}}$".format(i) for i in range(10, 21)])
plt.xlabel("w")
plt.ylabel("Mean Approx. Fact. Abs. Values")
plt.legend()
# plt.title("Mean Approximation Factor for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/google_books/mean_approx_factor_absolute_values_plot_varying_w.png", bbox_inches="tight")


plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(1, 12), cms_add_error_l, label="Count-Min Baseline (l)", marker="o", markersize=10, linewidth=3)
plt.plot(range(1, 12), cms_add_error_h, label="Count-Min Baseline (h)", marker="o", markersize=10, linewidth=3)
plt.plot(range(1, 12), fcm_c_add_error_l, label="Fair-Count-Min (l)", marker="s", markersize=10, linewidth=3)
plt.plot(range(1, 12), fcm_c_add_error_h, label="Fair-Count-Min (h)", marker="s", markersize=10, linewidth=3)
# plt.plot(range(1, 12), fcm_r_add_error_l, label="Row-Partitioning Baseline (l)", marker="^", markersize=10, linewidth=3)
# plt.plot(range(1, 12), fcm_r_add_error_h, label="Row-Partitioning Baseline (h)", marker="^", markersize=10, linewidth=3)
plt.xticks(ticks=range(1, 12), labels=[r"$2^{{{}}}$".format(i) for i in range(10, 21)])
plt.xlabel("w")
plt.ylabel("Additive Absolute Error ")
plt.legend()
# plt.title("Additive Error for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/google_books/additive_absolute_error_plot_varying_w.png", bbox_inches="tight")


# cms_approx_factor_diffs = []
# fcm_c_approx_factor_diffs = []
# fcm_r_approx_factor_diffs = []
# cms_add_error_totals = []
# fcm_c_add_error_totals = []
# fcm_r_add_error_totals = []
# cms_approx_factor_l = []
# fcm_c_approx_factor_l = []
# fcm_r_approx_factor_l = []
# cms_add_error_l = []
# fcm_c_add_error_l = []
# fcm_r_add_error_l = []
# cms_approx_factor_h = []
# fcm_c_approx_factor_h = []
# fcm_r_approx_factor_h = []
# cms_add_error_h = []
# fcm_c_add_error_h = []
# fcm_r_add_error_h = []

# for d in [i for i in range(2, 11)]:
#     cms_approx_factor_diffs_run = []
#     fcm_c_approx_factor_diffs_run = []
#     fcm_r_approx_factor_diffs_run = []
#     cms_add_error_totals_run = []
#     fcm_c_add_error_totals_run = []
#     fcm_r_add_error_totals_run = []
#     cms_approx_factor_l_run = []
#     fcm_c_approx_factor_l_run = []
#     fcm_r_approx_factor_l_run = []
#     cms_add_error_l_run = []
#     fcm_c_add_error_l_run = []
#     fcm_r_add_error_l_run = []
#     cms_approx_factor_h_run = []
#     fcm_c_approx_factor_h_run = []
#     fcm_r_approx_factor_h_run = []
#     cms_add_error_h_run = []
#     fcm_c_add_error_h_run = []
#     fcm_r_add_error_h_run = []

#     for _ in range(1):
#         (
#             mean_approx_factor_cms_l,
#             total_add_error_cms_l,
#             mean_approx_factor_cms_h,
#             total_add_error_cms_h,
#             mean_approx_factor_fcm_c_l,
#             total_add_error_fcm_c_l,
#             mean_approx_factor_fcm_c_h,
#             total_add_error_fcm_c_h,
#             mean_approx_factor_fcm_r_l,
#             total_add_error_fcm_r_l,
#             mean_approx_factor_fcm_r_h,
#             total_add_error_fcm_r_h,
#             n_l
#         ) = run_fairness_google_books(
#             w=65536,
#             d=d,
#             cutoff=10,
#         )
#         cms_approx_factor_diffs_run.append(mean_approx_factor_cms_l - mean_approx_factor_cms_h)
#         fcm_c_approx_factor_diffs_run.append(mean_approx_factor_fcm_c_l - mean_approx_factor_fcm_c_h)
#         fcm_r_approx_factor_diffs_run.append(mean_approx_factor_fcm_r_l - mean_approx_factor_fcm_r_h)
#         cms_add_error_totals_run.append(total_add_error_cms_h + total_add_error_cms_l)
#         fcm_c_add_error_totals_run.append(total_add_error_fcm_c_h + total_add_error_fcm_c_l)
#         fcm_r_add_error_totals_run.append(total_add_error_fcm_r_h + total_add_error_fcm_r_l)
#         cms_approx_factor_l_run.append(mean_approx_factor_cms_l)
#         cms_approx_factor_h_run.append(mean_approx_factor_cms_h)
#         cms_add_error_l_run.append(total_add_error_cms_l)
#         cms_add_error_h_run.append(total_add_error_cms_h)
#         fcm_c_approx_factor_l_run.append(mean_approx_factor_fcm_c_l)
#         fcm_c_approx_factor_h_run.append(mean_approx_factor_fcm_c_h)
#         fcm_c_add_error_l_run.append(total_add_error_fcm_c_l)
#         fcm_c_add_error_h_run.append(total_add_error_fcm_c_h)
#         fcm_r_approx_factor_l_run.append(mean_approx_factor_fcm_r_l)
#         fcm_r_approx_factor_h_run.append(mean_approx_factor_fcm_r_h)
#         fcm_r_add_error_l_run.append(total_add_error_fcm_r_l)
#         fcm_r_add_error_h_run.append(total_add_error_fcm_r_h)

#     cms_approx_factor_diffs.append(np.mean(cms_approx_factor_diffs_run))
#     fcm_c_approx_factor_diffs.append(np.mean(fcm_c_approx_factor_diffs_run))
#     fcm_r_approx_factor_diffs.append(np.mean(fcm_r_approx_factor_diffs_run))
#     cms_add_error_totals.append(np.mean(cms_add_error_totals_run))
#     fcm_c_add_error_totals.append(np.mean(fcm_c_add_error_totals_run))
#     fcm_r_add_error_totals.append(np.mean(fcm_r_add_error_totals_run))
#     cms_approx_factor_l.append(np.mean(cms_approx_factor_l_run))
#     cms_approx_factor_h.append(np.mean(cms_approx_factor_h_run))
#     cms_add_error_l.append(np.mean(cms_add_error_l_run))
#     cms_add_error_h.append(np.mean(cms_add_error_h_run))
#     fcm_c_approx_factor_l.append(np.mean(fcm_c_approx_factor_l_run))
#     fcm_c_approx_factor_h.append(np.mean(fcm_c_approx_factor_h_run))
#     fcm_c_add_error_l.append(np.mean(fcm_c_add_error_l_run))
#     fcm_c_add_error_h.append(np.mean(fcm_c_add_error_h_run))
#     fcm_r_approx_factor_l.append(np.mean(fcm_r_approx_factor_l_run))
#     fcm_r_approx_factor_h.append(np.mean(fcm_r_approx_factor_h_run))
#     fcm_r_add_error_l.append(np.mean(fcm_r_add_error_l_run))
#     fcm_r_add_error_h.append(np.mean(fcm_r_add_error_h_run))

# results = {
#     "cms_approx_factor_diffs": cms_approx_factor_diffs,
#     "fcm_c_approx_factor_diffs": fcm_c_approx_factor_diffs,
#     "fcm_r_approx_factor_diffs": fcm_r_approx_factor_diffs,
#     "cms_add_error_totals": cms_add_error_totals,
#     "fcm_c_add_error_totals": fcm_c_add_error_totals,
#     "fcm_r_add_error_totals": fcm_r_add_error_totals,
#     "cms_approx_factor_l": cms_approx_factor_l,
#     "fcm_c_approx_factor_l": fcm_c_approx_factor_l,
#     "fcm_r_approx_factor_l": fcm_r_approx_factor_l,
#     "cms_add_error_l": cms_add_error_l,
#     "fcm_c_add_error_l": fcm_c_add_error_l,
#     "fcm_r_add_error_l": fcm_r_add_error_l,
#     "cms_approx_factor_h": cms_approx_factor_h,
#     "fcm_c_approx_factor_h": fcm_c_approx_factor_h,
#     "fcm_r_approx_factor_h": fcm_r_approx_factor_h,
#     "cms_add_error_h": cms_add_error_h,
#     "fcm_c_add_error_h": fcm_c_add_error_h,
#     "fcm_r_add_error_h": fcm_r_add_error_h,
# }

# with open("results_google_books_varying_d.json", "w") as f:
#     json.dump(results, f, indent=4)

with open("results_google_books_varying_d.json", "r") as f:
    results = json.load(f)

cms_approx_factor_diffs = results["cms_approx_factor_diffs"]
fcm_c_approx_factor_diffs = results["fcm_c_approx_factor_diffs"]
fcm_r_approx_factor_diffs = results["fcm_r_approx_factor_diffs"]
cms_add_error_totals = results["cms_add_error_totals"]
fcm_c_add_error_totals = results["fcm_c_add_error_totals"]
fcm_r_add_error_totals = results["fcm_r_add_error_totals"]
cms_approx_factor_l = results["cms_approx_factor_l"]
fcm_c_approx_factor_l = results["fcm_c_approx_factor_l"]
fcm_r_approx_factor_l = results["fcm_r_approx_factor_l"]
cms_add_error_l = results["cms_add_error_l"]
fcm_c_add_error_l = results["fcm_c_add_error_l"]
fcm_r_add_error_l = results["fcm_r_add_error_l"]
cms_approx_factor_h = results["cms_approx_factor_h"]
fcm_c_approx_factor_h = results["fcm_c_approx_factor_h"]
fcm_r_approx_factor_h = results["fcm_r_approx_factor_h"]
cms_add_error_h = results["cms_add_error_h"]
fcm_c_add_error_h = results["fcm_c_add_error_h"]
fcm_r_add_error_h = results["fcm_r_add_error_h"]

plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(2, 11), cms_approx_factor_diffs, label="Count-Min Baseline", marker="o", markersize=10, linewidth=3)
plt.plot(range(2, 11), fcm_c_approx_factor_diffs, label="Fair-Count-Min", marker="s", markersize=10, linewidth=3)
# plt.plot(range(2, 11), fcm_r_approx_factor_diffs, label="Row-Partitioning Baseline", marker="^", markersize=10, linewidth=3)
plt.xticks(ticks=range(2, 11), labels=[i for i in range(2, 11)])
plt.xlabel("d")
plt.ylabel("Mean Approx. Fact. Diff. (l - h)")
plt.legend()
# plt.title("Mean Approximation Factor Differences for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/google_books/mean_approx_factor_difference_plot_varying_d.png", bbox_inches="tight")


plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(2, 11), cms_add_error_totals, label="Count-Min Baseline", marker="o", markersize=10, linewidth=3)
plt.plot(range(2, 11), fcm_c_add_error_totals, label="Fair-Count-Min", marker="s", markersize=10, linewidth=3)
# plt.plot(range(2, 11), fcm_r_add_error_totals, label="Row-Partitioning Baseline", marker="^", markersize=10, linewidth=3)
plt.xticks(ticks=range(2, 11), labels=[i for i in range(2, 11)])
plt.xlabel("d")
plt.ylim(0, 8 * 10**7)
plt.ylabel("Total Additive Error")
plt.legend()
# plt.title("Price of Fairness w.r.t. Additive Error for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/google_books/price_of_fairness_plot_varying_d.png", bbox_inches="tight")


plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(2, 11), cms_approx_factor_l, label="Count-Min Baseline (l)", marker="o", markersize=10, linewidth=3)
plt.plot(range(2, 11), cms_approx_factor_h, label="Count-Min Baseline (h)", marker="o", markersize=10, linewidth=3)
plt.plot(range(2, 11), fcm_c_approx_factor_l, label="Fair-Count-Min (l)", marker="s", markersize=10, linewidth=3)
plt.plot(range(2, 11), fcm_c_approx_factor_h, label="Fair-Count-Min (h)", marker="s", markersize=10, linewidth=3)
# plt.plot(range(2, 11), fcm_r_approx_factor_l, label="Row-Partitioning Baseline (l)", marker="^", markersize=10, linewidth=3)
# plt.plot(range(2, 11), fcm_r_approx_factor_h, label="Row-Partitioning Baseline (h)", marker="^", markersize=10, linewidth=3)
plt.xticks(ticks=range(2, 11), labels=[i for i in range(2, 11)])
plt.xlabel("d")
plt.ylabel("Mean Approx. Fact. Abs. Values")
plt.legend()
# plt.title("Mean Approximation Factor for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/google_books/mean_approx_factor_absolute_values_plot_varying_d.png", bbox_inches="tight")


plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(2, 11), cms_add_error_l, label="Count-Min Baseline (l)", marker="o", markersize=10, linewidth=3)
plt.plot(range(2, 11), cms_add_error_h, label="Count-Min Baseline (h)", marker="o", markersize=10, linewidth=3)
plt.plot(range(2, 11), fcm_c_add_error_l, label="Fair-Count-Min (l)", marker="s", markersize=10, linewidth=3)
plt.plot(range(2, 11), fcm_c_add_error_h, label="Fair-Count-Min (h)", marker="s", markersize=10, linewidth=3)
# plt.plot(range(2, 11), fcm_r_add_error_l, label="Row-Partitioning Baseline (l)", marker="^", markersize=10, linewidth=3)
# plt.plot(range(2, 11), fcm_r_add_error_h, label="Row-Partitioning Baseline (h)", marker="^", markersize=10, linewidth=3)
plt.xticks(ticks=range(2, 11), labels=[i for i in range(2, 11)])
plt.xlabel("d")
plt.ylabel("Additive Absolute Error")
plt.legend()
# plt.title("Additive Error for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/google_books/additive_absolute_error_plot_varying_d.png", bbox_inches="tight")

# # --------------------------------------------------------------------------------------------------

# cms_approx_factor_diffs = []
# fcm_c_approx_factor_diffs = []
# fcm_r_approx_factor_diffs = []
# cms_add_error_totals = []
# fcm_c_add_error_totals = []
# fcm_r_add_error_totals = []
# cms_approx_factor_l = []
# fcm_c_approx_factor_l = []
# fcm_r_approx_factor_l = []
# cms_add_error_l = []
# fcm_c_add_error_l = []
# fcm_r_add_error_l = []
# cms_approx_factor_h = []
# fcm_c_approx_factor_h = []
# fcm_r_approx_factor_h = []
# cms_add_error_h = []
# fcm_c_add_error_h = []
# fcm_r_add_error_h = []

# for w in [2**i for i in range(3, 9)]:
#     cms_approx_factor_diffs_run = []
#     fcm_c_approx_factor_diffs_run = []
#     fcm_r_approx_factor_diffs_run = []
#     cms_add_error_totals_run = []
#     fcm_c_add_error_totals_run = []
#     fcm_r_add_error_totals_run = []
#     cms_approx_factor_l_run = []
#     fcm_c_approx_factor_l_run = []
#     fcm_r_approx_factor_l_run = []
#     cms_add_error_l_run = []
#     fcm_c_add_error_l_run = []
#     fcm_r_add_error_l_run = []
#     cms_approx_factor_h_run = []
#     fcm_c_approx_factor_h_run = []
#     fcm_r_approx_factor_h_run = []
#     cms_add_error_h_run = []
#     fcm_c_add_error_h_run = []
#     fcm_r_add_error_h_run = []

#     for _ in range(5):
#         (
#             mean_approx_factor_cms_l,
#             total_add_error_cms_l,
#             mean_approx_factor_cms_h,
#             total_add_error_cms_h,
#             mean_approx_factor_fcm_c_l,
#             total_add_error_fcm_c_l,
#             mean_approx_factor_fcm_c_h,
#             total_add_error_fcm_c_h,
#             mean_approx_factor_fcm_r_l,
#             total_add_error_fcm_r_l,
#             mean_approx_factor_fcm_r_h,
#             total_add_error_fcm_r_h,
#         ) = run_fairness_census(
#             w=w,
#             d=10,
#         )
#         cms_approx_factor_diffs_run.append(mean_approx_factor_cms_l - mean_approx_factor_cms_h)
#         fcm_c_approx_factor_diffs_run.append(mean_approx_factor_fcm_c_l - mean_approx_factor_fcm_c_h)
#         fcm_r_approx_factor_diffs_run.append(mean_approx_factor_fcm_r_l - mean_approx_factor_fcm_r_h)
#         cms_add_error_totals_run.append(total_add_error_cms_h + total_add_error_cms_l)
#         fcm_c_add_error_totals_run.append(total_add_error_fcm_c_h + total_add_error_fcm_c_l)
#         fcm_r_add_error_totals_run.append(total_add_error_fcm_r_h + total_add_error_fcm_r_l)
#         cms_approx_factor_l_run.append(mean_approx_factor_cms_l)
#         cms_approx_factor_h_run.append(mean_approx_factor_cms_h)
#         cms_add_error_l_run.append(total_add_error_cms_l)
#         cms_add_error_h_run.append(total_add_error_cms_h)
#         fcm_c_approx_factor_l_run.append(mean_approx_factor_fcm_c_l)
#         fcm_c_approx_factor_h_run.append(mean_approx_factor_fcm_c_h)
#         fcm_c_add_error_l_run.append(total_add_error_fcm_c_l)
#         fcm_c_add_error_h_run.append(total_add_error_fcm_c_h)
#         fcm_r_approx_factor_l_run.append(mean_approx_factor_fcm_r_l)
#         fcm_r_approx_factor_h_run.append(mean_approx_factor_fcm_r_h)
#         fcm_r_add_error_l_run.append(total_add_error_fcm_r_l)
#         fcm_r_add_error_h_run.append(total_add_error_fcm_r_h)

#     cms_approx_factor_diffs.append(np.mean(cms_approx_factor_diffs_run))
#     fcm_c_approx_factor_diffs.append(np.mean(fcm_c_approx_factor_diffs_run))
#     fcm_r_approx_factor_diffs.append(np.mean(fcm_r_approx_factor_diffs_run))
#     cms_add_error_totals.append(np.mean(cms_add_error_totals_run))
#     fcm_c_add_error_totals.append(np.mean(fcm_c_add_error_totals_run))
#     fcm_r_add_error_totals.append(np.mean(fcm_r_add_error_totals_run))
#     cms_approx_factor_l.append(np.mean(cms_approx_factor_l_run))
#     cms_approx_factor_h.append(np.mean(cms_approx_factor_h_run))
#     cms_add_error_l.append(np.mean(cms_add_error_l_run))
#     cms_add_error_h.append(np.mean(cms_add_error_h_run))
#     fcm_c_approx_factor_l.append(np.mean(fcm_c_approx_factor_l_run))
#     fcm_c_approx_factor_h.append(np.mean(fcm_c_approx_factor_h_run))
#     fcm_c_add_error_l.append(np.mean(fcm_c_add_error_l_run))
#     fcm_c_add_error_h.append(np.mean(fcm_c_add_error_h_run))
#     fcm_r_approx_factor_l.append(np.mean(fcm_r_approx_factor_l_run))
#     fcm_r_approx_factor_h.append(np.mean(fcm_r_approx_factor_h_run))
#     fcm_r_add_error_l.append(np.mean(fcm_r_add_error_l_run))
#     fcm_r_add_error_h.append(np.mean(fcm_r_add_error_h_run))

# results = {
#     "cms_approx_factor_diffs": cms_approx_factor_diffs,
#     "fcm_c_approx_factor_diffs": fcm_c_approx_factor_diffs,
#     "fcm_r_approx_factor_diffs": fcm_r_approx_factor_diffs,
#     "cms_add_error_totals": cms_add_error_totals,
#     "fcm_c_add_error_totals": fcm_c_add_error_totals,
#     "fcm_r_add_error_totals": fcm_r_add_error_totals,
#     "cms_approx_factor_l": cms_approx_factor_l,
#     "fcm_c_approx_factor_l": fcm_c_approx_factor_l,
#     "fcm_r_approx_factor_l": fcm_r_approx_factor_l,
#     "cms_add_error_l": cms_add_error_l,
#     "fcm_c_add_error_l": fcm_c_add_error_l,
#     "fcm_r_add_error_l": fcm_r_add_error_l,
#     "cms_approx_factor_h": cms_approx_factor_h,
#     "fcm_c_approx_factor_h": fcm_c_approx_factor_h,
#     "fcm_r_approx_factor_h": fcm_r_approx_factor_h,
#     "cms_add_error_h": cms_add_error_h,
#     "fcm_c_add_error_h": fcm_c_add_error_h,
#     "fcm_r_add_error_h": fcm_r_add_error_h,
# }

# with open("results_census_varying_w.json", "w") as f:
#     json.dump(results, f, indent=4)

with open("results_census_varying_w.json", "r") as f:
    results = json.load(f)

cms_approx_factor_diffs = results["cms_approx_factor_diffs"]
fcm_c_approx_factor_diffs = results["fcm_c_approx_factor_diffs"]
fcm_r_approx_factor_diffs = results["fcm_r_approx_factor_diffs"]
cms_add_error_totals = results["cms_add_error_totals"]
fcm_c_add_error_totals = results["fcm_c_add_error_totals"]
fcm_r_add_error_totals = results["fcm_r_add_error_totals"]
cms_approx_factor_l = results["cms_approx_factor_l"]
fcm_c_approx_factor_l = results["fcm_c_approx_factor_l"]
fcm_r_approx_factor_l = results["fcm_r_approx_factor_l"]
cms_add_error_l = results["cms_add_error_l"]
fcm_c_add_error_l = results["fcm_c_add_error_l"]
fcm_r_add_error_l = results["fcm_r_add_error_l"]
cms_approx_factor_h = results["cms_approx_factor_h"]
fcm_c_approx_factor_h = results["fcm_c_approx_factor_h"]
fcm_r_approx_factor_h = results["fcm_r_approx_factor_h"]
cms_add_error_h = results["cms_add_error_h"]
fcm_c_add_error_h = results["fcm_c_add_error_h"]
fcm_r_add_error_h = results["fcm_r_add_error_h"]

plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(1, 7), cms_approx_factor_diffs, label="Count-Min Baseline", marker="o", markersize=10, linewidth=3)
plt.plot(range(1, 7), fcm_c_approx_factor_diffs, label="Fair-Count-Min", marker="s", markersize=10, linewidth=3)
# plt.plot(range(1, 6), fcm_r_approx_factor_diffs, label="Row-Partitioning Baseline", marker="^", markersize=10, linewidth=3)
plt.xticks(ticks=range(1, 7), labels=[2**i for i in range(3, 9)])
plt.xlabel("w")
plt.ylabel("Mean Approx. Fact. Diff. (F - M)")
plt.legend()
# plt.title("Mean Approximation Factor Differences for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/census/mean_approx_factor_difference_plot_varying_w.png", bbox_inches="tight")

plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(1, 7), cms_add_error_totals, label="Count-Min Baseline", marker="o", markersize=10, linewidth=3)
plt.plot(range(1, 7), fcm_c_add_error_totals, label="Fair-Count-Min", marker="s", markersize=10, linewidth=3)
# plt.plot(range(1, 6), fcm_r_add_error_totals, label="Row-Partitioning Baseline", marker="^", markersize=10, linewidth=3)
plt.xticks(ticks=range(1, 7), labels=[2**i for i in range(3, 9)])
plt.xlabel("w")
plt.ylabel("Total Additive Error")
plt.legend()
# plt.title("Price of Fairness w.r.t. Additive Error for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/census/price_of_fairness_plot_varying_w.png", bbox_inches="tight")


plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(1, 7), cms_approx_factor_l, label="Count-Min Baseline (l)", marker="o", markersize=10, linewidth=3)
plt.plot(range(1, 7), cms_approx_factor_h, label="Count-Min Baseline (h)", marker="o", markersize=10, linewidth=3)
plt.plot(range(1, 7), fcm_c_approx_factor_l, label="Fair-Count-Min (l)", marker="s", markersize=10, linewidth=3)
plt.plot(range(1, 7), fcm_c_approx_factor_h, label="Fair-Count-Min (h)", marker="s", markersize=10, linewidth=3)
# plt.plot(range(1, 6), fcm_r_approx_factor_l, label="Row-Partitioning Baseline (l)", marker="^", markersize=10, linewidth=3)
# plt.plot(range(1, 6), fcm_r_approx_factor_h, label="Row-Partitioning Baseline (h)", marker="^", markersize=10, linewidth=3)
plt.xticks(ticks=range(1, 7), labels=[2**i for i in range(3, 9)])
plt.xlabel("w")
plt.ylabel("Mean Approx. Fact. Abs. Values")
plt.legend()
# plt.title("Mean Approximation Factor for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/census/mean_approx_factor_absolute_values_plot_varying_w.png", bbox_inches="tight")


plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(1, 7), cms_add_error_l, label="Count-Min Baseline (l)", marker="o", markersize=10, linewidth=3)
plt.plot(range(1, 7), cms_add_error_h, label="Count-Min Baseline (h)", marker="o", markersize=10, linewidth=3)
plt.plot(range(1, 7), fcm_c_add_error_l, label="Fair-Count-Min (l)", marker="s", markersize=10, linewidth=3)
plt.plot(range(1, 7), fcm_c_add_error_h, label="Fair-Count-Min (h)", marker="s", markersize=10, linewidth=3)
# plt.plot(range(1, 6), fcm_r_add_error_l, label="Row-Partitioning Baseline (l)", marker="^", markersize=10, linewidth=3)
# plt.plot(range(1, 6), fcm_r_add_error_h, label="Row-Partitioning Baseline (h)", marker="^", markersize=10, linewidth=3)
plt.xticks(ticks=range(1, 7), labels=[2**i for i in range(3, 9)])
plt.xlabel("w")
plt.ylabel("Additive Absolute Error")
plt.legend()
# plt.title("Additive Error for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/census/additive_absolute_error_plot_varying_w.png", bbox_inches="tight")


# cms_approx_factor_diffs = []
# fcm_c_approx_factor_diffs = []
# fcm_r_approx_factor_diffs = []
# cms_add_error_totals = []
# fcm_c_add_error_totals = []
# fcm_r_add_error_totals = []
# cms_approx_factor_l = []
# fcm_c_approx_factor_l = []
# fcm_r_approx_factor_l = []
# cms_add_error_l = []
# fcm_c_add_error_l = []
# fcm_r_add_error_l = []
# cms_approx_factor_h = []
# fcm_c_approx_factor_h = []
# fcm_r_approx_factor_h = []
# cms_add_error_h = []
# fcm_c_add_error_h = []
# fcm_r_add_error_h = []

# for d in [i for i in range(2, 11)]:
#     cms_approx_factor_diffs_run = []
#     fcm_c_approx_factor_diffs_run = []
#     fcm_r_approx_factor_diffs_run = []
#     cms_add_error_totals_run = []
#     fcm_c_add_error_totals_run = []
#     fcm_r_add_error_totals_run = []
#     cms_approx_factor_l_run = []
#     fcm_c_approx_factor_l_run = []
#     fcm_r_approx_factor_l_run = []
#     cms_add_error_l_run = []
#     fcm_c_add_error_l_run = []
#     fcm_r_add_error_l_run = []
#     cms_approx_factor_h_run = []
#     fcm_c_approx_factor_h_run = []
#     fcm_r_approx_factor_h_run = []
#     cms_add_error_h_run = []
#     fcm_c_add_error_h_run = []
#     fcm_r_add_error_h_run = []

#     for _ in range(5):
#         (
#             mean_approx_factor_cms_l,
#             total_add_error_cms_l,
#             mean_approx_factor_cms_h,
#             total_add_error_cms_h,
#             mean_approx_factor_fcm_c_l,
#             total_add_error_fcm_c_l,
#             mean_approx_factor_fcm_c_h,
#             total_add_error_fcm_c_h,
#             mean_approx_factor_fcm_r_l,
#             total_add_error_fcm_r_l,
#             mean_approx_factor_fcm_r_h,
#             total_add_error_fcm_r_h,
#         ) = run_fairness_census(
#             w=64,
#             d=d,
#         )
#         cms_approx_factor_diffs_run.append(mean_approx_factor_cms_l - mean_approx_factor_cms_h)
#         fcm_c_approx_factor_diffs_run.append(mean_approx_factor_fcm_c_l - mean_approx_factor_fcm_c_h)
#         fcm_r_approx_factor_diffs_run.append(mean_approx_factor_fcm_r_l - mean_approx_factor_fcm_r_h)
#         cms_add_error_totals_run.append(total_add_error_cms_h + total_add_error_cms_l)
#         fcm_c_add_error_totals_run.append(total_add_error_fcm_c_h + total_add_error_fcm_c_l)
#         fcm_r_add_error_totals_run.append(total_add_error_fcm_r_h + total_add_error_fcm_r_l)
#         cms_approx_factor_l_run.append(mean_approx_factor_cms_l)
#         cms_approx_factor_h_run.append(mean_approx_factor_cms_h)
#         cms_add_error_l_run.append(total_add_error_cms_l)
#         cms_add_error_h_run.append(total_add_error_cms_h)
#         fcm_c_approx_factor_l_run.append(mean_approx_factor_fcm_c_l)
#         fcm_c_approx_factor_h_run.append(mean_approx_factor_fcm_c_h)
#         fcm_c_add_error_l_run.append(total_add_error_fcm_c_l)
#         fcm_c_add_error_h_run.append(total_add_error_fcm_c_h)
#         fcm_r_approx_factor_l_run.append(mean_approx_factor_fcm_r_l)
#         fcm_r_approx_factor_h_run.append(mean_approx_factor_fcm_r_h)
#         fcm_r_add_error_l_run.append(total_add_error_fcm_r_l)
#         fcm_r_add_error_h_run.append(total_add_error_fcm_r_h)

#     cms_approx_factor_diffs.append(np.mean(cms_approx_factor_diffs_run))
#     fcm_c_approx_factor_diffs.append(np.mean(fcm_c_approx_factor_diffs_run))
#     fcm_r_approx_factor_diffs.append(np.mean(fcm_r_approx_factor_diffs_run))
#     cms_add_error_totals.append(np.mean(cms_add_error_totals_run))
#     fcm_c_add_error_totals.append(np.mean(fcm_c_add_error_totals_run))
#     fcm_r_add_error_totals.append(np.mean(fcm_r_add_error_totals_run))
#     cms_approx_factor_l.append(np.mean(cms_approx_factor_l_run))
#     cms_approx_factor_h.append(np.mean(cms_approx_factor_h_run))
#     cms_add_error_l.append(np.mean(cms_add_error_l_run))
#     cms_add_error_h.append(np.mean(cms_add_error_h_run))
#     fcm_c_approx_factor_l.append(np.mean(fcm_c_approx_factor_l_run))
#     fcm_c_approx_factor_h.append(np.mean(fcm_c_approx_factor_h_run))
#     fcm_c_add_error_l.append(np.mean(fcm_c_add_error_l_run))
#     fcm_c_add_error_h.append(np.mean(fcm_c_add_error_h_run))
#     fcm_r_approx_factor_l.append(np.mean(fcm_r_approx_factor_l_run))
#     fcm_r_approx_factor_h.append(np.mean(fcm_r_approx_factor_h_run))
#     fcm_r_add_error_l.append(np.mean(fcm_r_add_error_l_run))
#     fcm_r_add_error_h.append(np.mean(fcm_r_add_error_h_run))

# results = {
#     "cms_approx_factor_diffs": cms_approx_factor_diffs,
#     "fcm_c_approx_factor_diffs": fcm_c_approx_factor_diffs,
#     "fcm_r_approx_factor_diffs": fcm_r_approx_factor_diffs,
#     "cms_add_error_totals": cms_add_error_totals,
#     "fcm_c_add_error_totals": fcm_c_add_error_totals,
#     "fcm_r_add_error_totals": fcm_r_add_error_totals,
#     "cms_approx_factor_l": cms_approx_factor_l,
#     "fcm_c_approx_factor_l": fcm_c_approx_factor_l,
#     "fcm_r_approx_factor_l": fcm_r_approx_factor_l,
#     "cms_add_error_l": cms_add_error_l,
#     "fcm_c_add_error_l": fcm_c_add_error_l,
#     "fcm_r_add_error_l": fcm_r_add_error_l,
#     "cms_approx_factor_h": cms_approx_factor_h,
#     "fcm_c_approx_factor_h": fcm_c_approx_factor_h,
#     "fcm_r_approx_factor_h": fcm_r_approx_factor_h,
#     "cms_add_error_h": cms_add_error_h,
#     "fcm_c_add_error_h": fcm_c_add_error_h,
#     "fcm_r_add_error_h": fcm_r_add_error_h,
# }

# with open("results_census_varying_d.json", "w") as f:
#     json.dump(results, f, indent=4)

with open("results_census_varying_d.json", "r") as f:
    results = json.load(f)

cms_approx_factor_diffs = results["cms_approx_factor_diffs"]
fcm_c_approx_factor_diffs = results["fcm_c_approx_factor_diffs"]
fcm_r_approx_factor_diffs = results["fcm_r_approx_factor_diffs"]
cms_add_error_totals = results["cms_add_error_totals"]
fcm_c_add_error_totals = results["fcm_c_add_error_totals"]
fcm_r_add_error_totals = results["fcm_r_add_error_totals"]
cms_approx_factor_l = results["cms_approx_factor_l"]
fcm_c_approx_factor_l = results["fcm_c_approx_factor_l"]
fcm_r_approx_factor_l = results["fcm_r_approx_factor_l"]
cms_add_error_l = results["cms_add_error_l"]
fcm_c_add_error_l = results["fcm_c_add_error_l"]
fcm_r_add_error_l = results["fcm_r_add_error_l"]
cms_approx_factor_h = results["cms_approx_factor_h"]
fcm_c_approx_factor_h = results["fcm_c_approx_factor_h"]
fcm_r_approx_factor_h = results["fcm_r_approx_factor_h"]
cms_add_error_h = results["cms_add_error_h"]
fcm_c_add_error_h = results["fcm_c_add_error_h"]
fcm_r_add_error_h = results["fcm_r_add_error_h"]

plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(2, 11), cms_approx_factor_diffs, label="Count-Min Baseline", marker="o", markersize=10, linewidth=3)
plt.plot(range(2, 11), fcm_c_approx_factor_diffs, label="Fair-Count-Min", marker="s", markersize=10, linewidth=3)
plt.plot(range(2, 11), fcm_r_approx_factor_diffs, label="Row-Partitioning Baseline", marker="^", markersize=10, linewidth=3)
plt.xticks(ticks=range(2, 11), labels=[i for i in range(2, 11)])
plt.xlabel("d")
plt.ylabel("Mean Approx. Fact. Diff. (F - M)")
plt.legend()
# plt.title("Mean Approximation Factor Differences for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/census/mean_approx_factor_difference_plot_varying_d.png", bbox_inches="tight")


plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(2, 11), cms_add_error_totals, label="Count-Min Baseline", marker="o", markersize=10, linewidth=3)
plt.plot(range(2, 11), fcm_c_add_error_totals, label="Fair-Count-Min", marker="s", markersize=10, linewidth=3)
# plt.plot(range(2, 11), fcm_r_add_error_totals, label="Row-Partitioning Baseline", marker="^", markersize=10, linewidth=3)
plt.xticks(ticks=range(2, 11), labels=[i for i in range(2, 11)])
plt.xlabel("d")
plt.ylabel("Total Additive Error")
plt.legend()
plt.ylim(0, 9 * 10**6)
# plt.title("Price of Fairness w.r.t. Additive Error for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/census/price_of_fairness_plot_varying_d.png", bbox_inches="tight")


plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(2, 11), cms_approx_factor_l, label="Count-Min Baseline (l)", marker="o", markersize=10, linewidth=3)
plt.plot(range(2, 11), cms_approx_factor_h, label="Count-Min Baseline (h)", marker="o", markersize=10, linewidth=3)
plt.plot(range(2, 11), fcm_c_approx_factor_l, label="Fair-Count-Min (l)", marker="s", markersize=10, linewidth=3)
plt.plot(range(2, 11), fcm_c_approx_factor_h, label="Fair-Count-Min (h)", marker="s", markersize=10, linewidth=3)
plt.plot(
    range(2, 11), fcm_r_approx_factor_l, label="Row-Partitioning Baseline (l)", marker="^", markersize=10, linewidth=3
)
plt.plot(
    range(2, 11), fcm_r_approx_factor_h, label="Row-Partitioning Baseline (h)", marker="^", markersize=10, linewidth=3
)
plt.xticks(ticks=range(2, 11), labels=[i for i in range(2, 11)])
plt.xlabel("d")
plt.ylabel("Mean Approx. Fact. Abs. Values")
plt.legend()
# plt.title("Mean Approximation Factor for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/census/mean_approx_factor_absolute_values_plot_varying_d.png", bbox_inches="tight")


plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(2, 11), cms_add_error_l, label="Count-Min Baseline (l)", marker="o", markersize=10, linewidth=3)
plt.plot(range(2, 11), cms_add_error_h, label="Count-Min Baseline (h)", marker="o", markersize=10, linewidth=3)
plt.plot(range(2, 11), fcm_c_add_error_l, label="Fair-Count-Min (l)", marker="s", markersize=10, linewidth=3)
plt.plot(range(2, 11), fcm_c_add_error_h, label="Fair-Count-Min (h)", marker="s", markersize=10, linewidth=3)
# plt.plot(range(2, 11), fcm_r_add_error_l, label="Row-Partitioning Baseline (l)", marker="^", markersize=10, linewidth=3)
# plt.plot(range(2, 11), fcm_r_add_error_h, label="Row-Partitioning Baseline (h)", marker="^", markersize=10, linewidth=3)
plt.xticks(ticks=range(2, 11), labels=[i for i in range(2, 11)])
plt.xlabel("d")
plt.ylabel("Additive Absolute Error")
plt.legend()
# plt.title("Additive Error for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/census/additive_absolute_error_plot_varying_d.png", bbox_inches="tight")

# # --------------------------------------------------------------------------------------------------

# cms_construction_time_avg = []
# fcm_c_construction_time_avg = []
# fcm_r_construction_time_avg = []
# cms_query_time_avg = []
# fcm_c_query_time_avg = []
# fcm_r_query_time_avg = []
# n_l_label = []

# for cutoff in [i for i in range(2, 13)]:
#     cms_construction_time_run = []
#     fcm_c_construction_time_run = []
#     fcm_r_construction_time_run = []
#     cms_query_time_run = []
#     fcm_c_query_time_run = []
#     fcm_r_query_time_run = []
#     for _ in range(5):
#         (
#             cms_construction_time,
#             fcm_c_construction_time,
#             fcm_r_construction_time,
#             cms_query_time,
#             fcm_c_query_time,
#             fcm_r_query_time,
#             n_l,
#         ) = run_efficiency_google_books(
#             w=1024,
#             d=5,
#             cutoff=cutoff,
#         )

#         cms_construction_time_run.append(cms_construction_time)
#         fcm_c_construction_time_run.append(fcm_c_construction_time)
#         fcm_r_construction_time_run.append(fcm_r_construction_time)
#         cms_query_time_run.append(cms_query_time)
#         fcm_c_query_time_run.append(fcm_c_query_time)
#         fcm_r_query_time_run.append(fcm_r_query_time)
#         n_l_label.append(int(n_l))

#     cms_construction_time_avg.append(np.mean(cms_construction_time_run))
#     fcm_c_construction_time_avg.append(np.mean(fcm_c_construction_time_run))
#     fcm_r_construction_time_avg.append(np.mean(fcm_r_construction_time_run))
#     cms_query_time_avg.append(np.mean(cms_query_time_run))
#     fcm_c_query_time_avg.append(np.mean(fcm_c_query_time_run))
#     fcm_r_query_time_avg.append(np.mean(fcm_r_query_time_run))

# results_efficiency = {
#     "cms_construction_time_avg": cms_construction_time_avg,
#     "fcm_c_construction_time_avg": fcm_c_construction_time_avg,
#     "fcm_r_construction_time_avg": fcm_r_construction_time_avg,
#     "cms_query_time_avg": cms_query_time_avg,
#     "fcm_c_query_time_avg": fcm_c_query_time_avg,
#     "fcm_r_query_time_avg": fcm_r_query_time_avg,
#     "n_l_label": [n_l_label[5 * i] for i in range(11)],
# }

# with open("results_efficiency_google_books_varying_n_l.json", "w") as f:
#     json.dump(results_efficiency, f, indent=4)

with open("results_efficiency_google_books_varying_n_l.json", "r") as f:
    results_efficiency = json.load(f)

cms_construction_time_avg = results_efficiency["cms_construction_time_avg"]
fcm_c_construction_time_avg = results_efficiency["fcm_c_construction_time_avg"]
fcm_r_construction_time_avg = results_efficiency["fcm_r_construction_time_avg"]
cms_query_time_avg = results_efficiency["cms_query_time_avg"]
fcm_c_query_time_avg = results_efficiency["fcm_c_query_time_avg"]
fcm_r_query_time_avg = results_efficiency["fcm_r_query_time_avg"]
n_l_label = results_efficiency["n_l_label"]
n_l_label = [round(x / 10**5, 1) for x in n_l_label]


n_l = range(1, 12)
plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(n_l, cms_construction_time_avg, label="Count-Min Baseline", marker="o", markersize=10, linewidth=3)
plt.plot(
    n_l,
    fcm_c_construction_time_avg,
    label="Fair-Count-Min",
    marker="s",
    markersize=10,
    linewidth=3,
)
# plt.plot(
#     n_l, fcm_r_construction_time_avg, label="Row-Partitioning Baseline", marker="^", markersize=10, linewidth=3
# )
# plt.xscale("log")
plt.xticks(ticks=n_l, labels=n_l_label)
plt.xlabel("$n_{l}$ ×100000")
plt.ylabel("Average Construction Time (s)")
plt.legend()
# plt.title("Average Construction Times for Different Count-Min Sketch Variants")
plt.grid(True)
plt.ylim((10, 20))
plt.savefig("plots/google_books/construction_time_plot_varying_n_l.png", bbox_inches="tight")

plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(n_l, cms_query_time_avg, label="Count-Min Baseline", marker="o", markersize=10, linewidth=3)
plt.plot(n_l, fcm_c_query_time_avg, label="Fair-Count-Min", marker="s", markersize=10, linewidth=3)
# plt.plot(n_l, fcm_r_query_time_avg, label="Row-Partitioning Baseline", marker="^", markersize=10, linewidth=3)
# plt.xscale("log")
plt.xticks(ticks=n_l, labels=n_l_label)
plt.xlabel("$n_{l}$ ×100000")
plt.ylabel("Average Query Time (s)")
plt.ylim((0, 5e-6))
plt.legend()
# plt.title("Average Query Times for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/google_books/query_time_plot_varying_n_l.png", bbox_inches="tight")

# cms_construction_time_avg = []
# fcm_c_construction_time_avg = []
# fcm_r_construction_time_avg = []
# cms_query_time_avg = []
# fcm_c_query_time_avg = []
# fcm_r_query_time_avg = []

# for w in [2**i for i in range(10, 17)]:
#     cms_construction_time_run = []
#     fcm_c_construction_time_run = []
#     fcm_r_construction_time_run = []
#     cms_query_time_run = []
#     fcm_c_query_time_run = []
#     fcm_r_query_time_run = []
#     for _ in range(5):
#         (
#             cms_construction_time,
#             fcm_c_construction_time,
#             fcm_r_construction_time,
#             cms_query_time,
#             fcm_c_query_time,
#             fcm_r_query_time,
#             n_l,
#         ) = run_efficiency_google_books(
#             w=w,
#             d=5,
#             cutoff=10,
#         )
#         cms_construction_time_run.append(cms_construction_time)
#         fcm_c_construction_time_run.append(fcm_c_construction_time)
#         fcm_r_construction_time_run.append(fcm_r_construction_time)
#         cms_query_time_run.append(cms_query_time)
#         fcm_c_query_time_run.append(fcm_c_query_time)
#         fcm_r_query_time_run.append(fcm_r_query_time)

#     cms_construction_time_avg.append(np.mean(cms_construction_time_run))
#     fcm_c_construction_time_avg.append(np.mean(fcm_c_construction_time_run))
#     fcm_r_construction_time_avg.append(np.mean(fcm_r_construction_time_run))
#     cms_query_time_avg.append(np.mean(cms_query_time_run))
#     fcm_c_query_time_avg.append(np.mean(fcm_c_query_time_run))
#     fcm_r_query_time_avg.append(np.mean(fcm_r_query_time_run))


# results_efficiency = {
#     "cms_construction_time_avg": cms_construction_time_avg,
#     "fcm_c_construction_time_avg": fcm_c_construction_time_avg,
#     "fcm_r_construction_time_avg": fcm_r_construction_time_avg,
#     "cms_query_time_avg": cms_query_time_avg,
#     "fcm_c_query_time_avg": fcm_c_query_time_avg,
#     "fcm_r_query_time_avg": fcm_r_query_time_avg,
# }

# with open("results_efficiency_google_books_varying_w.json", "w") as f:
#     json.dump(results_efficiency, f, indent=4)

with open("results_efficiency_google_books_varying_w.json", "r") as f:
    results_efficiency = json.load(f)

cms_construction_time_avg = results_efficiency["cms_construction_time_avg"]
fcm_c_construction_time_avg = results_efficiency["fcm_c_construction_time_avg"]
fcm_r_construction_time_avg = results_efficiency["fcm_r_construction_time_avg"]
cms_query_time_avg = results_efficiency["cms_query_time_avg"]
fcm_c_query_time_avg = results_efficiency["fcm_c_query_time_avg"]
fcm_r_query_time_avg = results_efficiency["fcm_r_query_time_avg"]

plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(1, 8), cms_construction_time_avg, label="Count-Min Baseline", marker="o", markersize=10, linewidth=3)
plt.plot(
    range(1, 8),
    fcm_c_construction_time_avg,
    label="Fair-Count-Min",
    marker="s",
    markersize=10,
    linewidth=3,
)
# plt.plot(
#     range(1, 8),
#     fcm_r_construction_time_avg,
#     label="Row-Partitioning Baseline",
#     marker="^",
#     markersize=10,
#     linewidth=3,
# )
plt.xticks(ticks=range(1, 8), labels=[2**i for i in range(10, 17)])
plt.xlabel("w")
plt.ylim((10, 20))
plt.ylabel("Average Construction Time (s)")
plt.legend()
# plt.title("Average Construction Times for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/google_books/construction_time_plot_varying_w.png", bbox_inches="tight")

plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(1, 8), cms_query_time_avg, label="Count-Min Baseline", marker="o", markersize=10, linewidth=3)
plt.plot(range(1, 8), fcm_c_query_time_avg, label="Fair-Count-Min", marker="s", markersize=10, linewidth=3)
# plt.plot(range(1, 8), fcm_r_query_time_avg, label="Row-Partitioning Baseline", marker="^", markersize=10, linewidth=3)
plt.xticks(ticks=range(1, 8), labels=[2**i for i in range(10, 17)])
plt.xlabel("w")
plt.ylabel("Average Query Time (s)")
plt.ylim((0, 5e-6))
plt.legend()
# plt.title("Average Query Times for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/google_books/query_time_plot_varying_w.png", bbox_inches="tight")

# cms_construction_time_avg = []
# fcm_c_construction_time_avg = []
# fcm_r_construction_time_avg = []
# cms_query_time_avg = []
# fcm_c_query_time_avg = []
# fcm_r_query_time_avg = []

# for d in [i for i in range(2, 11)]:
#     cms_construction_time_run = []
#     fcm_c_construction_time_run = []
#     fcm_r_construction_time_run = []
#     cms_query_time_run = []
#     fcm_c_query_time_run = []
#     fcm_r_query_time_run = []
#     for _ in range(5):
#         (
#             cms_construction_time,
#             fcm_c_construction_time,
#             fcm_r_construction_time,
#             cms_query_time,
#             fcm_c_query_time,
#             fcm_r_query_time,
#             n_l,
#         ) = run_efficiency_google_books(
#             w=1024,
#             d=d,
#             cutoff=10,
#         )
#         cms_construction_time_run.append(cms_construction_time)
#         fcm_c_construction_time_run.append(fcm_c_construction_time)
#         fcm_r_construction_time_run.append(fcm_r_construction_time)
#         cms_query_time_run.append(cms_query_time)
#         fcm_c_query_time_run.append(fcm_c_query_time)
#         fcm_r_query_time_run.append(fcm_r_query_time)

#     cms_construction_time_avg.append(np.mean(cms_construction_time_run))
#     fcm_c_construction_time_avg.append(np.mean(fcm_c_construction_time_run))
#     fcm_r_construction_time_avg.append(np.mean(fcm_r_construction_time_run))
#     cms_query_time_avg.append(np.mean(cms_query_time_run))
#     fcm_c_query_time_avg.append(np.mean(fcm_c_query_time_run))
#     fcm_r_query_time_avg.append(np.mean(fcm_r_query_time_run))

# results_efficiency = {
#     "cms_construction_time_avg": cms_construction_time_avg,
#     "fcm_c_construction_time_avg": fcm_c_construction_time_avg,
#     "fcm_r_construction_time_avg": fcm_r_construction_time_avg,
#     "cms_query_time_avg": cms_query_time_avg,
#     "fcm_c_query_time_avg": fcm_c_query_time_avg,
#     "fcm_r_query_time_avg": fcm_r_query_time_avg,
# }

# with open("results_efficiency_google_books_varying_d.json", "w") as f:
#     json.dump(results_efficiency, f, indent=4)

with open("results_efficiency_google_books_varying_d.json", "r") as f:
    results_efficiency = json.load(f)

cms_construction_time_avg = results_efficiency["cms_construction_time_avg"]
fcm_c_construction_time_avg = results_efficiency["fcm_c_construction_time_avg"]
fcm_r_construction_time_avg = results_efficiency["fcm_r_construction_time_avg"]
cms_query_time_avg = results_efficiency["cms_query_time_avg"]
fcm_c_query_time_avg = results_efficiency["fcm_c_query_time_avg"]
fcm_r_query_time_avg = results_efficiency["fcm_r_query_time_avg"]

plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(2, 11), cms_construction_time_avg, label="Count-Min Baseline", marker="o", markersize=10, linewidth=3)
plt.plot(
    range(2, 11),
    fcm_c_construction_time_avg,
    label="Fair-Count-Min",
    marker="s",
    markersize=10,
    linewidth=3,
)
# plt.plot(
#     range(2, 11),
#     fcm_r_construction_time_avg,
#     label="Row-Partitioning Baseline",
#     marker="^",
#     markersize=10,
#     linewidth=3,
# )
plt.xticks(ticks=range(2, 11), labels=[i for i in range(2, 11)])
plt.xlabel("d")
plt.ylabel("Average Construction Time (s)")
plt.legend()
# plt.title("Average Construction Times for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/google_books/construction_time_plot_varying_d.png", bbox_inches="tight")

plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(2, 11), cms_query_time_avg, label="Count-Min Baseline", marker="o", markersize=10, linewidth=3)
plt.plot(range(2, 11), fcm_c_query_time_avg, label="Fair-Count-Min", marker="s", markersize=10, linewidth=3)
# plt.plot(range(2, 11), fcm_r_query_time_avg, label="Row-Partitioning Baseline", marker="^", markersize=10, linewidth=3)
plt.xticks(ticks=range(2, 11), labels=[i for i in range(2, 11)])
plt.xlabel("d")
plt.ylabel("Average Query Time (s)")
plt.legend()
# plt.title("Average Query Times for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/google_books/query_time_plot_varying_d.png", bbox_inches="tight")

# # --------------------------------------------------------------------------------------------------

# cms_construction_time_avg = []
# fcm_c_construction_time_avg = []
# fcm_r_construction_time_avg = []
# cms_query_time_avg = []
# fcm_c_query_time_avg = []
# fcm_r_query_time_avg = []

# for w in [2**i for i in range(10, 17)]:
#     cms_construction_time_run = []
#     fcm_c_construction_time_run = []
#     fcm_r_construction_time_run = []
#     cms_query_time_run = []
#     fcm_c_query_time_run = []
#     fcm_r_query_time_run = []
#     for j in range(5):
#         (
#             cms_construction_time,
#             fcm_c_construction_time,
#             fcm_r_construction_time,
#             cms_query_time,
#             fcm_c_query_time,
#             fcm_r_query_time,
#         ) = run_efficiency_census(
#             w=w,
#             d=5,
#         )
#         cms_construction_time_run.append(cms_construction_time)
#         fcm_c_construction_time_run.append(fcm_c_construction_time)
#         fcm_r_construction_time_run.append(fcm_r_construction_time)
#         cms_query_time_run.append(cms_query_time)
#         fcm_c_query_time_run.append(fcm_c_query_time)
#         fcm_r_query_time_run.append(fcm_r_query_time)

#     cms_construction_time_avg.append(np.mean(cms_construction_time_run))
#     fcm_c_construction_time_avg.append(np.mean(fcm_c_construction_time_run))
#     fcm_r_construction_time_avg.append(np.mean(fcm_r_construction_time_run))
#     cms_query_time_avg.append(np.mean(cms_query_time_run))
#     fcm_c_query_time_avg.append(np.mean(fcm_c_query_time_run))
#     fcm_r_query_time_avg.append(np.mean(fcm_r_query_time_run))


# results_efficiency = {
#     "cms_construction_time_avg": cms_construction_time_avg,
#     "fcm_c_construction_time_avg": fcm_c_construction_time_avg,
#     "fcm_r_construction_time_avg": fcm_r_construction_time_avg,
#     "cms_query_time_avg": cms_query_time_avg,
#     "fcm_c_query_time_avg": fcm_c_query_time_avg,
#     "fcm_r_query_time_avg": fcm_r_query_time_avg,
# }

# with open("results_efficiency_census_varying_w.json", "w") as f:
#     json.dump(results_efficiency, f, indent=4)

with open("results_efficiency_census_varying_w.json", "r") as f:
    results_efficiency = json.load(f)

cms_construction_time_avg = results_efficiency["cms_construction_time_avg"]
fcm_c_construction_time_avg = results_efficiency["fcm_c_construction_time_avg"]
fcm_r_construction_time_avg = results_efficiency["fcm_r_construction_time_avg"]
cms_query_time_avg = results_efficiency["cms_query_time_avg"]
fcm_c_query_time_avg = results_efficiency["fcm_c_query_time_avg"]
fcm_r_query_time_avg = results_efficiency["fcm_r_query_time_avg"]

plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(1, 8), cms_construction_time_avg, label="Count-Min Baseline", marker="o", markersize=10, linewidth=3)
plt.plot(
    range(1, 8),
    fcm_c_construction_time_avg,
    label="Fair-Count-Min",
    marker="s",
    markersize=10,
    linewidth=3,
)
# plt.plot(
#     range(1, 8),
#     fcm_r_construction_time_avg,
#     label="Row-Partitioning Baseline",
#     marker="^",
#     markersize=10,
#     linewidth=3,
# )
plt.xticks(ticks=range(1, 8), labels=[2**i for i in range(10, 17)])
plt.xlabel("w")
plt.ylabel("Average Construction Time (s)")
plt.legend()
# plt.title("Average Construction Times for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/census/construction_time_plot_varying_w.png", bbox_inches="tight")

plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(1, 8), cms_query_time_avg, label="Count-Min Baseline", marker="o", markersize=10, linewidth=3)
plt.plot(range(1, 8), fcm_c_query_time_avg, label="Fair-Count-Min", marker="s", markersize=10, linewidth=3)
# plt.plot(range(1, 8), fcm_r_query_time_avg, label="Row-Partitioning Baseline", marker="^", markersize=10, linewidth=3)
plt.xticks(ticks=range(1, 8), labels=[2**i for i in range(10, 17)])
plt.xlabel("w")
plt.ylabel("Average Query Time (s)")
plt.legend()
# plt.title("Average Query Times for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/census/query_time_plot_varying_w.png", bbox_inches="tight")

# cms_construction_time_avg = []
# fcm_c_construction_time_avg = []
# fcm_r_construction_time_avg = []
# cms_query_time_avg = []
# fcm_c_query_time_avg = []
# fcm_r_query_time_avg = []

# for d in [i for i in range(2, 11)]:
#     cms_construction_time_run = []
#     fcm_c_construction_time_run = []
#     fcm_r_construction_time_run = []
#     cms_query_time_run = []
#     fcm_c_query_time_run = []
#     fcm_r_query_time_run = []
#     for j in range(5):
#         (
#             cms_construction_time,
#             fcm_c_construction_time,
#             fcm_r_construction_time,
#             cms_query_time,
#             fcm_c_query_time,
#             fcm_r_query_time,
#         ) = run_efficiency_census(
#             w=64,
#             d=d,
#         )
#         cms_construction_time_run.append(cms_construction_time)
#         fcm_c_construction_time_run.append(fcm_c_construction_time)
#         fcm_r_construction_time_run.append(fcm_r_construction_time)
#         cms_query_time_run.append(cms_query_time)
#         fcm_c_query_time_run.append(fcm_c_query_time)
#         fcm_r_query_time_run.append(fcm_r_query_time)

#     cms_construction_time_avg.append(np.mean(cms_construction_time_run))
#     fcm_c_construction_time_avg.append(np.mean(fcm_c_construction_time_run))
#     fcm_r_construction_time_avg.append(np.mean(fcm_r_construction_time_run))
#     cms_query_time_avg.append(np.mean(cms_query_time_run))
#     fcm_c_query_time_avg.append(np.mean(fcm_c_query_time_run))
#     fcm_r_query_time_avg.append(np.mean(fcm_r_query_time_run))

# results_efficiency = {
#     "cms_construction_time_avg": cms_construction_time_avg,
#     "fcm_c_construction_time_avg": fcm_c_construction_time_avg,
#     "fcm_r_construction_time_avg": fcm_r_construction_time_avg,
#     "cms_query_time_avg": cms_query_time_avg,
#     "fcm_c_query_time_avg": fcm_c_query_time_avg,
#     "fcm_r_query_time_avg": fcm_r_query_time_avg,
# }

# with open("results_efficiency_census_varying_d.json", "w") as f:
#     json.dump(results_efficiency, f, indent=4)

with open("results_efficiency_census_varying_d.json", "r") as f:
    results_efficiency = json.load(f)

cms_construction_time_avg = results_efficiency["cms_construction_time_avg"]
fcm_c_construction_time_avg = results_efficiency["fcm_c_construction_time_avg"]
fcm_r_construction_time_avg = results_efficiency["fcm_r_construction_time_avg"]
cms_query_time_avg = results_efficiency["cms_query_time_avg"]
fcm_c_query_time_avg = results_efficiency["fcm_c_query_time_avg"]
fcm_r_query_time_avg = results_efficiency["fcm_r_query_time_avg"]

plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(2, 11), cms_construction_time_avg, label="Count-Min Baseline", marker="o", markersize=10, linewidth=3)
plt.plot(
    range(2, 11),
    fcm_c_construction_time_avg,
    label="Fair-Count-Min",
    marker="s",
    markersize=10,
    linewidth=3,
)
# plt.plot(
#     range(2, 11),
#     fcm_r_construction_time_avg,
#     label="Row-Partitioning Baseline",
#     marker="^",
#     markersize=10,
#     linewidth=3,
# )
plt.xticks(ticks=range(2, 11), labels=[i for i in range(2, 11)])
plt.xlabel("d")
plt.ylabel("Average Construction Time (s)")
plt.legend()
# plt.title("Average Construction Times for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/census/construction_time_plot_varying_d.png", bbox_inches="tight")

plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 20})
plt.plot(range(2, 11), cms_query_time_avg, label="Count-Min Baseline", marker="o", markersize=10, linewidth=3)
plt.plot(range(2, 11), fcm_c_query_time_avg, label="Fair-Count-Min", marker="s", markersize=10, linewidth=3)
# plt.plot(range(2, 11), fcm_r_query_time_avg, label="Row-Partitioning Baseline", marker="^", markersize=10, linewidth=3)
plt.xticks(ticks=range(2, 11), labels=[i for i in range(2, 11)])
plt.xlabel("d")
plt.ylabel("Average Query Time (s)")
plt.legend()
# plt.title("Average Query Times for Different Count-Min Sketch Variants")
plt.grid(True)
plt.savefig("plots/census/query_time_plot_varying_d.png", bbox_inches="tight")


# 1 row + multiple row based on the similation just for the column-based for google_books. should be at zero. Additve show the sum of the two errors (1 row should be zero). try normal and zipf for the frequency distribution. add error bars for the multiplicative erros difference.m

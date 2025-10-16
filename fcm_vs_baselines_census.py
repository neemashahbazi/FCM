import hashlib
import numpy as np
import random
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier

plt.rcParams["font.family"] = "serif"


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


class LearnedCountMinSketch:
    def __init__(self, width, depth, heavy_hitter_fraction=0.3):
        self.width = width
        self.depth = depth
        self.heavy_hitter_fraction = heavy_hitter_fraction
        self.heavy_width = int(width * heavy_hitter_fraction)
        self.normal_width = width - self.heavy_width
        self.heavy_table = [[0] * self.heavy_width for _ in range(depth)] if self.heavy_width > 0 else None
        self.normal_table = [[0] * self.normal_width for _ in range(depth)]
        self.hash_seeds = [random.randint(0, 2**32 - 1) for _ in range(depth)]
        self.heavy_hitters = set()
        self.classifier = None

    def _hash(self, item, seed, width):
        return int(hashlib.md5((str(seed) + str(item)).encode("utf-8")).hexdigest(), 16) % width

    def train_heavy_hitter_predictor(self, sample_stream, sample_groups=None, threshold_percentile=90):
        freq_counts = defaultdict(int)
        for item in sample_stream:
            if isinstance(item, tuple):
                freq_counts[item[0]] += 1
            else:
                freq_counts[item] += 1
        if not freq_counts:
            return
        frequencies = list(freq_counts.values())
        threshold = np.percentile(frequencies, threshold_percentile)
        X = []
        y = []
        for item, count in freq_counts.items():
            features = self._extract_features(item)
            X.append(features)
            y.append(1 if count >= threshold else 0)
        if len(set(y)) > 1 and len(X) > 10:
            self.classifier = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
            self.classifier.fit(X, y)
            for item, count in freq_counts.items():
                if count >= threshold:
                    self.heavy_hitters.add(item)
        else:
            sorted_items = sorted(freq_counts.items(), key=lambda x: x[1], reverse=True)
            k = max(1, int(len(sorted_items) * (1 - threshold_percentile / 100)))
            for item, _ in sorted_items[:k]:
                self.heavy_hitters.add(item)

    def _extract_features(self, item):
        item_str = str(item)
        features = []
        for i in range(5):
            hash_val = int(hashlib.md5((str(i) + item_str).encode()).hexdigest(), 16)
            features.append(hash_val % 1000)
        features.append(len(item_str))
        features.append(sum(ord(c) for c in item_str[:10]))
        return features

    def _is_heavy_hitter(self, item):
        if item in self.heavy_hitters:
            return True
        if self.classifier is not None:
            features = self._extract_features(item)
            prediction = self.classifier.predict([features])[0]
            return prediction == 1
        return False

    def add(self, item, group=None):
        if isinstance(item, tuple):
            item = item[0]
        if self.heavy_width > 0 and self._is_heavy_hitter(item):
            for i in range(self.depth):
                index = self._hash(item, self.hash_seeds[i], self.heavy_width)
                self.heavy_table[i][index] += 1
        else:
            for i in range(self.depth):
                index = self._hash(item, self.hash_seeds[i], self.normal_width)
                self.normal_table[i][index] += 1

    def count(self, item, group=None):
        if isinstance(item, tuple):
            item = item[0]
        if self.heavy_width > 0 and self._is_heavy_hitter(item):
            min_count = float("inf")
            for i in range(self.depth):
                index = self._hash(item, self.hash_seeds[i], self.heavy_width)
                min_count = min(min_count, self.heavy_table[i][index])
        else:
            min_count = float("inf")
            for i in range(self.depth):
                index = self._hash(item, self.hash_seeds[i], self.normal_width)
                min_count = min(min_count, self.normal_table[i][index])
        return min_count


class CountMinSketchConservativeUpdate:
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.table = [[0] * width for _ in range(depth)]
        self.hash_seeds = [random.randint(0, 2**32 - 1) for _ in range(depth)]

    def _hash(self, item, seed):
        return int(hashlib.md5((str(seed) + str(item)).encode("utf-8")).hexdigest(), 16) % self.width

    def add(self, item):
        indices = [self._hash(item, self.hash_seeds[i]) for i in range(self.depth)]
        current_counts = [self.table[i][indices[i]] for i in range(self.depth)]
        min_count = min(current_counts)
        for i in range(self.depth):
            if self.table[i][indices[i]] == min_count:
                self.table[i][indices[i]] += 1

    def count(self, item):
        min_count = float("inf")
        for i in range(self.depth):
            index = self._hash(item, self.hash_seeds[i])
            min_count = min(min_count, self.table[i][index])
        return min_count


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


stream, n, n_l = read_census()


def run_fairness_census(w, d):
    n_h = n - n_l
    w_l = find_w_l(n_l, n_h, d, w)
    d_l = find_d_l(n_l, n_h, d, w)
    true_counts_h = defaultdict(int)
    true_counts_l = defaultdict(int)
    N_l = 0
    N = len(stream)
    for element, group in stream:
        (true_counts_l if group == "l" else true_counts_h)[element] += 1
        if group == "l":
            N_l += 1
    N_h = N - N_l
    cms = CountMinSketch(w, d)
    fcm_c = FairCountMinColumnBased(w, d, [{"l": (0, w_l), "h": (w_l, w)} for _ in range(d)])
    fcm_r = FairCountMinRowBased(w, d_l, d - d_l)
    lcms = LearnedCountMinSketch(w, d, heavy_hitter_fraction=0.3)
    sample_size = int(len(stream) * 0.2)
    sample_stream = stream[:sample_size]
    lcms.train_heavy_hitter_predictor(sample_stream)
    cmscu = CountMinSketchConservativeUpdate(w, d)
    for element, group in stream:
        cms.add(element)
        fcm_c.add(element, group)
        fcm_r.add(element, group)
        lcms.add(element, group)
        cmscu.add(element)

    mean_approx_factor_cms_l, total_add_error_cms_l = calculate_errors(true_counts_l, cms)
    mean_approx_factor_cms_h, total_add_error_cms_h = calculate_errors(true_counts_h, cms)
    mean_approx_factor_fcm_c_l, total_add_error_fcm_c_l = calculate_errors(true_counts_l, fcm_c, "l")
    mean_approx_factor_fcm_c_h, total_add_error_fcm_c_h = calculate_errors(true_counts_h, fcm_c, "h")
    mean_approx_factor_l, total_add_error_l = calculate_errors(true_counts_l, lcms)
    mean_approx_factor_h, total_add_error_h = calculate_errors(true_counts_h, lcms)
    mean_approx_factor_cmscu_l, total_add_error_cmscu_l = calculate_errors(true_counts_l, cmscu)
    mean_approx_factor_cmscu_h, total_add_error_cmscu_h = calculate_errors(true_counts_h, cmscu)
    mean_approx_factor_fcm_r_l, total_add_error_fcm_r_l = calculate_errors(true_counts_l, fcm_r, "l")
    mean_approx_factor_fcm_r_h, total_add_error_fcm_r_h = calculate_errors(true_counts_h, fcm_r, "h")
    print(
        "======================== Census, ",
        f"n={n}, n_l={n_l}, n_h={n_h}, N={N}, N_l={N_l}, N_h={N_h}, w={w}, w_l={w_l}, w_h={w - w_l}, d={d}",
        "========================",
    )
    print(
        f"CountMinSketch (Group l) - Mean Approximation Factor: {mean_approx_factor_cms_l}, Additive Error: {total_add_error_cms_l}"
    )
    print(
        f"CountMinSketch (Group h) - Mean Approximation Factor: {mean_approx_factor_cms_h}, Additive Error: {total_add_error_cms_h}"
    )
    print(
        f"CountMinSketchConservativeUpdate (Group l) - Mean Approximation Factor: {mean_approx_factor_cmscu_l}, Additive Error: {total_add_error_cmscu_l}"
    )
    print(
        f"CountMinSketchConservativeUpdate (Group h) - Mean Approximation Factor: {mean_approx_factor_cmscu_h}, Additive Error: {total_add_error_cmscu_h}"
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
    print(
        f"LearnedCountMinSketch (Group l) - Mean Approximation Factor: {mean_approx_factor_l}, Additive Error: {total_add_error_l}"
    )
    print(
        f"LearnedCountMinSketch (Group h) - Mean Approximation Factor: {mean_approx_factor_h}, Additive Error: {total_add_error_h}"
    )
    print("============================================")
    return (
        mean_approx_factor_cms_l,
        total_add_error_cms_l,
        mean_approx_factor_cms_h,
        total_add_error_cms_h,
        mean_approx_factor_cmscu_l,
        total_add_error_cmscu_l,
        mean_approx_factor_cmscu_h,
        total_add_error_cmscu_h,
        mean_approx_factor_fcm_c_l,
        total_add_error_fcm_c_l,
        mean_approx_factor_fcm_c_h,
        total_add_error_fcm_c_h,
        mean_approx_factor_l,
        total_add_error_l,
        mean_approx_factor_h,
        total_add_error_h,
        mean_approx_factor_fcm_r_l,
        total_add_error_fcm_r_l,
        mean_approx_factor_fcm_r_h,
        total_add_error_fcm_r_h,
    )


def run_efficiency_census(w, d):
    n_h = n - n_l
    w_l = find_w_l(n_l, n_h, d, w)
    start_time = time.time()
    fcm_c = FairCountMinColumnBased(w, d, [{"l": (0, w_l), "h": (w_l, w)} for _ in range(d)])
    for element, group in stream:
        fcm_c.add(element, group)
    end_time = time.time()
    fcm_c_construction_time = end_time - start_time
    start_time = time.time()
    cms = CountMinSketch(w, d)
    for element, group in stream:
        cms.add(element)
    end_time = time.time()
    cms_construction_time = end_time - start_time
    start_time = time.time()
    cmscu = CountMinSketchConservativeUpdate(w, d)
    for element, group in stream:
        cmscu.add(element)
    end_time = time.time()
    cmscu_construction_time = end_time - start_time
    start_time = time.time()
    lcms = LearnedCountMinSketch(w, d, heavy_hitter_fraction=0.3)
    sample_size = int(len(stream) * 0.2)
    sample_stream = stream[:sample_size]
    lcms.train_heavy_hitter_predictor(sample_stream)
    for element, group in stream:
        lcms.add(element, group)
    end_time = time.time()
    lcms_construction_time = end_time - start_time
    cms_query_time = []
    for element, group in stream[:1000]:
        start_time = time.time()
        cms.count(element)
        end_time = time.time()
        cms_query_time.append(end_time - start_time)
    cmscu_query_time = []
    for element, group in stream[:1000]:
        start_time = time.time()
        cmscu.count(element)
        end_time = time.time()
        cmscu_query_time.append(end_time - start_time)
    fcm_c_query_time = []
    for element, group in stream[:1000]:
        start_time = time.time()
        fcm_c.count(element, group)
        end_time = time.time()
        fcm_c_query_time.append(end_time - start_time)
    lcms_query_time = []
    for element, group in stream[:1000]:
        start_time = time.time()
        lcms.count(element, group)
        end_time = time.time()
        lcms_query_time.append(end_time - start_time)
    cms_query_time = np.mean(cms_query_time)
    cmscu_query_time = np.mean(cmscu_query_time)
    fcm_c_query_time = np.mean(fcm_c_query_time)
    lcms_query_time = np.mean(lcms_query_time)

    print("CountMinSketch Construction Time:", cms_construction_time)
    print("CountMinSketchConservativeUpdate Construction Time:", cmscu_construction_time)
    print("FairCountMinColumnBased Construction Time:", fcm_c_construction_time)
    print("LearnedCountMinSketch Construction Time:", lcms_construction_time)
    print("CountMinSketch Query Time:", cms_query_time)
    print("CountMinSketchConservativeUpdate Query Time:", cmscu_query_time)
    print("FairCountMinColumnBased Query Time:", fcm_c_query_time)
    print("LearnedCountMinSketch Query Time:", lcms_query_time)
    print("============================================")
    return (
        cms_construction_time,
        cmscu_construction_time,
        fcm_c_construction_time,
        lcms_construction_time,
        cms_query_time,
        cmscu_query_time,
        fcm_c_query_time,
        lcms_query_time,
    )


cms_approx_factor_diffs = []
cmscu_approx_factor_diffs = []
fcm_c_approx_factor_diffs = []
lcms_approx_factor_diffs = []
cms_add_error_totals = []
cmscu_add_error_totals = []
fcm_c_add_error_totals = []
lcms_add_error_totals = []
cms_approx_factor_l = []
cmscu_approx_factor_l = []
fcm_c_approx_factor_l = []
lcms_approx_factor_l = []
cms_add_error_l = []
cmscu_add_error_l = []
fcm_c_add_error_l = []
lcms_add_error_l = []
cms_approx_factor_h = []
cmscu_approx_factor_h = []
fcm_c_approx_factor_h = []
lcms_approx_factor_h = []
cms_add_error_h = []
cmscu_add_error_h = []
fcm_c_add_error_h = []
lcms_add_error_h = []
fcm_r_approx_factor_diffs = []
fcm_r_add_error_totals = []
fcm_r_approx_factor_l = []
fcm_r_add_error_l = []
fcm_r_approx_factor_h = []
fcm_r_add_error_h = []


for w in [2**i for i in range(3, 9)]:
    cms_approx_factor_diffs_run = []
    cmscu_approx_factor_diffs_run = []
    fcm_c_approx_factor_diffs_run = []
    lcms_approx_factor_diffs_run = []
    cms_add_error_totals_run = []
    cmscu_add_error_totals_run = []
    fcm_c_add_error_totals_run = []
    lcms_add_error_totals_run = []
    cms_approx_factor_l_run = []
    cmscu_approx_factor_l_run = []
    fcm_c_approx_factor_l_run = []
    lcms_approx_factor_l_run = []
    cms_add_error_l_run = []
    cmscu_add_error_l_run = []
    fcm_c_add_error_l_run = []
    lcms_add_error_l_run = []
    cms_approx_factor_h_run = []
    cmscu_approx_factor_h_run = []
    fcm_c_approx_factor_h_run = []
    lcms_approx_factor_h_run = []
    cms_add_error_h_run = []
    cmscu_add_error_h_run = []
    fcm_c_add_error_h_run = []
    lcms_add_error_h_run = []
    fcm_r_approx_factor_diffs_run = []
    fcm_r_add_error_totals_run = []
    fcm_r_approx_factor_l_run = []
    fcm_r_add_error_l_run = []
    fcm_r_approx_factor_h_run = []
    fcm_r_add_error_h_run = []

    for _ in range(5):
        (
            mean_approx_factor_cms_l,
            total_add_error_cms_l,
            mean_approx_factor_cms_h,
            total_add_error_cms_h,
            mean_approx_factor_cmscu_l,
            total_add_error_cmscu_l,
            mean_approx_factor_cmscu_h,
            total_add_error_cmscu_h,
            mean_approx_factor_fcm_c_l,
            total_add_error_fcm_c_l,
            mean_approx_factor_fcm_c_h,
            total_add_error_fcm_c_h,
            mean_approx_factor_l,
            total_add_error_l,
            mean_approx_factor_h,
            total_add_error_h,
            mean_approx_factor_fcm_r_l,
            total_add_error_fcm_r_l,
            mean_approx_factor_fcm_r_h,
            total_add_error_fcm_r_h,
        ) = run_fairness_census(w=w, d=5)

        cms_approx_factor_diffs_run.append(mean_approx_factor_cms_l - mean_approx_factor_cms_h)
        cmscu_approx_factor_diffs_run.append(mean_approx_factor_cmscu_l - mean_approx_factor_cmscu_h)
        fcm_c_approx_factor_diffs_run.append(mean_approx_factor_fcm_c_l - mean_approx_factor_fcm_c_h)
        lcms_approx_factor_diffs_run.append(mean_approx_factor_l - mean_approx_factor_h)
        cms_add_error_totals_run.append(total_add_error_cms_h + total_add_error_cms_l)
        cmscu_add_error_totals_run.append(total_add_error_cmscu_h + total_add_error_cmscu_l)
        fcm_c_add_error_totals_run.append(total_add_error_fcm_c_h + total_add_error_fcm_c_l)
        lcms_add_error_totals_run.append(total_add_error_h + total_add_error_l)
        cms_approx_factor_l_run.append(mean_approx_factor_cms_l)
        cmscu_approx_factor_l_run.append(mean_approx_factor_cmscu_l)
        fcm_c_approx_factor_l_run.append(mean_approx_factor_fcm_c_l)
        lcms_approx_factor_l_run.append(mean_approx_factor_l)
        cms_add_error_l_run.append(total_add_error_cms_l)
        cmscu_add_error_l_run.append(total_add_error_cmscu_l)
        fcm_c_add_error_l_run.append(total_add_error_fcm_c_l)
        lcms_add_error_l_run.append(total_add_error_l)
        cms_approx_factor_h_run.append(mean_approx_factor_cms_h)
        cmscu_approx_factor_h_run.append(mean_approx_factor_cmscu_h)
        fcm_c_approx_factor_h_run.append(mean_approx_factor_fcm_c_h)
        lcms_approx_factor_h_run.append(mean_approx_factor_h)
        cms_add_error_h_run.append(total_add_error_cms_h)
        cmscu_add_error_h_run.append(total_add_error_cmscu_h)
        fcm_c_add_error_h_run.append(total_add_error_fcm_c_h)
        lcms_add_error_h_run.append(total_add_error_h)
        fcm_r_approx_factor_diffs_run.append(mean_approx_factor_fcm_r_l - mean_approx_factor_fcm_r_h)
        fcm_r_add_error_totals_run.append(total_add_error_fcm_r_h + total_add_error_fcm_r_l)
        fcm_r_approx_factor_l_run.append(mean_approx_factor_fcm_r_l)
        fcm_r_add_error_l_run.append(total_add_error_fcm_r_l)
        fcm_r_approx_factor_h_run.append(mean_approx_factor_fcm_r_h)
        fcm_r_add_error_h_run.append(total_add_error_fcm_r_h)

    cms_approx_factor_diffs.append(np.mean(cms_approx_factor_diffs_run))
    cmscu_approx_factor_diffs.append(np.mean(cmscu_approx_factor_diffs_run))
    fcm_c_approx_factor_diffs.append(np.mean(fcm_c_approx_factor_diffs_run))
    lcms_approx_factor_diffs.append(np.mean(lcms_approx_factor_diffs_run))
    cms_add_error_totals.append(np.mean(cms_add_error_totals_run))
    cmscu_add_error_totals.append(np.mean(cmscu_add_error_totals_run))
    fcm_c_add_error_totals.append(np.mean(fcm_c_add_error_totals_run))
    lcms_add_error_totals.append(np.mean(lcms_add_error_totals_run))
    cms_approx_factor_l.append(np.mean(cms_approx_factor_l_run))
    cmscu_approx_factor_l.append(np.mean(cmscu_approx_factor_l_run))
    fcm_c_approx_factor_l.append(np.mean(fcm_c_approx_factor_l_run))
    lcms_approx_factor_l.append(np.mean(lcms_approx_factor_l_run))
    cms_add_error_l.append(np.mean(cms_add_error_l_run))
    cmscu_add_error_l.append(np.mean(cmscu_add_error_l_run))
    fcm_c_add_error_l.append(np.mean(fcm_c_add_error_l_run))
    lcms_add_error_l.append(np.mean(lcms_add_error_l_run))
    cms_approx_factor_h.append(np.mean(cms_approx_factor_h_run))
    cmscu_approx_factor_h.append(np.mean(cmscu_approx_factor_h_run))
    fcm_c_approx_factor_h.append(np.mean(fcm_c_approx_factor_h_run))
    lcms_approx_factor_h.append(np.mean(lcms_approx_factor_h_run))
    cms_add_error_h.append(np.mean(cms_add_error_h_run))
    cmscu_add_error_h.append(np.mean(cmscu_add_error_h_run))
    fcm_c_add_error_h.append(np.mean(fcm_c_add_error_h_run))
    lcms_add_error_h.append(np.mean(lcms_add_error_h_run))
    fcm_r_approx_factor_diffs.append(np.mean(fcm_r_approx_factor_diffs_run))
    fcm_r_add_error_totals.append(np.mean(fcm_r_add_error_totals_run))
    fcm_r_approx_factor_l.append(np.mean(fcm_r_approx_factor_l_run))
    fcm_r_add_error_l.append(np.mean(fcm_r_add_error_l_run))
    fcm_r_approx_factor_h.append(np.mean(fcm_r_approx_factor_h_run))
    fcm_r_add_error_h.append(np.mean(fcm_r_add_error_h_run))

results = {
    "cms_approx_factor_diffs": cms_approx_factor_diffs,
    "cmscu_approx_factor_diffs": cmscu_approx_factor_diffs,
    "fcm_c_approx_factor_diffs": fcm_c_approx_factor_diffs,
    "lcms_approx_factor_diffs": lcms_approx_factor_diffs,
    "cms_add_error_totals": cms_add_error_totals,
    "cmscu_add_error_totals": cmscu_add_error_totals,
    "fcm_c_add_error_totals": fcm_c_add_error_totals,
    "lcms_add_error_totals": lcms_add_error_totals,
    "cms_approx_factor_l": cms_approx_factor_l,
    "cmscu_approx_factor_l": cmscu_approx_factor_l,
    "fcm_c_approx_factor_l": fcm_c_approx_factor_l,
    "lcms_approx_factor_l": lcms_approx_factor_l,
    "cms_add_error_l": cms_add_error_l,
    "cmscu_add_error_l": cmscu_add_error_l,
    "fcm_c_add_error_l": fcm_c_add_error_l,
    "lcms_add_error_l": lcms_add_error_l,
    "cms_approx_factor_h": cms_approx_factor_h,
    "cmscu_approx_factor_h": cmscu_approx_factor_h,
    "fcm_c_approx_factor_h": fcm_c_approx_factor_h,
    "lcms_approx_factor_h": lcms_approx_factor_h,
    "cms_add_error_h": cms_add_error_h,
    "cmscu_add_error_h": cmscu_add_error_h,
    "fcm_c_add_error_h": fcm_c_add_error_h,
    "lcms_add_error_h": lcms_add_error_h,
    "fcm_r_approx_factor_diffs": fcm_r_approx_factor_diffs,
    "fcm_r_add_error_totals": fcm_r_add_error_totals,
    "fcm_r_approx_factor_l": fcm_r_approx_factor_l,
    "fcm_r_add_error_l": fcm_r_add_error_l,
    "fcm_r_approx_factor_h": fcm_r_approx_factor_h,
    "fcm_r_add_error_h": fcm_r_add_error_h,
}

with open("results_census_varying_w_with.json", "w") as f:
    json.dump(results, f, indent=4)

with open("results_census_varying_w_with.json", "r") as f:
    results = json.load(f)

cms_approx_factor_diffs = list(map(abs, results["cms_approx_factor_diffs"]))
cms_add_error_totals = list(map(abs, results["cms_add_error_totals"]))
fcm_c_approx_factor_diffs = list(map(abs, results["fcm_c_approx_factor_diffs"]))
fcm_c_add_error_totals = list(map(abs, results["fcm_c_add_error_totals"]))
lcms_approx_factor_diffs = list(map(abs, results["lcms_approx_factor_diffs"]))
lcms_add_error_totals = list(map(abs, results["lcms_add_error_totals"]))
cmscu_approx_factor_diffs = list(map(abs, results["cmscu_approx_factor_diffs"]))
cmscu_add_error_totals = list(map(abs, results["cmscu_add_error_totals"]))
fcm_r_approx_factor_diffs = list(map(abs, results["fcm_r_approx_factor_diffs"]))
fcm_r_add_error_totals = list(map(abs, results["fcm_r_add_error_totals"]))

plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 16})
plt.plot(
    range(1, 7),
    cms_approx_factor_diffs,
    label="CM",
    color="tab:blue",
    marker="o",
    linestyle="-",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(1, 7),
    cmscu_approx_factor_diffs,
    label="CM-CU",
    color="tab:orange",
    marker="D",
    linestyle=":",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(1, 7),
    fcm_c_approx_factor_diffs,
    label="FCM",
    color="tab:green",
    marker="s",
    linestyle="--",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(1, 7),
    lcms_approx_factor_diffs,
    label="Learned-CM",
    color="tab:red",
    marker="^",
    linestyle="-.",
    markersize=10,
    linewidth=3,
)

plt.plot(
    range(1, 7),
    fcm_r_approx_factor_diffs,
    label="Row-Partitioning",
    color="tab:purple",
    marker="v",
    linestyle="--",
    markersize=10,
    linewidth=3,
)

plt.xticks(ticks=range(1, 7), labels=[f"$2^{{{i}}}$" for i in range(3, 9)])
plt.xlabel("w")
plt.ylabel("Mean Approx. Fact. Diff. (F - M)")
# plt.legend()
plt.grid(True)
plt.savefig("plots/census/mean_approx_factor_difference_plot_varying_w.png", bbox_inches="tight")


plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 16})
plt.plot(
    range(1, 7),
    cms_add_error_totals,
    label="CM",
    color="tab:blue",
    marker="o",
    linestyle="-",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(1, 7),
    cmscu_add_error_totals,
    label="CM-CU",
    color="tab:orange",
    marker="D",
    linestyle=":",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(1, 7),
    fcm_c_add_error_totals,
    label="FCM",
    color="tab:green",
    marker="s",
    linestyle="--",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(1, 7),
    lcms_add_error_totals,
    label="Learned-CM",
    color="tab:red",
    marker="^",
    linestyle="-.",
    markersize=10,
    linewidth=3,
)

plt.plot(
    range(1, 7),
    fcm_r_add_error_totals,
    label="Row-Partitioning",
    color="tab:purple",
    marker="v",
    linestyle="--",
    markersize=10,
    linewidth=3,
)
plt.xticks(ticks=range(1, 7), labels=[f"$2^{{{i}}}$" for i in range(3, 9)])
plt.xlabel("w")
plt.ylabel("Total Additive Error")
# plt.legend()
plt.grid(True)
plt.savefig("plots/census/price_of_fairness_plot_varying_w.png", bbox_inches="tight")


cms_approx_factor_diffs = []
cmscu_approx_factor_diffs = []
fcm_c_approx_factor_diffs = []
lcms_approx_factor_diffs = []
cms_add_error_totals = []
cmscu_add_error_totals = []
fcm_c_add_error_totals = []
lcms_add_error_totals = []
cms_approx_factor_l = []
cmscu_approx_factor_l = []
fcm_c_approx_factor_l = []
lcms_approx_factor_l = []
cms_add_error_l = []
cmscu_add_error_l = []
fcm_c_add_error_l = []
lcms_add_error_l = []
cms_approx_factor_h = []
cmscu_approx_factor_h = []
fcm_c_approx_factor_h = []
lcms_approx_factor_h = []
cms_add_error_h = []
cmscu_add_error_h = []
fcm_c_add_error_h = []
lcms_add_error_h = []
fcm_r_approx_factor_diffs = []
fcm_r_add_error_totals = []
fcm_r_approx_factor_l = []
fcm_r_add_error_l = []
fcm_r_approx_factor_h = []
fcm_r_add_error_h = []

for d in [i for i in range(2, 11)]:
    cms_approx_factor_diffs_run = []
    cmscu_approx_factor_diffs_run = []
    fcm_c_approx_factor_diffs_run = []
    lcms_approx_factor_diffs_run = []
    cms_add_error_totals_run = []
    cmscu_add_error_totals_run = []
    fcm_c_add_error_totals_run = []
    lcms_add_error_totals_run = []
    cms_approx_factor_l_run = []
    cmscu_approx_factor_l_run = []
    fcm_c_approx_factor_l_run = []
    lcms_approx_factor_l_run = []
    cms_add_error_l_run = []
    cmscu_add_error_l_run = []
    fcm_c_add_error_l_run = []
    lcms_add_error_l_run = []
    cms_approx_factor_h_run = []
    cmscu_approx_factor_h_run = []
    fcm_c_approx_factor_h_run = []
    lcms_approx_factor_h_run = []
    cms_add_error_h_run = []
    cmscu_add_error_h_run = []
    fcm_c_add_error_h_run = []
    lcms_add_error_h_run = []
    fcm_r_approx_factor_diffs_run = []
    fcm_r_add_error_totals_run = []
    fcm_r_approx_factor_l_run = []
    fcm_r_add_error_l_run = []
    fcm_r_approx_factor_h_run = []
    fcm_r_add_error_h_run = []

    for _ in range(5):
        (
            mean_approx_factor_cms_l,
            total_add_error_cms_l,
            mean_approx_factor_cms_h,
            total_add_error_cms_h,
            mean_approx_factor_cmscu_l,
            total_add_error_cmscu_l,
            mean_approx_factor_cmscu_h,
            total_add_error_cmscu_h,
            mean_approx_factor_fcm_c_l,
            total_add_error_fcm_c_l,
            mean_approx_factor_fcm_c_h,
            total_add_error_fcm_c_h,
            mean_approx_factor_lcms_l,
            total_add_error_lcms_l,
            mean_approx_factor_lcms_h,
            total_add_error_lcms_h,
            mean_approx_factor_fcm_r_l,
            total_add_error_fcm_r_l,
            mean_approx_factor_fcm_r_h,
            total_add_error_fcm_r_h,
        ) = run_fairness_census(w=64, d=d)

        cms_approx_factor_diffs_run.append(mean_approx_factor_cms_l - mean_approx_factor_cms_h)
        cmscu_approx_factor_diffs_run.append(mean_approx_factor_cmscu_l - mean_approx_factor_cmscu_h)
        fcm_c_approx_factor_diffs_run.append(mean_approx_factor_fcm_c_l - mean_approx_factor_fcm_c_h)
        lcms_approx_factor_diffs_run.append(mean_approx_factor_lcms_l - mean_approx_factor_lcms_h)
        cms_add_error_totals_run.append(total_add_error_cms_h + total_add_error_cms_l)
        cmscu_add_error_totals_run.append(total_add_error_cmscu_h + total_add_error_cmscu_l)
        fcm_c_add_error_totals_run.append(total_add_error_fcm_c_h + total_add_error_fcm_c_l)
        lcms_add_error_totals_run.append(total_add_error_lcms_h + total_add_error_lcms_l)
        cms_approx_factor_l_run.append(mean_approx_factor_cms_l)
        cmscu_approx_factor_l_run.append(mean_approx_factor_cmscu_l)
        fcm_c_approx_factor_l_run.append(mean_approx_factor_fcm_c_l)
        lcms_approx_factor_l_run.append(mean_approx_factor_lcms_l)
        cms_add_error_l_run.append(total_add_error_cms_l)
        cmscu_add_error_l_run.append(total_add_error_cmscu_l)
        fcm_c_add_error_l_run.append(total_add_error_fcm_c_l)
        lcms_add_error_l_run.append(total_add_error_lcms_l)
        cms_approx_factor_h_run.append(mean_approx_factor_cms_h)
        cmscu_approx_factor_h_run.append(mean_approx_factor_cmscu_h)
        fcm_c_approx_factor_h_run.append(mean_approx_factor_fcm_c_h)
        lcms_approx_factor_h_run.append(mean_approx_factor_lcms_h)
        cms_add_error_h_run.append(total_add_error_cms_h)
        cmscu_add_error_h_run.append(total_add_error_cmscu_h)
        fcm_c_add_error_h_run.append(total_add_error_fcm_c_h)
        lcms_add_error_h_run.append(total_add_error_lcms_h)
        fcm_r_approx_factor_diffs_run.append(mean_approx_factor_fcm_r_l - mean_approx_factor_fcm_r_h)
        fcm_r_add_error_totals_run.append(total_add_error_fcm_r_h + total_add_error_fcm_r_l)
        fcm_r_approx_factor_l_run.append(mean_approx_factor_fcm_r_l)
        fcm_r_add_error_l_run.append(total_add_error_fcm_r_l)
        fcm_r_approx_factor_h_run.append(mean_approx_factor_fcm_r_h)
        fcm_r_add_error_h_run.append(total_add_error_fcm_r_h)

    cms_approx_factor_diffs.append(np.mean(cms_approx_factor_diffs_run))
    cmscu_approx_factor_diffs.append(np.mean(cmscu_approx_factor_diffs_run))
    fcm_c_approx_factor_diffs.append(np.mean(fcm_c_approx_factor_diffs_run))
    lcms_approx_factor_diffs.append(np.mean(lcms_approx_factor_diffs_run))
    cms_add_error_totals.append(np.mean(cms_add_error_totals_run))
    cmscu_add_error_totals.append(np.mean(cmscu_add_error_totals_run))
    fcm_c_add_error_totals.append(np.mean(fcm_c_add_error_totals_run))
    lcms_add_error_totals.append(np.mean(lcms_add_error_totals_run))
    cms_approx_factor_l.append(np.mean(cms_approx_factor_l_run))
    cmscu_approx_factor_l.append(np.mean(cmscu_approx_factor_l_run))
    fcm_c_approx_factor_l.append(np.mean(fcm_c_approx_factor_l_run))
    lcms_approx_factor_l.append(np.mean(lcms_approx_factor_l_run))
    cms_add_error_l.append(np.mean(cms_add_error_l_run))
    cmscu_add_error_l.append(np.mean(cmscu_add_error_l_run))
    fcm_c_add_error_l.append(np.mean(fcm_c_add_error_l_run))
    lcms_add_error_l.append(np.mean(lcms_add_error_l_run))
    cms_approx_factor_h.append(np.mean(cms_approx_factor_h_run))
    cmscu_approx_factor_h.append(np.mean(cmscu_approx_factor_h_run))
    fcm_c_approx_factor_h.append(np.mean(fcm_c_approx_factor_h_run))
    lcms_approx_factor_h.append(np.mean(lcms_approx_factor_h_run))
    cms_add_error_h.append(np.mean(cms_add_error_h_run))
    cmscu_add_error_h.append(np.mean(cmscu_add_error_h_run))
    fcm_c_add_error_h.append(np.mean(fcm_c_add_error_h_run))
    lcms_add_error_h.append(np.mean(lcms_add_error_h_run))
    fcm_r_approx_factor_diffs.append(np.mean(fcm_r_approx_factor_diffs_run))
    fcm_r_add_error_totals.append(np.mean(fcm_r_add_error_totals_run))
    fcm_r_approx_factor_l.append(np.mean(fcm_r_approx_factor_l_run))
    fcm_r_add_error_l.append(np.mean(fcm_r_add_error_l_run))
    fcm_r_approx_factor_h.append(np.mean(fcm_r_approx_factor_h_run))
    fcm_r_add_error_h.append(np.mean(fcm_r_add_error_h_run))

results = {
    "cms_approx_factor_diffs": cms_approx_factor_diffs,
    "cmscu_approx_factor_diffs": cmscu_approx_factor_diffs,
    "fcm_c_approx_factor_diffs": fcm_c_approx_factor_diffs,
    "lcms_approx_factor_diffs": lcms_approx_factor_diffs,
    "cms_add_error_totals": cms_add_error_totals,
    "cmscu_add_error_totals": cmscu_add_error_totals,
    "fcm_c_add_error_totals": fcm_c_add_error_totals,
    "lcms_add_error_totals": lcms_add_error_totals,
    "cms_approx_factor_l": cms_approx_factor_l,
    "cmscu_approx_factor_l": cmscu_approx_factor_l,
    "fcm_c_approx_factor_l": fcm_c_approx_factor_l,
    "lcms_approx_factor_l": lcms_approx_factor_l,
    "cms_add_error_l": cms_add_error_l,
    "cmscu_add_error_l": cmscu_add_error_l,
    "fcm_c_add_error_l": fcm_c_add_error_l,
    "lcms_add_error_l": lcms_add_error_l,
    "cms_approx_factor_h": cms_approx_factor_h,
    "cmscu_approx_factor_h": cmscu_approx_factor_h,
    "fcm_c_approx_factor_h": fcm_c_approx_factor_h,
    "lcms_approx_factor_h": lcms_approx_factor_h,
    "cms_add_error_h": cms_add_error_h,
    "cmscu_add_error_h": cmscu_add_error_h,
    "fcm_c_add_error_h": fcm_c_add_error_h,
    "lcms_add_error_h": lcms_add_error_h,
    "fcm_r_approx_factor_diffs": fcm_r_approx_factor_diffs,
    "fcm_r_add_error_totals": fcm_r_add_error_totals,
    "fcm_r_approx_factor_l": fcm_r_approx_factor_l,
    "fcm_r_add_error_l": fcm_r_add_error_l,
    "fcm_r_approx_factor_h": fcm_r_approx_factor_h,
    "fcm_r_add_error_h": fcm_r_add_error_h,
}

with open("results_census_varying_d.json", "w") as f:
    json.dump(results, f, indent=4)


with open("results_census_varying_d.json", "r") as f:
    results = json.load(f)

cms_approx_factor_diffs = list(map(abs, results["cms_approx_factor_diffs"]))
cms_add_error_totals = list(map(abs, results["cms_add_error_totals"]))
fcm_c_approx_factor_diffs = list(map(abs, results["fcm_c_approx_factor_diffs"]))
fcm_c_add_error_totals = list(map(abs, results["fcm_c_add_error_totals"]))
lcms_approx_factor_diffs = list(map(abs, results["lcms_approx_factor_diffs"]))
lcms_add_error_totals = list(map(abs, results["lcms_add_error_totals"]))
cmscu_approx_factor_diffs = list(map(abs, results["cmscu_approx_factor_diffs"]))
cmscu_add_error_totals = list(map(abs, results["cmscu_add_error_totals"]))
fcm_r_approx_factor_diffs = list(map(abs, results["fcm_r_approx_factor_diffs"]))
fcm_r_add_error_totals = list(map(abs, results["fcm_r_add_error_totals"]))

plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 16})
plt.plot(
    range(2, 11),
    cms_approx_factor_diffs,
    label="CM",
    color="tab:blue",
    marker="o",
    linestyle="-",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(2, 11),
    cmscu_approx_factor_diffs,
    label="CM-CU",
    color="tab:orange",
    marker="D",
    linestyle=":",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(2, 11),
    fcm_c_approx_factor_diffs,
    label="FCM",
    color="tab:green",
    marker="s",
    linestyle="--",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(2, 11),
    lcms_approx_factor_diffs,
    label="Learned-CM",
    color="tab:red",
    marker="^",
    linestyle="-.",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(2, 11),
    fcm_r_approx_factor_diffs,
    label="Row-Partitioning",
    color="tab:purple",
    marker="v",
    linestyle="--",
    markersize=10,
    linewidth=3,
)
plt.xticks(ticks=range(2, 11), labels=[i for i in range(2, 11)])
plt.xlabel("d")
plt.ylabel("Mean Approx. Fact. Diff. (F - M)")
# plt.legend()
plt.grid(True)
plt.savefig("plots/census/mean_approx_factor_difference_plot_varying_d.png", bbox_inches="tight")


plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 16})
plt.plot(
    range(2, 11),
    cms_add_error_totals,
    label="CM",
    color="tab:blue",
    marker="o",
    linestyle="-",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(2, 11),
    cmscu_add_error_totals,
    label="CM-CU",
    color="tab:orange",
    marker="D",
    linestyle=":",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(2, 11),
    fcm_c_add_error_totals,
    label="FCM",
    color="tab:green",
    marker="s",
    linestyle="--",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(2, 11),
    lcms_add_error_totals,
    label="Learned-CM",
    color="tab:red",
    marker="^",
    linestyle="-.",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(2, 11),
    fcm_r_add_error_totals,
    label="Row-Partitioning",
    color="tab:purple",
    marker="v",
    linestyle="--",
    markersize=10,
    linewidth=3,
)
plt.xticks(ticks=range(2, 11), labels=[i for i in range(2, 11)])
plt.xlabel("d")
plt.ylabel("Total Additive Error")
plt.ylim(bottom=0)
# plt.legend()
plt.grid(True)
plt.savefig("plots/census/price_of_fairness_plot_varying_d.png", bbox_inches="tight")


cms_construction_time_avg = []
cmscu_construction_time_avg = []
fcm_c_construction_time_avg = []
lcms_construction_time_avg = []
cms_query_time_avg = []
cmscu_query_time_avg = []
fcm_c_query_time_avg = []
lcms_query_time_avg = []

for w in [2**i for i in range(3, 9)]:
    cms_construction_time_run = []
    cmscu_construction_time_run = []
    fcm_c_construction_time_run = []
    lcms_construction_time_run = []
    cms_query_time_run = []
    cmscu_query_time_run = []
    fcm_c_query_time_run = []
    lcms_query_time_run = []

    for j in range(5):
        (
            cms_construction_time,
            cmscu_construction_time,
            fcm_c_construction_time,
            lcms_construction_time,
            cms_query_time,
            cmscu_query_time,
            fcm_c_query_time,
            lcms_query_time,
        ) = run_efficiency_census(w=w, d=5)

        cms_construction_time_run.append(cms_construction_time)
        cmscu_construction_time_run.append(cmscu_construction_time)
        fcm_c_construction_time_run.append(fcm_c_construction_time)
        lcms_construction_time_run.append(lcms_construction_time)
        cms_query_time_run.append(cms_query_time)
        cmscu_query_time_run.append(cmscu_query_time)
        fcm_c_query_time_run.append(fcm_c_query_time)
        lcms_query_time_run.append(lcms_query_time)

    cms_construction_time_avg.append(np.mean(cms_construction_time_run))
    cmscu_construction_time_avg.append(np.mean(cmscu_construction_time_run))
    fcm_c_construction_time_avg.append(np.mean(fcm_c_construction_time_run))
    lcms_construction_time_avg.append(np.mean(lcms_construction_time_run))
    cms_query_time_avg.append(np.mean(cms_query_time_run))
    cmscu_query_time_avg.append(np.mean(cmscu_query_time_run))
    fcm_c_query_time_avg.append(np.mean(fcm_c_query_time_run))
    lcms_query_time_avg.append(np.mean(lcms_query_time_run))

results_efficiency = {
    "cms_construction_time_avg": cms_construction_time_avg,
    "cmscu_construction_time_avg": cmscu_construction_time_avg,
    "fcm_c_construction_time_avg": fcm_c_construction_time_avg,
    "lcms_construction_time_avg": lcms_construction_time_avg,
    "cms_query_time_avg": cms_query_time_avg,
    "cmscu_query_time_avg": cmscu_query_time_avg,
    "fcm_c_query_time_avg": fcm_c_query_time_avg,
    "lcms_query_time_avg": lcms_query_time_avg,
}

with open("results_efficiency_census_varying_w.json", "w") as f:
    json.dump(results_efficiency, f, indent=4)

with open("results_efficiency_census_varying_w.json", "r") as f:
    results = json.load(f)
cms_construction_time_avg = list(map(abs, results["cms_construction_time_avg"]))
cmscu_construction_time_avg = list(map(abs, results["cmscu_construction_time_avg"]))
fcm_c_construction_time_avg = list(map(abs, results["fcm_c_construction_time_avg"]))
lcms_construction_time_avg = list(map(abs, results["lcms_construction_time_avg"]))
cms_query_time_avg = list(map(abs, results["cms_query_time_avg"]))
cmscu_query_time_avg = list(map(abs, results["cmscu_query_time_avg"]))
fcm_c_query_time_avg = list(map(abs, results["fcm_c_query_time_avg"]))
lcms_query_time_avg = list(map(abs, results["lcms_query_time_avg"]))


plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 16})
plt.plot(
    range(1, 7),
    cms_construction_time_avg,
    label="CM",
    color="tab:blue",
    marker="o",
    linestyle="-",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(1, 7),
    cmscu_construction_time_avg,
    label="CM-CU",
    color="tab:orange",
    marker="D",
    linestyle=":",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(1, 7),
    fcm_c_construction_time_avg,
    label="FCM",
    color="tab:green",
    marker="s",
    linestyle="--",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(1, 7),
    lcms_construction_time_avg,
    label="Learned-CM",
    color="tab:red",
    marker="^",
    linestyle="-.",
    markersize=10,
    linewidth=3,
)
plt.xticks(ticks=range(1, 7), labels=[2**i for i in range(3, 9)])
plt.xlabel("w")
plt.yscale("log")
plt.ylabel("Average Construction Time (s)")
plt.ylim(bottom=1)
# plt.legend()
plt.grid(True)
plt.savefig("plots/census/construction_time_plot_varying_w.png", bbox_inches="tight")


plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 16})
plt.plot(
    range(1, 7), cms_query_time_avg, label="CM", color="tab:blue", marker="o", linestyle="-", markersize=10, linewidth=3
)
plt.plot(
    range(1, 7),
    cmscu_query_time_avg,
    label="CM-CU",
    color="tab:orange",
    marker="D",
    linestyle=":",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(1, 7),
    fcm_c_query_time_avg,
    label="FCM",
    color="tab:green",
    marker="s",
    linestyle="--",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(1, 7),
    lcms_query_time_avg,
    label="Learned-CM",
    color="tab:red",
    marker="^",
    linestyle="-.",
    markersize=10,
    linewidth=3,
)
plt.xticks(ticks=range(1, 7), labels=[2**i for i in range(3, 9)])
plt.xlabel("w")
plt.yscale("log")
plt.ylabel("Average Query Time (s)")
plt.ylim(bottom=1e-6)
# plt.legend()
plt.grid(True)
plt.savefig("plots/census/query_time_plot_varying_w.png", bbox_inches="tight")


cms_construction_time_avg = []
cmscu_construction_time_avg = []
fcm_c_construction_time_avg = []
lcms_construction_time_avg = []
cms_query_time_avg = []
cmscu_query_time_avg = []
fcm_c_query_time_avg = []
lcms_query_time_avg = []

for d in [i for i in range(2, 11)]:
    cms_construction_time_run = []
    cmscu_construction_time_run = []
    fcm_c_construction_time_run = []
    lcms_construction_time_run = []
    cms_query_time_run = []
    cmscu_query_time_run = []
    fcm_c_query_time_run = []
    lcms_query_time_run = []

    for j in range(5):
        (
            cms_construction_time,
            cmscu_construction_time,
            fcm_c_construction_time,
            lcms_construction_time,
            cms_query_time,
            cmscu_query_time,
            fcm_c_query_time,
            lcms_query_time,
        ) = run_efficiency_census(w=64, d=d)

        cms_construction_time_run.append(cms_construction_time)
        cmscu_construction_time_run.append(cmscu_construction_time)
        fcm_c_construction_time_run.append(fcm_c_construction_time)
        lcms_construction_time_run.append(lcms_construction_time)
        cms_query_time_run.append(cms_query_time)
        cmscu_query_time_run.append(cmscu_query_time)
        fcm_c_query_time_run.append(fcm_c_query_time)
        lcms_query_time_run.append(lcms_query_time)

    cms_construction_time_avg.append(np.mean(cms_construction_time_run))
    cmscu_construction_time_avg.append(np.mean(cmscu_construction_time_run))
    fcm_c_construction_time_avg.append(np.mean(fcm_c_construction_time_run))
    lcms_construction_time_avg.append(np.mean(lcms_construction_time_run))
    cms_query_time_avg.append(np.mean(cms_query_time_run))
    cmscu_query_time_avg.append(np.mean(cmscu_query_time_run))
    fcm_c_query_time_avg.append(np.mean(fcm_c_query_time_run))
    lcms_query_time_avg.append(np.mean(lcms_query_time_run))

results_efficiency = {
    "cms_construction_time_avg": cms_construction_time_avg,
    "cmscu_construction_time_avg": cmscu_construction_time_avg,
    "fcm_c_construction_time_avg": fcm_c_construction_time_avg,
    "lcms_construction_time_avg": lcms_construction_time_avg,
    "cms_query_time_avg": cms_query_time_avg,
    "cmscu_query_time_avg": cmscu_query_time_avg,
    "fcm_c_query_time_avg": fcm_c_query_time_avg,
    "lcms_query_time_avg": lcms_query_time_avg,
}

with open("results_efficiency_census_varying_d.json", "w") as f:
    json.dump(results_efficiency, f, indent=4)


with open("results_efficiency_census_varying_d.json", "r") as f:
    results = json.load(f)
cms_construction_time_avg = list(map(abs, results["cms_construction_time_avg"]))
cmscu_construction_time_avg = list(map(abs, results["cmscu_construction_time_avg"]))
fcm_c_construction_time_avg = list(map(abs, results["fcm_c_construction_time_avg"]))
lcms_construction_time_avg = list(map(abs, results["lcms_construction_time_avg"]))
cms_query_time_avg = list(map(abs, results["cms_query_time_avg"]))
cmscu_query_time_avg = list(map(abs, results["cmscu_query_time_avg"]))
fcm_c_query_time_avg = list(map(abs, results["fcm_c_query_time_avg"]))
lcms_query_time_avg = list(map(abs, results["lcms_query_time_avg"]))

plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 16})
plt.plot(
    range(2, 11),
    cms_construction_time_avg,
    label="CM",
    color="tab:blue",
    marker="o",
    linestyle="-",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(2, 11),
    cmscu_construction_time_avg,
    label="CM-CU",
    color="tab:orange",
    marker="D",
    linestyle=":",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(2, 11),
    fcm_c_construction_time_avg,
    label="FCM",
    color="tab:green",
    marker="s",
    linestyle="--",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(2, 11),
    lcms_construction_time_avg,
    label="Learned-CM",
    color="tab:red",
    marker="^",
    linestyle="-.",
    markersize=10,
    linewidth=3,
)
plt.xticks(ticks=range(2, 11), labels=[i for i in range(2, 11)])
plt.xlabel("d")
plt.ylabel("Average Construction Time (s)")
plt.yscale("log")
plt.ylim(bottom=1)
# plt.legend()
plt.grid(True)
plt.savefig("plots/census/construction_time_plot_varying_d_with.png", bbox_inches="tight")


plt.figure(figsize=(10, 6), dpi=200)
plt.rcParams.update({"font.size": 16})
plt.plot(
    range(2, 11),
    cms_query_time_avg,
    label="CM",
    color="tab:blue",
    marker="o",
    linestyle="-",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(2, 11),
    cmscu_query_time_avg,
    label="CM-CU",
    color="tab:orange",
    marker="D",
    linestyle=":",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(2, 11),
    fcm_c_query_time_avg,
    label="FCM",
    color="tab:green",
    marker="s",
    linestyle="--",
    markersize=10,
    linewidth=3,
)
plt.plot(
    range(2, 11),
    lcms_query_time_avg,
    label="Learned-CM",
    color="tab:red",
    marker="^",
    linestyle="-.",
    markersize=10,
    linewidth=3,
)
plt.xticks(ticks=range(2, 11), labels=[i for i in range(2, 11)])
plt.xlabel("d")
plt.ylabel("Average Query Time (s)")
plt.yscale("log")
plt.ylim(bottom=1e-6)
# plt.legend()
plt.grid(True)
plt.savefig("plots/census/query_time_plot_varying_d_with.png", bbox_inches="tight")

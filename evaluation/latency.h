// Inference latency tracking.

#ifndef OURS_NATIVE_EVALUATION_LATENCY_H
#define OURS_NATIVE_EVALUATION_LATENCY_H

#include <algorithm>
#include <cstddef>
#include <vector>

namespace ours {

struct LatencySummary {
    int count = 0;
    double mean = 0.0;
    double p50 = 0.0;
    double p99 = 0.0;
};

class LatencyTracker {
public:
    void record(double seconds) { records_.push_back(seconds); }

    LatencySummary summary() const {
        if (records_.empty()) {
            return LatencySummary{0, 0.0, 0.0, 0.0};
        }
        std::vector<double> sorted = records_;
        std::sort(sorted.begin(), sorted.end());
        double sum = 0.0;
        for (double v : sorted) {
            sum += v;
        }
        return LatencySummary{static_cast<int>(sorted.size()),
                              sum / static_cast<double>(sorted.size()),
                              percentile(sorted, 50.0), percentile(sorted, 99.0)};
    }

private:
    // Linear-interpolation percentile (matches numpy's default "linear").
    static double percentile(const std::vector<double> & sorted, double p) {
        const double idx = p / 100.0 * static_cast<double>(sorted.size() - 1);
        const size_t lo = static_cast<size_t>(idx);
        const size_t hi = lo + 1 < sorted.size() ? lo + 1 : lo;
        const double frac = idx - static_cast<double>(lo);
        return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
    }

    std::vector<double> records_;
};

} // namespace ours

#endif // OURS_NATIVE_EVALUATION_LATENCY_H

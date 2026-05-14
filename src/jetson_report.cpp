#include "inferedge_runtime/jetson_report.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace inferedge_runtime {
namespace {

struct TegrastatsSummary {
    std::string status = "not_provided";
    int sample_count = 0;
    double ram_used_mb_avg = 0.0;
    double ram_used_mb_max = 0.0;
    double ram_total_mb = 0.0;
    double max_temp_c = 0.0;
    std::string max_temp_name;
    double vdd_in_mw_avg = 0.0;
    double vdd_in_mw_max = 0.0;
};

struct RuntimeEvidence {
    std::string source_path;
    std::string model_name;
    std::string backend_key;
    std::string compare_key;
    std::string engine_backend;
    std::string device_name;
    std::string precision;
    std::string power_mode;
    std::string jetson_clocks;
    std::string tegrastats_log_path;
    std::string timestamp;
    int warmup = 0;
    int runs = 0;
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double fps_value = 0.0;
    TegrastatsSummary tegrastats;
};

std::string read_text_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to read file: " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void write_text_file(const std::filesystem::path& path, const std::string& text) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to write file: " + path.string());
    }
    output << text;
}

std::string find_object_section(const std::string& json, const std::string& key) {
    const std::string key_pattern = "\"" + key + "\"";
    const std::size_t key_pos = json.find(key_pattern);
    if (key_pos == std::string::npos) {
        return "";
    }

    const std::size_t colon_pos = json.find(':', key_pos + key_pattern.size());
    if (colon_pos == std::string::npos) {
        return "";
    }
    const std::size_t object_start = json.find('{', colon_pos + 1);
    if (object_start == std::string::npos) {
        return "";
    }

    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t i = object_start; i < json.size(); ++i) {
        const char ch = json[i];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (ch == '\\') {
            escaped = true;
            continue;
        }
        if (ch == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) {
            continue;
        }
        if (ch == '{') {
            depth += 1;
        } else if (ch == '}') {
            depth -= 1;
            if (depth == 0) {
                return json.substr(object_start, i - object_start + 1);
            }
        }
    }
    return "";
}

std::string extract_string_value(const std::string& json, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return match[1].str();
    }
    return "";
}

double extract_number_value(const std::string& json, const std::string& key, double fallback = 0.0) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return std::stod(match[1].str());
    }
    return fallback;
}

int extract_int_value(const std::string& json, const std::string& key, int fallback = 0) {
    return static_cast<int>(extract_number_value(json, key, static_cast<double>(fallback)));
}

std::string first_non_empty(const std::string& first, const std::string& second) {
    return first.empty() ? second : first;
}

std::string display_or_unknown(const std::string& value) {
    return value.empty() ? "unknown" : value;
}

double delta_pct(double base, double candidate) {
    if (base == 0.0) {
        return 0.0;
    }
    return ((candidate - base) / base) * 100.0;
}

std::string fmt(double value) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(4) << value;
    std::string text = stream.str();
    while (text.size() > 1 && text.back() == '0') {
        text.pop_back();
    }
    if (!text.empty() && text.back() == '.') {
        text.pop_back();
    }
    return text;
}

std::string capture_depth_label(int runs, int tegrastats_samples) {
    if (runs >= 500 || tegrastats_samples >= 300) {
        return "sustained";
    }
    if (runs >= 100 || tegrastats_samples >= 60) {
        return "sustained_candidate";
    }
    return "short_smoke";
}

std::string capture_depth_note(const std::string& depth) {
    if (depth == "sustained") {
        return "Longer run evidence suitable for thermal/power stability review.";
    }
    if (depth == "sustained_candidate") {
        return "Deeper than a quick smoke, but still review duration before calling it sustained.";
    }
    return "Short validation smoke; useful for contract and device evidence, not a sustained thermal claim.";
}

TegrastatsSummary parse_tegrastats_log(const std::string& path) {
    TegrastatsSummary summary;
    if (path.empty()) {
        return summary;
    }

    std::ifstream input(path);
    if (!input) {
        summary.status = "unavailable";
        return summary;
    }

    summary.status = "parsed";
    const std::regex ram_pattern(R"(RAM\s+([0-9]+)/([0-9]+)MB)");
    const std::regex temp_pattern(R"(([A-Za-z0-9_]+)@([0-9]+(?:\.[0-9]+)?)C)");
    const std::regex vdd_in_pattern(R"(VDD_IN\s+([0-9]+)mW(/([0-9]+)mW)?)");

    std::string line;
    double ram_used_sum = 0.0;
    int ram_samples = 0;
    double vdd_in_sum = 0.0;
    int vdd_in_samples = 0;

    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        summary.sample_count += 1;

        std::smatch match;
        if (std::regex_search(line, match, ram_pattern)) {
            const double used = std::stod(match[1].str());
            const double total = std::stod(match[2].str());
            ram_used_sum += used;
            ram_samples += 1;
            summary.ram_used_mb_max = std::max(summary.ram_used_mb_max, used);
            summary.ram_total_mb = total;
        }

        for (auto it = std::sregex_iterator(line.begin(), line.end(), temp_pattern);
             it != std::sregex_iterator();
             ++it) {
            const std::smatch temp_match = *it;
            const double temp_c = std::stod(temp_match[2].str());
            if (temp_c > summary.max_temp_c) {
                summary.max_temp_c = temp_c;
                summary.max_temp_name = temp_match[1].str();
            }
        }

        if (std::regex_search(line, match, vdd_in_pattern)) {
            const double vdd_in = std::stod(match[1].str());
            vdd_in_sum += vdd_in;
            vdd_in_samples += 1;
            summary.vdd_in_mw_max = std::max(summary.vdd_in_mw_max, vdd_in);
        }
    }

    if (summary.sample_count == 0) {
        summary.status = "no_samples";
    }
    if (ram_samples > 0) {
        summary.ram_used_mb_avg = ram_used_sum / static_cast<double>(ram_samples);
    }
    if (vdd_in_samples > 0) {
        summary.vdd_in_mw_avg = vdd_in_sum / static_cast<double>(vdd_in_samples);
    }

    return summary;
}

TegrastatsSummary parse_embedded_tegrastats_summary(const std::string& json) {
    TegrastatsSummary summary;
    const std::string jetson_evidence = find_object_section(json, "jetson_evidence");
    const std::string section = find_object_section(jetson_evidence, "tegrastats_summary");
    if (section.empty()) {
        return summary;
    }
    summary.status = extract_string_value(section, "status");
    summary.sample_count = extract_int_value(section, "sample_count");
    summary.ram_used_mb_avg = extract_number_value(section, "ram_used_mb_avg");
    summary.ram_used_mb_max = extract_number_value(section, "ram_used_mb_max");
    summary.ram_total_mb = extract_number_value(section, "ram_total_mb");
    summary.max_temp_c = extract_number_value(section, "max_temp_c");
    summary.max_temp_name = extract_string_value(section, "max_temp_name");
    summary.vdd_in_mw_avg = extract_number_value(section, "vdd_in_mw_avg");
    summary.vdd_in_mw_max = extract_number_value(section, "vdd_in_mw_max");
    return summary;
}

RuntimeEvidence parse_runtime_evidence(const std::string& result_json_path) {
    const std::string json = read_text_file(result_json_path);
    const std::string run_config = find_object_section(json, "run_config");
    const std::string latency_ms = find_object_section(json, "latency_ms");
    const std::string model = find_object_section(json, "model");
    const std::string engine = find_object_section(json, "engine");
    const std::string device = find_object_section(json, "device");
    const std::string jetson_evidence = find_object_section(json, "jetson_evidence");

    RuntimeEvidence evidence;
    evidence.source_path = result_json_path;
    evidence.model_name = first_non_empty(
        extract_string_value(json, "model_name"),
        first_non_empty(extract_string_value(model, "name"), extract_string_value(model, "path")));
    evidence.backend_key = extract_string_value(json, "backend_key");
    evidence.compare_key = extract_string_value(json, "compare_key");
    evidence.engine_backend = first_non_empty(
        extract_string_value(json, "engine_backend"),
        first_non_empty(extract_string_value(json, "engine_name"), extract_string_value(engine, "backend")));
    evidence.device_name = first_non_empty(
        extract_string_value(json, "device_name"),
        extract_string_value(device, "name"));
    evidence.precision = extract_string_value(json, "precision");
    evidence.power_mode = first_non_empty(
        extract_string_value(run_config, "power_mode"),
        extract_string_value(jetson_evidence, "power_mode"));
    evidence.jetson_clocks = first_non_empty(
        extract_string_value(run_config, "jetson_clocks"),
        extract_string_value(jetson_evidence, "jetson_clocks"));
    evidence.tegrastats_log_path = first_non_empty(
        extract_string_value(run_config, "tegrastats_log_path"),
        extract_string_value(jetson_evidence, "tegrastats_log_path"));
    evidence.timestamp = extract_string_value(json, "timestamp");
    evidence.warmup = extract_int_value(run_config, "warmup");
    evidence.runs = extract_int_value(run_config, "runs");
    evidence.mean_ms = extract_number_value(json, "mean_ms", extract_number_value(latency_ms, "mean"));
    evidence.p50_ms = extract_number_value(json, "p50_ms", extract_number_value(latency_ms, "p50"));
    evidence.p95_ms = extract_number_value(json, "p95_ms", extract_number_value(latency_ms, "p95"));
    evidence.p99_ms = extract_number_value(json, "p99_ms", extract_number_value(latency_ms, "p99"));
    evidence.fps_value = extract_number_value(json, "fps_value", extract_number_value(json, "fps"));
    evidence.tegrastats = parse_embedded_tegrastats_summary(json);
    return evidence;
}

std::string runtime_summary_table(const RuntimeEvidence& evidence) {
    std::ostringstream markdown;
    markdown
        << "| Field | Value |\n"
        << "|---|---|\n"
        << "| source | `" << evidence.source_path << "` |\n"
        << "| model | `" << display_or_unknown(evidence.model_name) << "` |\n"
        << "| backend_key | `" << display_or_unknown(evidence.backend_key) << "` |\n"
        << "| compare_key | `" << display_or_unknown(evidence.compare_key) << "` |\n"
        << "| engine | `" << display_or_unknown(evidence.engine_backend) << "` |\n"
        << "| device | `" << display_or_unknown(evidence.device_name) << "` |\n"
        << "| precision | `" << display_or_unknown(evidence.precision) << "` |\n"
        << "| power_mode | `" << display_or_unknown(evidence.power_mode) << "` |\n"
        << "| jetson_clocks | `" << display_or_unknown(evidence.jetson_clocks) << "` |\n"
        << "| warmup | " << evidence.warmup << " |\n"
        << "| runs | " << evidence.runs << " |\n"
        << "| mean_ms | " << fmt(evidence.mean_ms) << " |\n"
        << "| p50_ms | " << fmt(evidence.p50_ms) << " |\n"
        << "| p95_ms | " << fmt(evidence.p95_ms) << " |\n"
        << "| p99_ms | " << fmt(evidence.p99_ms) << " |\n"
        << "| fps | " << fmt(evidence.fps_value) << " |\n"
        << "| capture_depth | `" << capture_depth_label(evidence.runs, evidence.tegrastats.sample_count) << "` |\n"
        << "| timestamp | `" << display_or_unknown(evidence.timestamp) << "` |\n";
    return markdown.str();
}

std::string tegrastats_summary_table(const TegrastatsSummary& summary, const std::string& source) {
    std::ostringstream markdown;
    markdown
        << "| Field | Value |\n"
        << "|---|---|\n"
        << "| source | `" << display_or_unknown(source) << "` |\n"
        << "| status | `" << display_or_unknown(summary.status) << "` |\n"
        << "| sample_count | " << summary.sample_count << " |\n"
        << "| ram_used_mb_avg | " << fmt(summary.ram_used_mb_avg) << " |\n"
        << "| ram_used_mb_max | " << fmt(summary.ram_used_mb_max) << " |\n"
        << "| ram_total_mb | " << fmt(summary.ram_total_mb) << " |\n"
        << "| max_temp_c | " << fmt(summary.max_temp_c) << " |\n"
        << "| max_temp_name | `" << display_or_unknown(summary.max_temp_name) << "` |\n"
        << "| vdd_in_mw_avg | " << fmt(summary.vdd_in_mw_avg) << " |\n"
        << "| vdd_in_mw_max | " << fmt(summary.vdd_in_mw_max) << " |\n";
    return markdown.str();
}

std::string metric_comparison_row(
    const std::string& metric_name,
    double base,
    double candidate,
    const std::string& base_label,
    const std::string& candidate_label) {
    const double delta = candidate - base;
    std::ostringstream row;
    row
        << "| " << metric_name
        << " | " << fmt(base)
        << " | " << fmt(candidate)
        << " | " << fmt(delta)
        << " | " << fmt(delta_pct(base, candidate)) << "% |";
    (void)base_label;
    (void)candidate_label;
    return row.str();
}

std::string depth_comparison_row(
    const std::string& field,
    const std::string& base,
    const std::string& candidate) {
    std::ostringstream row;
    row << "| " << field << " | `" << base << "` | `" << candidate << "` |";
    return row.str();
}

}  // namespace

std::filesystem::path write_jetson_evidence_markdown_report(
    const std::string& result_json_path,
    const std::string& tegrastats_log_path,
    const std::string& output_path) {
    RuntimeEvidence evidence = parse_runtime_evidence(result_json_path);
    std::string tegrastats_source = "embedded result JSON";
    if (!tegrastats_log_path.empty()) {
        evidence.tegrastats = parse_tegrastats_log(tegrastats_log_path);
        tegrastats_source = tegrastats_log_path;
    }

    std::ostringstream markdown;
    markdown
        << "# InferEdge Runtime Jetson Evidence Summary\n\n"
        << "This report summarizes Runtime JSON and optional tegrastats evidence for Lab-compatible deployment validation.\n"
        << "It is not a production inference server report or a TensorRT INT8 calibration workflow.\n\n"
        << "## Runtime Result\n\n"
        << runtime_summary_table(evidence)
        << "\n## Tegrastats Summary\n\n"
        << tegrastats_summary_table(evidence.tegrastats, tegrastats_source)
        << "\n## Evidence Depth\n\n"
        << "| Field | Value |\n"
        << "|---|---|\n"
        << "| capture_depth | `" << capture_depth_label(evidence.runs, evidence.tegrastats.sample_count) << "` |\n"
        << "| interpretation | " << capture_depth_note(capture_depth_label(evidence.runs, evidence.tegrastats.sample_count)) << " |\n"
        << "| sustained_threshold | `runs >= 500` or `tegrastats samples >= 300` |\n"
        << "| current_samples | " << evidence.tegrastats.sample_count << " |\n"
        << "\n## Lab Handoff\n\n"
        << "- Lab-compatible import path: `" << result_json_path << "`\n"
        << "- Runtime supplies execution evidence. InferEdgeLab owns comparison and deployment decision interpretation.\n";

    const std::filesystem::path report_path(output_path);
    write_text_file(report_path, markdown.str());
    return report_path;
}

std::filesystem::path write_power_mode_comparison_markdown_report(
    const std::string& base_result_json_path,
    const std::string& candidate_result_json_path,
    const std::string& output_path) {
    const RuntimeEvidence base = parse_runtime_evidence(base_result_json_path);
    const RuntimeEvidence candidate = parse_runtime_evidence(candidate_result_json_path);
    const std::string base_label = display_or_unknown(base.power_mode);
    const std::string candidate_label = display_or_unknown(candidate.power_mode);

    std::ostringstream markdown;
    markdown
        << "# InferEdge Runtime Jetson Power Mode Comparison\n\n"
        << "This report compares two Runtime result JSON files as Jetson system evidence.\n"
        << "Different power modes are not treated as the same run_config regression test.\n\n"
        << "## Compared Results\n\n"
        << "| Field | Base | Candidate |\n"
        << "|---|---|---|\n"
        << "| source | `" << base.source_path << "` | `" << candidate.source_path << "` |\n"
        << "| backend_key | `" << display_or_unknown(base.backend_key) << "` | `" << display_or_unknown(candidate.backend_key) << "` |\n"
        << "| compare_key | `" << display_or_unknown(base.compare_key) << "` | `" << display_or_unknown(candidate.compare_key) << "` |\n"
        << "| power_mode | `" << base_label << "` | `" << candidate_label << "` |\n"
        << "| precision | `" << display_or_unknown(base.precision) << "` | `" << display_or_unknown(candidate.precision) << "` |\n"
        << "| jetson_clocks | `" << display_or_unknown(base.jetson_clocks) << "` | `" << display_or_unknown(candidate.jetson_clocks) << "` |\n\n"
        << "## Latency / FPS Comparison\n\n"
        << "| Metric | " << base_label << " | " << candidate_label << " | Delta | Delta % |\n"
        << "|---|---:|---:|---:|---:|\n"
        << metric_comparison_row("mean_ms", base.mean_ms, candidate.mean_ms, base_label, candidate_label) << "\n"
        << metric_comparison_row("p50_ms", base.p50_ms, candidate.p50_ms, base_label, candidate_label) << "\n"
        << metric_comparison_row("p95_ms", base.p95_ms, candidate.p95_ms, base_label, candidate_label) << "\n"
        << metric_comparison_row("p99_ms", base.p99_ms, candidate.p99_ms, base_label, candidate_label) << "\n"
        << metric_comparison_row("fps", base.fps_value, candidate.fps_value, base_label, candidate_label) << "\n\n"
        << "## Run Depth Comparison\n\n"
        << "| Field | " << base_label << " | " << candidate_label << " |\n"
        << "|---|---|---|\n"
        << depth_comparison_row("warmup", std::to_string(base.warmup), std::to_string(candidate.warmup)) << "\n"
        << depth_comparison_row("runs", std::to_string(base.runs), std::to_string(candidate.runs)) << "\n"
        << depth_comparison_row(
            "tegrastats_sample_count",
            std::to_string(base.tegrastats.sample_count),
            std::to_string(candidate.tegrastats.sample_count)) << "\n"
        << depth_comparison_row(
            "capture_depth",
            capture_depth_label(base.runs, base.tegrastats.sample_count),
            capture_depth_label(candidate.runs, candidate.tegrastats.sample_count)) << "\n\n"
        << "## Tegrastats Comparison\n\n"
        << "| Metric | " << base_label << " | " << candidate_label << " | Delta | Delta % |\n"
        << "|---|---:|---:|---:|---:|\n"
        << metric_comparison_row("sample_count", base.tegrastats.sample_count, candidate.tegrastats.sample_count, base_label, candidate_label) << "\n"
        << metric_comparison_row("max_temp_c", base.tegrastats.max_temp_c, candidate.tegrastats.max_temp_c, base_label, candidate_label) << "\n"
        << metric_comparison_row("vdd_in_mw_avg", base.tegrastats.vdd_in_mw_avg, candidate.tegrastats.vdd_in_mw_avg, base_label, candidate_label) << "\n"
        << metric_comparison_row("vdd_in_mw_max", base.tegrastats.vdd_in_mw_max, candidate.tegrastats.vdd_in_mw_max, base_label, candidate_label) << "\n\n"
        << "## Interpretation Notes\n\n"
        << "- Power mode changes are deployment validation evidence, not a same-run_config latency regression test.\n"
        << "- `capture_depth=short_smoke` should not be described as a sustained thermal benchmark.\n"
        << "- Runtime exports evidence; InferEdgeLab owns comparison policy and deployment decision.\n"
        << "- TensorRT INT8 automatic calibration is outside this report scope.\n";

    const std::filesystem::path report_path(output_path);
    write_text_file(report_path, markdown.str());
    return report_path;
}

}  // namespace inferedge_runtime

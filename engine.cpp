// Fresh implementation of the BFF noheads semantics described by CuBFF
// (Google LLC, Apache-2.0) and Computational Life, arXiv:2406.19108.
// The engineered control is the sanitized palindrome in that paper's Figure 4.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Bytes = std::vector<uint8_t>;
using U64 = uint64_t;

struct Rng {
  std::array<U64, 4> state{};
  explicit Rng(U64 seed = 1) {
    for (auto &x : state) {
      seed += 0x9e3779b97f4a7c15ULL;
      U64 z = seed;
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
      x = z ^ (z >> 31);
    }
  }
  static U64 rotl(U64 x, unsigned k) { return (x << k) | (x >> (64 - k)); }
  U64 next() {
    const U64 result = rotl(state[1] * 5, 7) * 9;
    const U64 t = state[1] << 17;
    state[2] ^= state[0]; state[3] ^= state[1];
    state[1] ^= state[2]; state[0] ^= state[3];
    state[2] ^= t; state[3] = rotl(state[3], 45);
    return result;
  }
  U64 bounded(U64 n) {
    if (!n) throw std::runtime_error("empty random range");
    const U64 threshold = -n % n;
    U64 x;
    do { x = next(); } while (x < threshold);
    return x % n;
  }
  double unit() { return (next() >> 11) * 0x1.0p-53; }
};

struct VmResult { U64 steps = 0, active_ops = 0; bool exhausted = false; };

VmResult execute(Bytes &tape, const Bytes &disabled, U64 budget) {
  if (tape.empty() || disabled.size() != tape.size())
    throw std::runtime_error("invalid VM tape or mask");
  const int64_t n = static_cast<int64_t>(tape.size());
  int64_t ip = 0, h0 = 0, h1 = 0;
  VmResult result;
  auto op_at = [&](int64_t p) -> uint8_t { return disabled[p] ? 0 : tape[p]; };
  while (ip >= 0 && ip < n && result.steps < budget) {
    const uint8_t op = op_at(ip);
    ++result.steps;
    bool recognized = true;
    switch (op) {
      case '<': h0 = (h0 + n - 1) % n; break;
      case '>': h0 = (h0 + 1) % n; break;
      case '{': h1 = (h1 + n - 1) % n; break;
      case '}': h1 = (h1 + 1) % n; break;
      case '+': ++tape[h0]; break;
      case '-': --tape[h0]; break;
      case '.': tape[h1] = tape[h0]; break;
      case ',': tape[h0] = tape[h1]; break;
      case '[':
      case ']': {
        const bool forward = op == '[';
        if ((forward && tape[h0] == 0) || (!forward && tape[h0] != 0)) {
          const int64_t direction = forward ? 1 : -1;
          int depth = 1;
          int64_t scan = ip + direction;
          while (scan >= 0 && scan < n && depth && result.steps < budget) {
            const uint8_t scanned = op_at(scan);
            ++result.steps;
            if (scanned == op) ++depth;
            else if (scanned == (forward ? ']' : '[')) --depth;
            if (depth) scan += direction;
          }
          if (depth) {
            result.exhausted = result.steps >= budget && scan >= 0 && scan < n;
            ++result.active_ops;
            return result;
          }
          ip = scan;
        }
        break;
      }
      default: recognized = false;
    }
    result.active_ops += recognized;
    ++ip;
  }
  result.exhausted = result.steps >= budget && ip >= 0 && ip < n;
  return result;
}

struct Lesion {
  std::string kind = "none";
  double severity = 0.25, exposure = 1.0;
  U64 seed = 991;
};

struct World {
  U64 population = 1024, length = 64, budget = 8192, interaction = 0, seed = 1;
  U64 executed_steps = 0, active_ops = 0, changed_bytes = 0, exact_copy_events = 0;
  double mutation = 1.0 / 4096.0;
  Rng rng{1};
  Bytes soup, disabled;
  Lesion lesion;
};

void validate_dimensions(U64 population, U64 length) {
  if (population < 2 || population > 1000000 || length < 8 || length > 4096 ||
      population > 536870912ULL / length)
    throw std::runtime_error("population must be 2..1000000, length 8..4096, total bytes <=512 MiB");
}

Bytes control_program(U64 length) {
  if (length != 64) throw std::runtime_error("the engineered positive control requires --length 64");
  const std::string s = "[[{.>]-]" + std::string(48, ' ') + "]-]>.{[[";
  return Bytes(s.begin(), s.end());
}

World create_world(U64 population, U64 length, U64 budget, U64 seed, double mutation, bool seeded) {
  validate_dimensions(population, length);
  World w;
  w.population = population; w.length = length; w.budget = budget;
  w.seed = seed; w.rng = Rng(seed); w.mutation = mutation;
  w.soup.resize(population * length); w.disabled.assign(w.soup.size(), 0);
  for (auto &v : w.soup) v = static_cast<uint8_t>(w.rng.next());
  if (seeded) {
    const Bytes control = control_program(length);
    for (U64 slot = 0; slot < std::max<U64>(1, population / 16); ++slot)
      std::copy(control.begin(), control.end(), w.soup.begin() + slot * length);
  }
  return w;
}

template <typename T> void shuffle(std::vector<T> &v, Rng &rng) {
  for (U64 i = v.size(); i > 1; --i) std::swap(v[i - 1], v[rng.bounded(i)]);
}

void apply_lesion(World &w, const Lesion &lesion) {
  if (lesion.kind == "none") return;
  if (lesion.kind != "focal" && lesion.kind != "random")
    throw std::runtime_error("lesion must be none, focal or random");
  if (std::any_of(w.disabled.begin(), w.disabled.end(), [](uint8_t v) { return v; }) &&
      (w.lesion.kind != lesion.kind || w.lesion.severity != lesion.severity ||
       w.lesion.exposure != lesion.exposure || w.lesion.seed != lesion.seed))
    throw std::runtime_error("checkpoint already has a different lesion; branch an unlesioned checkpoint");
  Rng lesion_rng(lesion.seed);
  std::vector<U64> slots(w.population);
  std::iota(slots.begin(), slots.end(), 0);
  shuffle(slots, lesion_rng);
  const U64 count = std::llround(lesion.severity * w.length);
  const U64 affected = std::llround(lesion.exposure * w.population);
  for (U64 i = 0; i < affected; ++i) {
    const U64 base = slots[i] * w.length;
    if (lesion.kind == "focal") {
      const U64 start = lesion_rng.bounded(w.length - count + 1);
      for (U64 p = start; p < start + count; ++p) w.disabled[base + p] = 1;
    } else {
      std::vector<U64> positions(w.length);
      std::iota(positions.begin(), positions.end(), 0);
      shuffle(positions, lesion_rng);
      for (U64 p = 0; p < count; ++p) w.disabled[base + positions[p]] = 1;
    }
  }
  w.lesion = lesion;
}

bool equal_half(const Bytes &whole, U64 offset, const Bytes &original, U64 original_offset, U64 length) {
  return std::equal(whole.begin() + offset, whole.begin() + offset + length,
                    original.begin() + original_offset);
}

bool new_exact_copy(const Bytes &before, const Bytes &after, U64 length) {
  if (equal_half(before, 0, before, length, length)) return false;
  for (const U64 source : {U64(0), length}) {
    if (equal_half(after, 0, before, source, length) &&
        equal_half(after, length, before, source, length)) return true;
  }
  return false;
}

void interact(World &w, Bytes &tape, Bytes &mask, Bytes &before) {
  const U64 a = w.rng.bounded(w.population);
  U64 b = w.rng.bounded(w.population - 1);
  if (b >= a) ++b;
  for (U64 j = 0; j < w.length; ++j) {
    tape[j] = w.soup[a * w.length + j];
    tape[w.length + j] = w.soup[b * w.length + j];
    mask[j] = w.disabled[a * w.length + j];
    mask[w.length + j] = w.disabled[b * w.length + j];
  }
  before = tape;
  for (auto &v : tape) {
    const double draw = w.rng.unit();
    const uint8_t replacement = static_cast<uint8_t>(w.rng.next());
    if (draw < w.mutation) v = replacement;
  }
  const Bytes mutated = tape;
  const VmResult result = execute(tape, mask, w.budget);
  w.executed_steps += result.steps; w.active_ops += result.active_ops;
  w.exact_copy_events += new_exact_copy(mutated, tape, w.length);
  for (U64 j = 0; j < tape.size(); ++j) w.changed_bytes += tape[j] != before[j];
  std::copy_n(tape.begin(), w.length, w.soup.begin() + a * w.length);
  std::copy_n(tape.begin() + w.length, w.length, w.soup.begin() + b * w.length);
  ++w.interaction;
}

void put_u64(std::ostream &out, U64 value) {
  for (unsigned b = 0; b < 8; ++b) out.put(static_cast<char>(value >> (8 * b)));
}
U64 get_u64(std::istream &in) {
  U64 value = 0;
  for (unsigned b = 0; b < 8; ++b) {
    const int c = in.get();
    if (c == EOF) throw std::runtime_error("truncated checkpoint");
    value |= U64(static_cast<uint8_t>(c)) << (8 * b);
  }
  return value;
}
void put_double(std::ostream &out, double value) {
  U64 bits; std::memcpy(&bits, &value, sizeof(bits)); put_u64(out, bits);
}
double get_double(std::istream &in) {
  const U64 bits = get_u64(in); double value; std::memcpy(&value, &bits, sizeof(value)); return value;
}
void save_world(const World &w, const std::string &path) {
  const std::string temporary = path + ".tmp";
  std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write checkpoint " + path);
  out.write("BFFLAB02", 8);
  for (U64 value : {w.population, w.length, w.budget, w.interaction, w.seed,
                    w.executed_steps, w.active_ops, w.changed_bytes, w.exact_copy_events}) put_u64(out, value);
  put_double(out, w.mutation);
  for (U64 value : w.rng.state) put_u64(out, value);
  put_u64(out, w.lesion.kind == "focal" ? 1 : w.lesion.kind == "random" ? 2 : 0);
  put_double(out, w.lesion.severity); put_double(out, w.lesion.exposure); put_u64(out, w.lesion.seed);
  out.write(reinterpret_cast<const char *>(w.soup.data()), w.soup.size());
  out.write(reinterpret_cast<const char *>(w.disabled.data()), w.disabled.size());
  out.close();
  if (!out) throw std::runtime_error("checkpoint write failed " + path);
  if (std::rename(temporary.c_str(), path.c_str()) != 0)
    throw std::runtime_error("checkpoint rename failed " + path);
}

World load_world(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read checkpoint " + path);
  char magic[8]; in.read(magic, 8);
  if (!in || std::memcmp(magic, "BFFLAB02", 8)) throw std::runtime_error("unsupported checkpoint format");
  World w;
  w.population = get_u64(in); w.length = get_u64(in); validate_dimensions(w.population, w.length);
  w.budget = get_u64(in); w.interaction = get_u64(in); w.seed = get_u64(in);
  w.executed_steps = get_u64(in); w.active_ops = get_u64(in);
  w.changed_bytes = get_u64(in); w.exact_copy_events = get_u64(in); w.mutation = get_double(in);
  for (auto &value : w.rng.state) value = get_u64(in);
  const U64 kind = get_u64(in);
  if (kind > 2 || !std::isfinite(w.mutation) || w.mutation < 0 || w.mutation > 1 ||
      w.budget == 0 || w.budget > 100000000)
    throw std::runtime_error("invalid checkpoint parameters");
  w.lesion.kind = kind == 1 ? "focal" : kind == 2 ? "random" : "none";
  w.lesion.severity = get_double(in); w.lesion.exposure = get_double(in); w.lesion.seed = get_u64(in);
  if (!std::isfinite(w.lesion.severity) || w.lesion.severity < 0 || w.lesion.severity > 1 ||
      !std::isfinite(w.lesion.exposure) || w.lesion.exposure < 0 || w.lesion.exposure > 1 ||
      std::all_of(w.rng.state.begin(), w.rng.state.end(), [](U64 value) { return value == 0; }))
    throw std::runtime_error("invalid checkpoint lesion or RNG state");
  w.soup.resize(w.population * w.length); w.disabled.resize(w.soup.size());
  in.read(reinterpret_cast<char *>(w.soup.data()), w.soup.size());
  in.read(reinterpret_cast<char *>(w.disabled.data()), w.disabled.size());
  if (!in || in.peek() != EOF || std::any_of(w.disabled.begin(), w.disabled.end(), [](uint8_t v) { return v > 1; }))
    throw std::runtime_error("truncated, oversized or invalid checkpoint");
  return w;
}

U64 affected_slots(const World &w) {
  U64 count = 0;
  for (U64 i = 0; i < w.population; ++i)
    count += std::any_of(w.disabled.begin() + i * w.length, w.disabled.begin() + (i + 1) * w.length,
                         [](uint8_t v) { return v; });
  return count;
}
U64 unique_tapes(const World &w) {
  std::set<std::string> unique;
  for (U64 i = 0; i < w.population; ++i)
    unique.emplace(reinterpret_cast<const char *>(w.soup.data() + i * w.length), w.length);
  return unique.size();
}
double entropy(const World &w) {
  std::array<U64, 256> counts{};
  for (const uint8_t v : w.soup) ++counts[v];
  double result = 0;
  for (U64 n : counts) if (n) { const double p = double(n) / w.soup.size(); result -= p * std::log2(p); }
  return result;
}
std::string hex(const Bytes &v) {
  static const char *digits = "0123456789abcdef";
  std::string result; result.reserve(v.size() * 2);
  for (uint8_t b : v) { result += digits[b >> 4]; result += digits[b & 15]; }
  return result;
}
void metadata_json(std::ostream &out, const World &w) {
  out << "\"schema_version\":2,\"population\":" << w.population << ",\"length\":" << w.length
      << ",\"steps\":" << w.budget << ",\"interaction\":" << w.interaction << ",\"seed\":" << w.seed
      << ",\"mutation\":" << std::setprecision(17) << w.mutation
      << ",\"vm\":\"bff_noheads_bounded_scans\",\"lesion\":{\"kind\":\"" << w.lesion.kind
      << "\",\"severity\":" << w.lesion.severity << ",\"exposure\":" << w.lesion.exposure
      << ",\"seed\":" << w.lesion.seed << ",\"disabled_sites\":"
      << std::count(w.disabled.begin(), w.disabled.end(), uint8_t(1))
      << ",\"affected_slots\":" << affected_slots(w) << "}";
}

struct Options {
  std::string command = "run";
  std::map<std::string, std::string> values;
  bool seeded = false;
  bool has(const std::string &k) const { return values.count(k); }
  std::string str(const std::string &k, const std::string &fallback = "") const {
    auto it = values.find(k); return it == values.end() ? fallback : it->second;
  }
  U64 number(const std::string &k, U64 fallback) const {
    if (!has(k)) return fallback;
    const std::string value = str(k);
    if (value.empty() || value[0] == '-') throw std::runtime_error("invalid nonnegative integer --" + k);
    size_t pos = 0; U64 result = std::stoull(value, &pos);
    if (pos != value.size()) throw std::runtime_error("invalid integer --" + k);
    return result;
  }
  double real(const std::string &k, double fallback) const {
    if (!has(k)) return fallback;
    size_t pos = 0; const auto value = str(k); double result = std::stod(value, &pos);
    if (pos != value.size() || !std::isfinite(result) || result < 0 || result > 1)
      throw std::runtime_error("--" + k + " must be between 0 and 1");
    return result;
  }
};
Options options(int argc, char **argv) {
  Options o;
  int i = 1;
  if (i < argc && argv[i][0] != '-') o.command = argv[i++];
  const std::set<std::string> known = {"population", "length", "steps", "seed", "interactions", "checkpoint",
    "metrics", "load", "lesion", "severity", "exposure", "lesion-seed", "mutation", "metric-every",
    "samples", "trials", "output"};
  for (; i < argc; ++i) {
    const std::string token = argv[i];
    if (token == "--self-test") { o.command = "self-test"; continue; }
    if (token == "--help" || token == "-h") { o.command = "help"; continue; }
    if (token == "--seeded") { o.seeded = true; continue; }
    if (token.size() < 3 || token.substr(0, 2) != "--" || !known.count(token.substr(2)))
      throw std::runtime_error("unknown option " + token);
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + token);
    if (o.values.count(token.substr(2))) throw std::runtime_error("duplicate option " + token);
    o.values[token.substr(2)] = argv[++i];
  }
  return o;
}
Lesion lesion_options(const Options &o) {
  Lesion l;
  l.kind = o.str("lesion", "none"); l.severity = o.real("severity", 0.25);
  l.exposure = o.real("exposure", 1); l.seed = o.number("lesion-seed", 991);
  if (l.kind != "none" && l.kind != "focal" && l.kind != "random")
    throw std::runtime_error("lesion must be none, focal or random");
  return l;
}
World world_from_options(const Options &o) {
  World w;
  if (o.has("load")) {
    w = load_world(o.str("load"));
    for (const std::string k : {"population", "length", "seed"})
      if (o.has(k) && o.number(k, 0) != (k == "population" ? w.population : k == "length" ? w.length : w.seed))
        throw std::runtime_error("--" + k + " cannot change loaded checkpoint");
    if (o.seeded) throw std::runtime_error("--seeded cannot modify a loaded checkpoint");
    if (o.has("steps")) w.budget = o.number("steps", 8192);
    if (o.has("mutation")) w.mutation = o.real("mutation", 1.0 / 4096);
  } else {
    w = create_world(o.number("population", 1024), o.number("length", 64),
                     o.number("steps", 8192), o.number("seed", 1), o.real("mutation", 1.0 / 4096), o.seeded);
  }
  if (!w.budget || w.budget > 100000000) throw std::runtime_error("steps must be 1..100000000");
  apply_lesion(w, lesion_options(o));
  return w;
}

void run(const Options &o) {
  if (!o.has("checkpoint") || !o.has("metrics")) throw std::runtime_error("run requires --checkpoint and --metrics");
  World w = world_from_options(o);
  const U64 count = o.number("interactions", 100000);
  const U64 every = o.number("metric-every", 10000);
  if (!every || count > std::numeric_limits<U64>::max() - w.interaction)
    throw std::runtime_error("invalid metric interval or interaction count");
  std::ofstream metrics(o.str("metrics"), std::ios::trunc);
  if (!metrics) throw std::runtime_error("cannot write metrics");
  metrics << "interaction,executed_steps,active_ops,changed_bytes,exact_copy_events,unique_tapes,byte_entropy,disabled_sites,affected_slots\n";
  auto record = [&]() {
    metrics << w.interaction << ',' << w.executed_steps << ',' << w.active_ops << ',' << w.changed_bytes << ','
            << w.exact_copy_events << ',' << unique_tapes(w) << ',' << std::setprecision(12) << entropy(w) << ','
            << std::count(w.disabled.begin(), w.disabled.end(), uint8_t(1)) << ',' << affected_slots(w) << '\n';
    metrics.flush();
    if (!metrics) throw std::runtime_error("metrics write failed");
  };
  record();
  Bytes tape(2 * w.length), mask(tape.size()), before(tape.size());
  for (U64 i = 0; i < count; ++i) {
    interact(w, tape, mask, before);
    if ((i + 1) % every == 0 || i + 1 == count) record();
  }
  save_world(w, o.str("checkpoint"));
  std::cout << '{'; metadata_json(std::cout, w);
  std::cout << ",\"exact_copy_events\":" << w.exact_copy_events << ",\"executed_steps\":" << w.executed_steps << "}\n";
}

struct ProbeResult { bool retained = false, copied = false; };
ProbeResult probe(const Bytes &source, const Bytes &source_mask, Bytes food, U64 budget) {
  const U64 length = source.size();
  if (food == source) food[0] ^= 1;
  Bytes tape = source; tape.insert(tape.end(), food.begin(), food.end());
  Bytes mask = source_mask; mask.resize(2 * length, 0);
  execute(tape, mask, budget);
  const bool retained = equal_half(tape, 0, source, 0, length);
  return {retained, retained && equal_half(tape, length, source, 0, length)};
}
Bytes random_food(Rng &rng, U64 length) {
  Bytes food(length); for (auto &v : food) v = static_cast<uint8_t>(rng.next()); return food;
}

void assay(const Options &o) {
  if (!o.has("load") || !o.has("output")) throw std::runtime_error("assay requires --load and --output");
  World w = load_world(o.str("load"));
  if (o.has("steps")) w.budget = o.number("steps", w.budget);
  if (!w.budget || w.budget > 100000000) throw std::runtime_error("steps must be 1..100000000");
  apply_lesion(w, lesion_options(o));
  const U64 trials = o.number("trials", 32), samples = std::min(o.number("samples", 128), w.population);
  if (trials < 1 || trials > 100000 || samples < 1 || samples > 100000 || trials > 10000000ULL / samples)
    throw std::runtime_error("assay trials and samples must be positive with <=10000000 probes");
  const U64 assay_seed = o.number("seed", 71831);
  Rng sampling(assay_seed ^ 0x53414d504c4553ULL), foods(assay_seed ^ 0x464f4f4453ULL);
  std::vector<U64> slots(w.population); std::iota(slots.begin(), slots.end(), 0); shuffle(slots, sampling);
  struct Row { U64 slot, copied = 0, retained = 0, second = 0; std::string source_hex; };
  std::vector<Row> rows;
  U64 copies = 0, retained = 0, second = 0, qualifying = 0;
  for (U64 s = 0; s < samples; ++s) {
    const U64 slot = slots[s];
    const Bytes source(w.soup.begin() + slot * w.length, w.soup.begin() + (slot + 1) * w.length);
    const Bytes mask(w.disabled.begin() + slot * w.length, w.disabled.begin() + (slot + 1) * w.length);
    Row row{slot, 0, 0, 0, hex(source)};
    for (U64 t = 0; t < trials; ++t) {
      Bytes first_food = random_food(foods, w.length), second_food = random_food(foods, w.length);
      const ProbeResult result = probe(source, mask, std::move(first_food), w.budget);
      const ProbeResult child = probe(source, Bytes(w.length, 0), std::move(second_food), w.budget);
      row.retained += result.retained; row.copied += result.copied;
      row.second += result.copied && child.copied;
    }
    copies += row.copied; retained += row.retained; second += row.second;
    qualifying += row.copied >= 3 && row.second >= 1;
    rows.push_back(row);
  }
  std::ofstream out(o.str("output"), std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write assay output");
  out << '{'; metadata_json(out, w);
  out << ",\"assay_seed\":" << assay_seed << ",\"trials\":" << trials << ",\"samples\":" << samples
      << ",\"exact_copy_successes\":" << copies << ",\"source_retained_trials\":" << retained
      << ",\"second_generation_successes\":" << second << ",\"qualifying_samples\":" << qualifying
      << ",\"assay_policy\":\"unmutated_source_left_independent_random_food_right; child_mask_healthy; fixed_step_budget\",\"per_sample\":[";
  for (U64 i = 0; i < rows.size(); ++i) {
    const auto &row = rows[i]; if (i) out << ',';
    out << "{\"slot\":" << row.slot << ",\"source_hex\":\"" << row.source_hex << "\",\"trials\":" << trials
        << ",\"exact_copy_successes\":" << row.copied << ",\"source_retained_trials\":" << row.retained
        << ",\"second_generation_successes\":" << row.second << '}';
  }
  out << "]}\n";
  if (!out) throw std::runtime_error("assay write failed");
  std::cout << "{\"samples\":" << samples << ",\"trials\":" << trials << ",\"exact_copy_successes\":" << copies
            << ",\"qualifying_samples\":" << qualifying << ",\"second_generation_successes\":" << second << "}\n";
}

void inspect(const Options &o) {
  if (!o.has("load")) throw std::runtime_error("inspect requires --load");
  const World w = load_world(o.str("load"));
  std::ostringstream result;
  result << '{'; metadata_json(result, w);
  result << ",\"unique_tapes\":" << unique_tapes(w) << ",\"byte_entropy\":" << entropy(w)
         << ",\"executed_steps\":" << w.executed_steps << ",\"active_ops\":" << w.active_ops
         << ",\"changed_bytes\":" << w.changed_bytes << ",\"exact_copy_events\":" << w.exact_copy_events
         << ",\"rng_state\":[";
  for (unsigned i = 0; i < 4; ++i) { if (i) result << ','; result << w.rng.state[i]; }
  result << "]}\n";
  if (o.has("output")) {
    std::ofstream out(o.str("output")); if (!out) throw std::runtime_error("cannot write inspect output");
    out << result.str(); if (!out) throw std::runtime_error("inspect write failed");
  } else std::cout << result.str();
}

void check(bool ok, const std::string &message) { if (!ok) throw std::runtime_error("self-test failed: " + message); }
void self_test() {
  {
    Bytes tape{'<', '+', '}', '.', 0, 0, 0, 255};
    execute(tape, Bytes(tape.size(), 0), 2);
    check(tape.back() == 0, "head and uint8 wrap");
  }
  {
    Bytes tape{'<', '[', '-', ']', 0, 0, 0, 3};
    const auto result = execute(tape, Bytes(tape.size(), 0), 1000);
    check(tape[7] == 0 && result.steps > 9 && result.steps < 1000, "loop decrements live data");
  }
  {
    Bytes tape{'<', '[', 0, 0, 0, 0, 0, 0};
    const auto result = execute(tape, Bytes(tape.size(), 0), 4);
    check(result.steps <= 4, "bounded unmatched loop scan");
  }
  {
    Bytes tape{'>', '>', '>', ',', ']', 0, 0, 0};
    const auto result = execute(tape, Bytes(tape.size(), 0), 100);
    check(tape[3] == '>' && result.steps == 9, "live bracket scan after program overwrites itself");
  }
  {
    const Bytes control = control_program(64);
    Rng rng(4001);
    for (unsigned trial = 0; trial < 8; ++trial) {
      const auto result = probe(control, Bytes(64, 0), random_food(rng, 64), 8192);
      check(result.copied, "engineered exact64 positive control");
    }
    const auto lesioned = probe(control, Bytes(64, 1), random_food(rng, 64), 8192);
    check(!lesioned.copied, "fully disabled source cannot replicate");
  }
  {
    World a = create_world(16, 64, 1024, 77, 0.01, false), b = a;
    Lesion lesion; lesion.kind = "focal"; lesion.exposure = .5; lesion.severity = .25; lesion.seed = 33;
    apply_lesion(a, lesion); lesion.kind = "random"; apply_lesion(b, lesion);
    check(a.rng.state == b.rng.state && a.soup == b.soup, "lesion independent of genome and environment RNG");
    check(std::count(a.disabled.begin(), a.disabled.end(), uint8_t(1)) == 128 &&
          std::count(b.disabled.begin(), b.disabled.end(), uint8_t(1)) == 128 &&
          affected_slots(a) == 8 && affected_slots(b) == 8, "matched lesion doses");
    const Bytes original_mask = a.disabled;
    Bytes tape(128), mask(128), before(128);
    for (unsigned i = 0; i < 30; ++i) { interact(a, tape, mask, before); interact(b, tape, mask, before); }
    check(a.rng.state == b.rng.state, "shared future random schedule despite lesion effects");
    check(a.disabled == original_mask, "acquired lesion remains resident and is not copied");
  }
  std::cout << "self-test passed: wraps, live loops, scan bounds, exact reproduction, lesions, random schedule\n";
}

int main(int argc, char **argv) {
  try {
    const Options o = options(argc, argv);
    if (o.command == "self-test") self_test();
    else if (o.command == "run") run(o);
    else if (o.command == "assay") assay(o);
    else if (o.command == "inspect") inspect(o);
    else if (o.command == "help")
      std::cout << "BFF Lesion Lab\nrun --interactions N --checkpoint PATH --metrics PATH [--load PATH] [--seeded]\n"
                << "  --population 1024 --length 64 --steps 8192 --seed 1 --mutation 0.000244140625 --metric-every 10000\n"
                << "  --lesion none|focal|random --severity 0.25 --exposure 1 --lesion-seed 991\n"
                << "assay --load PATH --trials 32 --samples 128 --output PATH [--seed 71831] [lesion options]\n"
                << "inspect --load PATH [--output PATH]\n--self-test\n";
    else throw std::runtime_error("unknown command " + o.command);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n'; return 1;
  }
}

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#define NOMINMAX
#include <fcntl.h>
#include <intrin.h>
#include <io.h>
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <cuda_runtime.h>

#include "Kernel.cuh"
#include "filter.h"
#include "big_int/big_int_host.h"
#include "lib/CudaHashLookup.h"
#include "lib/util.h"
#include "xor_filter.cuh"

using namespace std;

#if defined(_WIN32) || defined(_WIN64)
static std::wstring utf8_to_wstring(const std::string& s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), &w[0], n);
    return w;
}
#endif

enum class RunKind { Brain, Priv };
enum class BrainHash : uint8_t { Sha256 = 0, Sha3 = 1, Keccak = 2, Blake2b = 3, Raw = 4 };

static RunKind runKind = RunKind::Brain;
static BrainHash brainHash = BrainHash::Sha256;
vector<int> Iterations = { 1 };
vector<int> DeviceList = { 0 };
vector<string> inputFiles;
vector<string> FolderList;
vector<string> MaskList;
vector<string> MaskFileList;
array<string, 4> CustomSets{ string(""), string(""), string(""), string("") };
string fileResult = "result.txt";
string log_file = "log.txt";
string deviceSpec;
string start_point;
string end_point;
uint64_t step = 1;
uint64_t n_number = 0;
bool use_n_count = false;
unsigned int requestedThreads = 0;
unsigned int requestedBlocks = 0;
bool useStdin = true;
bool all_file = false;
bool deleteFile = false;
bool IS_HEX = false;
bool isRandom = false;
bool seqMode = false;
bool backward = false;
bool both = false;
bool save = false;
bool silent = false;
bool isLog = false;
bool Compressed = true;
bool Uncompressed = true;
bool Segwit = true;
bool Taproot = false;
bool Ethereum = false;
bool Xpoint = false;
bool Solana = false;
bool Ton = false;
bool Ton_all = false;
bool Dot = false;
bool Aptos = false;
bool Sui = false;
bool Xrp = false;
bool Iota = false;
bool Icp = false;
bool Fil = false;
bool Xtz = false;
bool Endomorphism = false;
bool useHashTarget = false;
uint32_t hashTargetWordsHost[5] = { 0, 0, 0, 0, 0 };
uint32_t hashTargetMasksHost[5] = { 0, 0, 0, 0, 0 };
uint32_t hashTargetLenBytes = 0;
string hashTargetHex;

bool useXorCPU = false;
bool useBloomCPU = false;

static vector<string> gBloomFiles;
static vector<string> gXorCFiles;
static vector<string> gXorUFiles;
static vector<string> gXorUcFiles;
static vector<string> gXorHFiles;
static vector<string> gCpuXorFiles;
static vector<string> gCpuBloomFiles;

static bool gUseBloom = false;
static bool gUseXorC = false;
static bool gUseXorU = false;
static bool gUseXorUc = false;
static bool gUseXorH = false;

thread_local int DEVICE_NR = 0;
static thread_local unsigned int BLOCK_THREADS = 0;
static thread_local unsigned int BLOCK_NUMBER = 0;
static thread_local uint64_t workSize = 0;
static thread_local CudaHashLookup _targetLookup;
static thread_local secp256k1_ge_storage* _dev_precomp = nullptr;
static thread_local size_t pitch = 0;

uint32_t MAX_FOUNDS = 150000;
bool g_save_output_with_gpu_prefix = false;
std::vector<std::thread> g_save_threads;
std::mutex g_save_threads_mutex;
std::atomic<uint64_t> counterTotal{ 0 };
std::atomic<uint64_t> g_runtime_founds{ 0 };
static std::atomic<bool> isRun{ false };
static std::thread speedThread;

#ifdef _DEBUG
static unsigned int PARAM_ECMULT_WINDOW_SIZE = 4;
#else
static unsigned int PARAM_ECMULT_WINDOW_SIZE = 18;
#endif
static constexpr int PRIV_SEQ_THREAD_STEPS = 1024;

vector<string> Derivations_list;

static int activeTargetCount()
{
    int count = 0;
    if (Compressed) ++count;
    if (Uncompressed) ++count;
    if (Segwit) ++count;
    if (Taproot) ++count;
    if (Ethereum) ++count;
    if (Xpoint) ++count;
    if (Solana) ++count;
    if (Ton) ++count;
    if (Ton_all) count += 13;
    if (Dot) count += 2;
    if (Aptos) count += 3;
    if (Sui) count += 2;
    if (Xrp) count += 2;
    if (Iota) count += 2;
    if (Icp) count += 2;
    if (Fil) count += 2;
    if (Xtz) count += 2;
    return max(1, count);
}

static double speedHashMultiplier()
{
    double multiplier = static_cast<double>(activeTargetCount());
    if (runKind == RunKind::Brain) {
        multiplier *= static_cast<double>(max<size_t>(1u, Iterations.size()));
    }
    return multiplier;
}

static uint64_t privEndomorphismKeyMultiplier()
{
    const bool hasOther = Solana || Ton || Ton_all || Dot || Aptos || Sui || Xrp || Iota || Icp || Fil || Xtz;
    if (runKind == RunKind::Priv && Endomorphism && !hasOther &&
        (Compressed || Uncompressed || Segwit || Taproot || Ethereum || Xpoint)) {
        return (Taproot || Ethereum || Xpoint) && !(Compressed || Uncompressed || Segwit) ? 3ull : 6ull;
    }
    return 1ull;
}

static void printSpeed(double speed)
{
    const unsigned long long total = static_cast<unsigned long long>(counterTotal.load(std::memory_order_relaxed));
    const unsigned long long found = static_cast<unsigned long long>(g_runtime_founds.load(std::memory_order_relaxed));
    const unsigned long long fp = static_cast<unsigned long long>(false_positive.load(std::memory_order_relaxed));
    const std::string speedStr = formatDouble("%.2f", speed) + (runKind == RunKind::Brain ? " Line/s" : " Key/s");
    const std::string speedCount = formatDouble("%.2f", speed * speedHashMultiplier()) + " Hash/s";
    printf("[!] T:[%llu] | S:[%s] [%s] | F:[%llu] | F/P:[%llu] [!]           \r",
        total, speedStr.c_str(), speedCount.c_str(), found, fp);
    fflush(stdout);
}

void SpeedThreadFunc()
{
    using namespace std::chrono;
    uint64_t lastSeenKeys = counterTotal.load(std::memory_order_relaxed);
    bool baseInit = false;
    uint64_t baseKeys = 0;
    steady_clock::time_point baseTime{};

    while (isRun.load(std::memory_order_acquire)) {
        const uint64_t totalKeys = counterTotal.load(std::memory_order_relaxed);
        if (totalKeys != lastSeenKeys) {
            const auto nowTime = steady_clock::now();
            lastSeenKeys = totalKeys;
            if (!baseInit) {
                baseInit = true;
                baseKeys = totalKeys;
                baseTime = nowTime;
            }
            else {
                double elapsed = duration_cast<duration<double>>(nowTime - baseTime).count();
                if (elapsed < 0.1) {
                    elapsed = 0.1;
                }
                const uint64_t doneKeys = totalKeys - baseKeys;
                if (doneKeys > 0) {
                    printSpeed(static_cast<double>(doneKeys) / elapsed);
                }
            }
        }
        else {
            this_thread::sleep_for(milliseconds(1));
        }
    }
}

static string normalizePointForPrint(string value, size_t width, char fill)
{
    if (value.size() >= 2 && value[0] == '0' && (value[1] == 'x' || value[1] == 'X')) {
        value.erase(0, 2);
    }
    if (value.empty()) {
        value.assign(width, fill);
    }
    if (value.size() < width) {
        value.insert(value.begin(), width - value.size(), '0');
    }
    if (value.size() > width) {
        value = value.substr(value.size() - width);
    }
    transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

static void printSequentialRangeBounds()
{
    if (!seqMode) {
        return;
    }
    const char* modeLabel = runKind == RunKind::Brain ? "BRAINWALLET" : "PVK";
    const size_t width = runKind == RunKind::Brain ? 512u : 64u;
    const char* label = runKind == RunKind::Brain ? "BRAIN POINT" : "PVK";
    const string startPrint = normalizePointForPrint(start_point, width, '0');
    const string endPrint = end_point.empty()
        ? normalizePointForPrint(string(), width, backward ? '0' : 'F')
        : normalizePointForPrint(end_point, width, '0');

    if (isRandom) {
        uint64_t randomSpan = static_cast<uint64_t>(BLOCK_NUMBER) *
            static_cast<uint64_t>(BLOCK_THREADS) *
            static_cast<uint64_t>(runKind == RunKind::Priv ? PRIV_SEQ_THREAD_STEPS : THREAD_STEPS_BRAIN);
        if (use_n_count && n_number > randomSpan) {
            randomSpan = n_number;
        }
        printf("[!] starting %s sequential random mode +/- n=%llu step=0x%llx [!]\n",
            modeLabel,
            static_cast<unsigned long long>(randomSpan),
            static_cast<unsigned long long>(step));
    }
    else if (both) {
        printf("[!] starting %s sequential both mode +/- step=0x%llx [!]\n",
            modeLabel,
            static_cast<unsigned long long>(step));
    }
    else if (backward) {
        printf("[!] starting %s sequential backward mode - step=0x%llx [!]\n",
            modeLabel,
            static_cast<unsigned long long>(step));
    }
    else {
        printf("[!] starting %s sequential mode + step=0x%llx [!]\n",
            modeLabel,
            static_cast<unsigned long long>(step));
    }
    printf("START %s: %s\n", label, startPrint.c_str());
    if (!both || !end_point.empty()) {
        printf("END %s: %s\n", label, endPrint.c_str());
    }
    fflush(stdout);
}

static void printBranchStartup()
{
    if (silent) {
        printf("[!] Silent mode activated! Do not print founds on the console, only in file! [!]\n");
    }

    std::cout << "[!] Secp256k1 precompute table size: " << PARAM_ECMULT_WINDOW_SIZE << " bits\n";
    std::cout << "[!} Starting allocation " << MAX_FOUNDS << " found arrays [!]\n";
    std::cout << "[!] Active CUDA devices:";
    for (int dev : DeviceList) {
        std::cout << " " << dev;
    }
    std::cout << "\n";

    if (runKind == RunKind::Brain) {
        printf("[!] Number of passwords per thread: %d\n", THREAD_STEPS_BRAIN);
    }
    else {
        printf("[!] Number of keys per thread: %d\n", (seqMode || isRandom) ? PRIV_SEQ_THREAD_STEPS : THREAD_STEPS);
    }

    if (seqMode) {
        return;
    }

    if (!MaskList.empty() || !MaskFileList.empty()) {
        printf("[!] starting %s mask mode [!]\n", runKind == RunKind::Brain ? "BRAINWALLET" : "PVK");
        return;
    }

    if (isRandom) {
        printf("[!] using %s from included random generator\n", runKind == RunKind::Brain ? "brainwallets" : "PVK");
        return;
    }

    if (useStdin && inputFiles.empty()) {
        printf("[!] using %s from external input\n", runKind == RunKind::Brain ? "brainwallets" : "PVK");
    }
    else {
        printf("[!] using %s from file(s)\n", runKind == RunKind::Brain ? "brainwallets" : "PVK");
    }
}

static bool isHexChar(char c)
{
    return std::isxdigit(static_cast<unsigned char>(c)) != 0;
}

static string trimCopy(const string& value)
{
    size_t begin = 0;
    size_t end = value.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(value[begin]))) begin++;
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) end--;
    return value.substr(begin, end - begin);
}

static void stripCr(string& line)
{
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
}

static const unsigned char unhex_tab_host[80] = {
  0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
  0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

static inline unsigned char* unhex_host(unsigned char* str, size_t str_sz, unsigned char* unhexed, size_t unhexed_sz)
{
    int i, j;
    for (i = j = 0; i < static_cast<int>(str_sz) && j < static_cast<int>(unhexed_sz); i += 2, ++j) {
        unhexed[j] = (unhex_tab_host[str[i + 0] & 0x4f] & 0xf0) |
            (unhex_tab_host[str[i + 1] & 0x4f] & 0x0f);
    }
    return unhexed;
}

thread_local static std::unordered_map<void**, size_t> g_device_buffer_capacities;
static std::mutex g_shared_stream_read_mutex;
static std::mutex g_shared_stream_registry_mutex;
static std::unordered_set<std::istream*> g_shared_input_streams;

static inline cudaError_t ensure_device_buffer_capacity(void** device_ptr, size_t bytes)
{
    if (bytes == 0) {
        bytes = 1;
    }

    size_t& capacity = g_device_buffer_capacities[device_ptr];
    if (*device_ptr == nullptr || capacity < bytes) {
        void* new_buffer = nullptr;
        cudaError_t st = cudaMalloc(&new_buffer, bytes);
        if (st != cudaSuccess) {
            return st;
        }
        if (*device_ptr != nullptr) {
            cudaFree(*device_ptr);
        }
        *device_ptr = new_buffer;
        capacity = bytes;
    }
    return cudaSuccess;
}

static inline cudaError_t copy_to_device_grow(void** device_ptr, const void* src, size_t bytes)
{
    cudaError_t st = ensure_device_buffer_capacity(device_ptr, bytes);
    if (st != cudaSuccess) {
        return st;
    }
    if (bytes == 0) {
        return cudaSuccess;
    }
    return cudaMemcpyAsync(*device_ptr, src, bytes, cudaMemcpyHostToDevice);
}

static inline void truncate_inplace(std::string& line, const size_t max_len)
{
    if (line.size() > max_len) {
        line.resize(max_len);
    }
}

static inline bool read_trimmed_line(std::istream& stream, std::string& line, const size_t max_len)
{
    bool requires_lock = false;
    {
        std::lock_guard<std::mutex> guard(g_shared_stream_registry_mutex);
        requires_lock = (g_shared_input_streams.find(&stream) != g_shared_input_streams.end());
    }
    if (requires_lock) {
        std::lock_guard<std::mutex> guard(g_shared_stream_read_mutex);
        if (!std::getline(stream, line)) {
            return false;
        }
    }
    else if (!std::getline(stream, line)) {
        return false;
    }
    stripCr(line);
    truncate_inplace(line, max_len);
    return true;
}

static inline void tune_ifstream_buffer(std::ifstream& stream)
{
    static std::mutex s_ifstream_buf_mutex;
    static std::unordered_map<std::streambuf*, std::vector<char>> s_ifstream_bufs;

    std::streambuf* const buf = stream.rdbuf();
    if (buf == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> guard(s_ifstream_buf_mutex);
    auto it = s_ifstream_bufs.find(buf);
    if (it == s_ifstream_bufs.end()) {
        it = s_ifstream_bufs.emplace(buf, std::vector<char>(4u << 20)).first;
    }
    else if (it->second.empty()) {
        it->second.resize(4u << 20);
    }
    buf->pubsetbuf(it->second.data(), static_cast<std::streamsize>(it->second.size()));
}

static inline void reserve_batch_buffers(std::string& combined, std::vector<uint32_t>& indexes, const uint64_t output_threads, const size_t max_line_len)
{
    const size_t reserve_lines = static_cast<size_t>(std::min<uint64_t>(output_threads, 1u << 20));
    if (indexes.capacity() < reserve_lines) {
        indexes.reserve(reserve_lines);
    }
    const size_t reserve_bytes = std::min<size_t>(reserve_lines * max_line_len, 128u * 1024u * 1024u);
    if (combined.capacity() < reserve_bytes) {
        combined.reserve(reserve_bytes);
    }
}

static bool parseUint64(const string& text, uint64_t& out)
{
    try {
        size_t idx = 0;
        int base = 10;
        string value = text;
        if (value.size() > 2 && value[0] == '0' && (value[1] == 'x' || value[1] == 'X')) {
            base = 16;
        }
        out = stoull(value, &idx, base);
        return idx == value.size();
    }
    catch (...) {
        return false;
    }
}

static bool decodeHex(const string& text, string& out, string& error)
{
    string h = trimCopy(text);
    if (h.size() >= 2 && h[0] == '0' && (h[1] == 'x' || h[1] == 'X')) {
        h.erase(0, 2);
    }
    if (h.empty()) {
        out.clear();
        return true;
    }
    if ((h.size() & 1) != 0) {
        error = "hex input has odd length";
        return false;
    }
    out.clear();
    out.reserve(h.size() / 2);
    for (size_t i = 0; i < h.size(); i += 2) {
        if (!isHexChar(h[i]) || !isHexChar(h[i + 1])) {
            error = "hex input contains non-hex character";
            return false;
        }
        unsigned int byte = 0;
        std::stringstream ss;
        ss << std::hex << h.substr(i, 2);
        ss >> byte;
        out.push_back(static_cast<char>(byte));
    }
    return true;
}

static inline bool append_normalized_priv_line(
    std::string& buffer,
    std::string& combined,
    std::vector<uint32_t>& indexes,
    uint32_t& nr,
    bool input_is_hex)
{
    if (input_is_hex) {
        if (buffer.length() > 64) {
            std::string buffer2 = buffer.substr(0, 64);
            if (buffer2 != std::string(64, '0')) {
                combined += buffer2;
                indexes.emplace_back(static_cast<uint32_t>(combined.size()));
                nr++;
            }
            buffer.erase(0, 64);
        }
        if (buffer.length() > 64) {
            buffer.resize(64);
        }
        if (buffer.length() < 64) {
            buffer.insert(0, 64 - buffer.length(), '0');
        }
        if (buffer == std::string(64, '0')) {
            return false;
        }
    }
    else {
        if (buffer.length() > 32) {
            std::string buffer2 = buffer.substr(0, 32);
            if ((buffer2 != std::string(32, 0x00)) && (buffer2 != std::string(32, '0'))) {
                combined += buffer2;
                indexes.emplace_back(static_cast<uint32_t>(combined.size()));
                nr++;
            }
            buffer.erase(0, 32);
        }
        if (buffer.length() > 32) {
            buffer.resize(32);
        }
        if (buffer.length() < 32) {
            buffer.insert(0, 32 - buffer.length(), 0x00);
        }
        if ((buffer == std::string(32, 0x00)) || (buffer == std::string(32, '0'))) {
            return false;
        }
    }

    combined += buffer;
    indexes.emplace_back(static_cast<uint32_t>(combined.size()));
    nr++;
    return true;
}

static vector<int> parseDeviceSpec(const string& spec)
{
    vector<int> out;
    string token;
    stringstream ss(spec);
    while (getline(ss, token, ',')) {
        token = trimCopy(token);
        if (token.empty()) {
            continue;
        }
        size_t dash = token.find('-');
        if (dash != string::npos) {
            int a = stoi(token.substr(0, dash));
            int b = stoi(token.substr(dash + 1));
            if (a > b || a < 0) {
                throw runtime_error("invalid device range");
            }
            for (int d = a; d <= b; ++d) {
                out.push_back(d);
            }
        }
        else {
            int d = stoi(token);
            if (d < 0) {
                throw runtime_error("invalid device id");
            }
            out.push_back(d);
        }
    }
    sort(out.begin(), out.end());
    out.erase(unique(out.begin(), out.end()), out.end());
    if (out.empty()) {
        throw runtime_error("empty device list");
    }
    return out;
}

static bool parseIterations(const string& text, vector<int>& out, string& error)
{
    out.clear();
    string token;
    stringstream ss(text);
    while (getline(ss, token, ',')) {
        token = trimCopy(token);
        if (token.empty()) {
            error = "empty iteration item";
            return false;
        }
        size_t dash = token.find('-');
        if (dash == string::npos) {
            uint64_t v = 0;
            if (!parseUint64(token, v) || v == 0 || v > static_cast<uint64_t>(numeric_limits<int>::max())) {
                error = "invalid iteration value";
                return false;
            }
            out.push_back(static_cast<int>(v));
        }
        else {
            uint64_t first = 0;
            uint64_t last = 0;
            if (!parseUint64(token.substr(0, dash), first) ||
                !parseUint64(token.substr(dash + 1), last) ||
                first == 0 || last < first || last > static_cast<uint64_t>(numeric_limits<int>::max())) {
                error = "invalid iteration range";
                return false;
            }
            for (uint64_t v = first; v <= last; ++v) {
                out.push_back(static_cast<int>(v));
            }
        }
    }
    if (out.empty()) {
        error = "iteration list is empty";
        return false;
    }
    return true;
}

static bool parseDirectTarget(const string& text, string& error)
{
    string bytes;
    if (!decodeHex(text, bytes, error)) {
        return false;
    }
    if (bytes.empty() || bytes.size() > 20) {
        error = "target must be 1..20 bytes";
        return false;
    }
    hashTargetHex = trimCopy(text);
    memset(hashTargetWordsHost, 0, sizeof(hashTargetWordsHost));
    memset(hashTargetMasksHost, 0, sizeof(hashTargetMasksHost));
    for (size_t i = 0; i < bytes.size(); ++i) {
        const uint32_t word = static_cast<uint32_t>(i / 4);
        const uint32_t off = static_cast<uint32_t>((i % 4) * 8);
        hashTargetWordsHost[word] |= static_cast<uint32_t>(static_cast<unsigned char>(bytes[i])) << off;
        hashTargetMasksHost[word] |= 0xffu << off;
    }
    hashTargetLenBytes = static_cast<uint32_t>(bytes.size());
    useHashTarget = true;
    return true;
}

static void printHelp()
{
    puts(R"HELP(
[!] ================== Brainflayer CUDA by @XopMC ================== [!]

[!] [!] QUICK START [!]
[!] -h / -help                      Show this help and exit.
[!] -i FILE / -f DIR                Input source (or STDIN if not set).
[!] Default mode                    Brainwallet mode.
[!] -priv                           Switch to base private key mode.
[!] -c TYPES                        Target types (default: cus).
[!] -bf/-xu/-xc/-xuc/-xh            Load GPU filters.
[!] -hash HEX / -target HEX         Direct hash target.
[!] -save                           Save hashes in formatted addresses output.

[!] [!] MAIN MODES [!] [!]
[!] 1) default                      Brainwallet mode.
[!] 2) -priv                        Base private key mode.

[!] Brainwallet mode is active unless -priv is set.

[!] ======================================================================
[!] GLOBAL INPUT / OUTPUT / FILE FLOW
[!] ======================================================================
[!] -i FILE                         Add input file (repeatable).
[!] -f DIR                          Recursively scan directory.
[!]     -all                        With -f: include all files, not only .txt.
[!]     -delete                     Delete processed files.

[!] -o FILE                         Output file (default: result.txt).
[!] -save                           Save hashes in formatted addresses output.
[!]                                 Without -save: output matched hash payloads.
[!]                                 With -save: output currency addresses for target type.
[!] -silent                         Do not print founds to console.
[!] -log FILE                       Sequential progress snapshot, overwritten each round.

[!] ======================================================================
[!] INPUT FORMAT / HASH TARGETS / FILTERS / CURVES / GPU
[!] ======================================================================
[!] -hex                            Unified semantics:
[!]                                 ON  -> input lines are HEX-encoded bytes.
[!]                                 OFF -> input lines are used directly (raw).

[!] -c TYPES                        Choose target families to verify.
[!]                                 Multiple letters allowed, example: -c cusex
[!]                                 Commas/spaces are ignored, example: -c c,u,s

[!] [!] TYPES FOR -c [!]
[!] c - BTC compressed hash160
[!] u - BTC uncompressed hash160
[!] s - BTC segwit hash160
[!] r - BTC taproot hash
[!] e - Ethereum address
[!] x - secp256k1 pubkey X coordinate
[!] t - TON popular variants
[!] T - TON all variants
[!] S - Solana
[!] d - DOT (ed25519 + sr25519)
[!] f - Filecoin
[!] i - IOTA (ed25519 + secp256k1)
[!] A - Aptos
[!] U - SUI
[!] X - XRP
[!] I - ICP
[!] Z - XTZ

[!] Default -c (if not set): cus

[!] [!] FILTERS [!]
[!] -bf PATH                        Bloom filter(s).
[!] -xu PATH                        XOR uncompressed filter(s).
[!] -xc PATH                        XOR compressed filter(s).
[!] -xuc PATH                       XOR ultra-compressed filter(s).
[!] -xh PATH                        XOR HC filter(s).
[!] -hash HEX                       Fast hash160 prefix match (1..20 bytes, 2..40 hex chars).
[!] -target HEX                     Alias for -hash.
[!] -xx PATH                        Optional extra CPU XOR Uncompressed verification.
[!] -xb PATH                        Extra CPU Bloom verification.

[!] [!] GPU PARAMS [!]
[!] -device LIST                    CUDA device list, e.g. 0 or 0,1,3 or 0-3.
[!] -b N                            Blocks.
[!] -t N                            Threads per block.

[!] ======================================================================
[!] BRAINWALLET MODE
[!] ======================================================================
[!] This is the default mode. No mode flag is required.

[!] Hashing:
[!] -sha256                         SHA-256 brain hash (default).
[!] -sha3                           SHA3-256 brain hash.
[!] -keccak                         Keccak-256 brain hash.
[!] -blake2b                        BLAKE2b-256 brain hash.
[!] -raw                            Use input bytes as the 32-byte scalar.
[!] -iter LIST                      Iterations list/range, example: 1,4,6-10.
[!]                                 -raw accepts only effective iteration 1 and input/mask sources.

[!] Candidate sources:
[!] STDIN                           Used when no -i/-f/seq/random/mask source is set.
[!] -i FILE                         Brainwallets from file.
[!] -f DIR                          Brainwallet files from directory.
[!] -hex                            Input lines are hex-encoded bytes.
[!] -start HEX [-end HEX]           Sequential 2048-bit brain point range.
[!] -random                         Random generated brain points.
[!] -n N                            With -random: +/- span before changing point.
[!] -mask MASK                      GPU mask brute force.
[!] -mask-file FILE                 One mask per line.
[!] -cs1/-cs2/-cs3/-cs4 CHARS       Custom mask charsets for ?1..?4.

[!] [!] MASK BRUTE MODE [!]
[!] Mask mode is a brainwallet candidate source. GPU builds candidate
[!] strings from the mask, then checks them with selected brain hash,
[!] -iter list, filters and -c targets.
[!] Use mask mode instead of stdin/file/seq/random sources.
[!] Multiple -mask values and -mask-file lines are processed in order.
[!] Multi-GPU splits mask ordinal ranges across selected devices.
[!]
[!] Mask tokens:
[!] ?l                              abcdefghijklmnopqrstuvwxyz
[!] ?u                              ABCDEFGHIJKLMNOPQRSTUVWXYZ
[!] ?d                              0123456789
[!] ?h                              0123456789abcdef
[!] ?H                              0123456789ABCDEF
[!] ?s                              Space and ASCII symbols
[!] ?a                              Printable ASCII 0x20..0x7e
[!] ??                              Literal question mark
[!] ?1/?2/?3/?4                     Custom charsets from -cs1..-cs4
[!]
[!] Literals are copied as-is. Max mask length is 64 positions.
[!] Total charset storage per mask is 4096 bytes. Candidate count must
[!] fit uint64.
[!]
[!] Mask examples:
[!] Brainflayer-CUDA.exe -mask pass?d?d?d -sha256 -c c -bf targets.blf
[!] Brainflayer-CUDA.exe -mask admin?l?l?d -iter 1,2,4 -c cus -save
[!] Brainflayer-CUDA.exe -cs1 abcDEF123 -mask key?1?1?1?1 -c u -hash HASH
[!] Brainflayer-CUDA.exe -mask-file masks.txt -iter 1,2,4 -c c -bf targets.blf

[!] ======================================================================
[!] PRIVATE KEY MODE
[!] ======================================================================
[!] Enable:
[!] -priv                           Base private key mode.

[!] Candidate sources:
[!] STDIN                           Used when no -i/-f/seq/random source is set.
[!] -i FILE                         Private keys from file.
[!] -f DIR                          Private key files from directory.
[!] -hex                            Input lines are hex-encoded bytes.
[!] -start HEX [-end HEX]           Sequential private key range.
[!] -random                         Random 32-byte private keys.
[!] -n N                            With -random: +/- span before changing point.
[!] -em                             Enable secp256k1 endomorphism for -priv seq/random.
[!]                                 -priv -random enables it by default.

[!] ======================================================================
[!] SEQUENTIAL FLAGS (default brainwallet mode and -priv)
[!] ======================================================================
[!] -start VALUE                    Start point (enables seq mode).
[!] -end VALUE                      End point.
[!] -step HEX                       Step, default 1.
[!] -back                           Backward seq branch.
[!] -both                           Without -end: +/- branch around start.
[!]                                 With -start/-end: + from start, - from end.
[!] -random                         Random branch; with -start/-end picks points inside range.
[!] -n N                            Random +/- span before changing the random point.
[!] -log FILE                       Overwrite progress snapshot after each seq round.
[!] -em                             Enable secp256k1 endomorphism for -priv seq/random.
[!]                                 -priv -random enables it by default.

[!] [!] SEQ POINT LENGTH / FORMAT [!]
[!] -priv                           64 hex chars, 256-bit big-endian.
[!] Default brainwallet mode        512 hex chars, 2048-bit big-endian brain point.
[!]                                 Values are left-padded with zero.
[!]                                 Range endpoints are inclusive.
[!]                                 Multi-GPU splits range across selected devices.

[!] Examples:
[!] Brainflayer-CUDA.exe -i brain.txt -c cus -bf targets.blf -save
[!] Brainflayer-CUDA.exe -sha3 -iter 1,4,6-10 -i brain.txt -c c -hash HASH
[!] Brainflayer-CUDA.exe -hex -i hex_input.txt -c u -bf full.blf -save
[!] Brainflayer-CUDA.exe -priv -start 1 -end ffffff -c c -hash HASH -save
[!] Brainflayer-CUDA.exe -priv -random -n 1000000 -c c -bf targets.blf

[!] ======================================================================
)HELP");
}

static bool parseTargetLetters(const string& letters, string& error)
{
    Compressed = Uncompressed = Segwit = Taproot = false;
    Ethereum = Xpoint = Solana = Ton = Ton_all = false;
    Dot = Aptos = Sui = Xrp = false;
    Iota = Icp = Fil = Xtz = false;

    for (char raw : letters) {
        if (raw == ',' || std::isspace(static_cast<unsigned char>(raw))) {
            continue;
        }
        switch (raw) {
        case 'c': Compressed = true; break;
        case 'u': Uncompressed = true; break;
        case 's': Segwit = true; break;
        case 'r': Taproot = true; break;
        case 'e': Ethereum = true; break;
        case 'x': Xpoint = true; break;
        case 'S': Solana = true; break;
        case 't': Ton = true; break;
        case 'T': Ton_all = true; break;
        case 'd': Dot = true; break;
        case 'f': Fil = true; break;
        case 'i': Iota = true; break;
        case 'A': Aptos = true; break;
        case 'U': Sui = true; break;
        case 'X': Xrp = true; break;
        case 'I': Icp = true; break;
        case 'Z': Xtz = true; break;
        default:
            error = string("unknown target letter: ") + raw;
            return false;
        }
    }
    if (!(Compressed || Uncompressed || Segwit || Taproot ||
        Ethereum || Xpoint || Solana || Ton || Ton_all ||
        Dot || Aptos || Sui || Xrp || Iota ||
        Icp || Fil || Xtz)) {
        error = "empty target set";
        return false;
    }
    return true;
}

static bool parseArgs(int argc, char** argv, bool& help, string& error)
{
    int hashFlags = 0;
    help = false;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        auto needValue = [&](const string& name) -> string {
            if (i + 1 >= argc) {
                throw runtime_error("missing value after " + name);
            }
            return string(argv[++i]);
        };

        try {
            if (arg == "-h" || arg == "-help") {
                help = true;
                return true;
            }
            else if (arg == "-priv") {
                runKind = RunKind::Priv;
            }
            else if (arg == "-sha256") {
                brainHash = BrainHash::Sha256;
                hashFlags++;
            }
            else if (arg == "-sha3") {
                brainHash = BrainHash::Sha3;
                hashFlags++;
            }
            else if (arg == "-keccak") {
                brainHash = BrainHash::Keccak;
                hashFlags++;
            }
            else if (arg == "-blake2b") {
                brainHash = BrainHash::Blake2b;
                hashFlags++;
            }
            else if (arg == "-raw") {
                brainHash = BrainHash::Raw;
                hashFlags++;
            }
            else if (arg == "-iter") {
                string err;
                if (!parseIterations(needValue(arg), Iterations, err)) {
                    error = err;
                    return false;
                }
            }
            else if (arg == "-i") {
                inputFiles.push_back(needValue(arg));
                useStdin = false;
            }
            else if (arg == "-f") {
                FolderList.push_back(needValue(arg));
                useStdin = false;
            }
            else if (arg == "-all") {
                all_file = true;
            }
            else if (arg == "-delete") {
                deleteFile = true;
            }
            else if (arg == "-hex") {
                IS_HEX = true;
            }
            else if (arg == "-start") {
                start_point = needValue(arg);
                seqMode = true;
                useStdin = false;
            }
            else if (arg == "-end") {
                end_point = needValue(arg);
                seqMode = true;
                useStdin = false;
            }
            else if (arg == "-step") {
                if (!parseUint64(needValue(arg), step) || step == 0) {
                    error = "invalid range step";
                    return false;
                }
            }
            else if (arg == "-back") {
                backward = true;
            }
            else if (arg == "-both") {
                both = true;
            }
            else if (arg == "-n") {
                if (!parseUint64(needValue(arg), n_number) || n_number == 0) {
                    error = "invalid random span";
                    return false;
                }
                use_n_count = true;
            }
            else if (arg == "-random") {
                isRandom = true;
                useStdin = false;
            }
            else if (arg == "-em") {
                Endomorphism = true;
            }
            else if (arg == "-mask") {
                MaskList.push_back(needValue(arg));
                useStdin = false;
            }
            else if (arg == "-mask-file") {
                MaskFileList.push_back(needValue(arg));
                useStdin = false;
            }
            else if (arg == "-cs1" || arg == "-cs2" || arg == "-cs3" || arg == "-cs4") {
                int idx = arg[3] - '1';
                CustomSets[idx] = needValue(arg);
            }
            else if (arg == "-c") {
                if (!parseTargetLetters(needValue(arg), error)) {
                    return false;
                }
            }
            else if (arg == "-hash" || arg == "-target") {
                if (!parseDirectTarget(needValue(arg), error)) {
                    return false;
                }
            }
            else if (arg == "-bf") {
                add_filter_path(needValue(arg).c_str(), gBloomFiles, gUseBloom, exts_xb, true);
            }
            else if (arg == "-xc") {
                add_filter_path(needValue(arg).c_str(), gXorCFiles, gUseXorC, exts_xc, true);
            }
            else if (arg == "-xu") {
                add_filter_path(needValue(arg).c_str(), gXorUFiles, gUseXorU, exts_xu, true);
            }
            else if (arg == "-xuc") {
                add_filter_path(needValue(arg).c_str(), gXorUcFiles, gUseXorUc, exts_xuc, true);
            }
            else if (arg == "-xh") {
                add_filter_path(needValue(arg).c_str(), gXorHFiles, gUseXorH, exts_xh, true);
            }
            else if (arg == "-xx") {
                add_filter_path(needValue(arg).c_str(), gCpuXorFiles, useXorCPU, exts_xu, true);
            }
            else if (arg == "-xb") {
                add_filter_path(needValue(arg).c_str(), gCpuBloomFiles, useBloomCPU, exts_xb, true);
            }
            else if (arg == "-device") {
                deviceSpec = needValue(arg);
                DeviceList = parseDeviceSpec(deviceSpec);
            }
            else if (arg == "-b") {
                uint64_t v = 0;
                if (!parseUint64(needValue(arg), v) || v == 0 || v > numeric_limits<unsigned int>::max()) {
                    error = "invalid block count";
                    return false;
                }
                requestedBlocks = static_cast<unsigned int>(v);
            }
            else if (arg == "-t") {
                uint64_t v = 0;
                if (!parseUint64(needValue(arg), v) || v == 0 || v > numeric_limits<unsigned int>::max()) {
                    error = "invalid thread count";
                    return false;
                }
                requestedThreads = static_cast<unsigned int>(v);
            }
            else if (arg == "-o") {
                fileResult = needValue(arg);
            }
            else if (arg == "-log") {
                log_file = needValue(arg);
                isLog = true;
            }
            else if (arg == "-save") {
                save = true;
            }
            else if (arg == "-silent") {
                silent = true;
            }
            else {
                error = "unknown option: " + arg;
                return false;
            }
        }
        catch (const exception& ex) {
            error = ex.what();
            return false;
        }
    }

    if (hashFlags > 1) {
        error = "only one brain hash flag can be active";
        return false;
    }
    if (brainHash == BrainHash::Raw) {
        if (Iterations.size() != 1 || Iterations[0] != 1) {
            error = "raw brain hash accepts only effective iteration 1";
            return false;
        }
        if (seqMode) {
            error = "raw brain hash does not support sequential mode";
            return false;
        }
        if (isRandom) {
            error = "raw brain hash does not support random mode";
            return false;
        }
    }
    if (use_n_count && !isRandom) {
        error = "n is only supported with random mode";
        return false;
    }
    if (isRandom && !use_n_count) {
        cerr << "[!] random source changes point after one +/- launch span; stop with Ctrl+C if needed [!]\n";
    }
    if (runKind == RunKind::Priv && isRandom) {
        Endomorphism = true;
    }
    if (!gUseBloom && !gUseXorC && !gUseXorU && !gUseXorUc && !gUseXorH && !useHashTarget) {
        error = "set at least one GPU filter or direct target";
        return false;
    }
    return true;
}

static vector<string> scanDir(const string& directory, bool allFiles)
{
    vector<string> out;
#if defined(_WIN32) || defined(_WIN64)
    string search = directory + "\\*";
    WIN32_FIND_DATAA data{};
    HANDLE handle = FindFirstFileA(search.c_str(), &data);
    if (handle == INVALID_HANDLE_VALUE) {
        throw runtime_error("input directory not found: " + directory);
    }
    do {
        string name = data.cFileName;
        if (name == "." || name == "..") {
            continue;
        }
        string full = directory + "\\" + name;
        if (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            vector<string> nested = scanDir(full, allFiles);
            out.insert(out.end(), nested.begin(), nested.end());
        }
        else if (allFiles || (name.size() >= 4 && name.substr(name.size() - 4) == ".txt")) {
            out.push_back(full);
        }
    } while (FindNextFileA(handle, &data));
    FindClose(handle);
#else
    DIR* dir = opendir(directory.c_str());
    if (!dir) {
        throw runtime_error("input directory not found: " + directory);
    }
    struct dirent* entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        string name = entry->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        string full = directory + "/" + name;
        struct stat st {};
        if (stat(full.c_str(), &st) != 0) {
            continue;
        }
        if (S_ISDIR(st.st_mode)) {
            vector<string> nested = scanDir(full, allFiles);
            out.insert(out.end(), nested.begin(), nested.end());
        }
        else if (S_ISREG(st.st_mode) && (allFiles || (name.size() >= 4 && name.substr(name.size() - 4) == ".txt"))) {
            out.push_back(full);
        }
    }
    closedir(dir);
#endif
    sort(out.begin(), out.end());
    return out;
}

static uint64_t fileSizeIfExists(const string& path)
{
    ifstream in(path, ios::binary | ios::ate);
    if (!in) {
        return 0;
    }
    const streampos pos = in.tellg();
    if (pos <= 0) {
        return 0;
    }
    return static_cast<uint64_t>(pos);
}

static uint64_t sumExistingFileSizes(const vector<string>& files)
{
    uint64_t total = 0;
    for (const string& path : files) {
        const uint64_t size = fileSizeIfExists(path);
        if (size > numeric_limits<uint64_t>::max() - total) {
            return numeric_limits<uint64_t>::max();
        }
        total += size;
    }
    return total;
}

static bool checkDevice(int device)
{
    DEVICE_NR = device;
    BLOCK_THREADS = requestedThreads;
    BLOCK_NUMBER = requestedBlocks;
    cudaError_t cudaStatus = cudaSetDevice(DEVICE_NR);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "[!] device %d failed: %s [!]\n", DEVICE_NR, cudaGetErrorString(cudaStatus));
        return false;
    }

    cudaDeviceProp props;
    cudaStatus = cudaGetDeviceProperties(&props, DEVICE_NR);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "[!] failed to read device %d properties: %s [!]\n", DEVICE_NR, cudaGetErrorString(cudaStatus));
        return false;
    }
    fprintf(stderr, "[!] Using device: %d\n", DEVICE_NR);
    if (BLOCK_THREADS == 0) {
        BLOCK_THREADS = props.maxThreadsPerBlock / 8 * 2;
    }
    if (BLOCK_NUMBER == 0) {
        unsigned int blocksPerSm = 8;
        const char* tuneProfile = "default";
        const bool fileInputMode = !inputFiles.empty() || !FolderList.empty();
        if (runKind == RunKind::Priv) {
            if (seqMode && !isRandom && !fileInputMode) {
                const bool onlyBloom = gUseBloom && !gUseXorC && !gUseXorU && !gUseXorUc && !gUseXorH;
                const bool onlyXorUn = gUseXorU && !gUseBloom && !gUseXorC && !gUseXorUc && !gUseXorH;
                if (onlyXorUn) {
                    const uint64_t xuBytes = sumExistingFileSizes(gXorUFiles);
                    if (xuBytes >= (8ull << 30)) {
                        blocksPerSm = 16;
                    }
                    else if (xuBytes >= (2ull << 30)) {
                        blocksPerSm = 24;
                    }
                    else if (xuBytes >= (512ull << 20)) {
                        blocksPerSm = 48;
                    }
                    else if (xuBytes >= (128ull << 20)) {
                        blocksPerSm = 96;
                    }
                    else {
                        blocksPerSm = 256;
                    }
                    tuneProfile = "priv_seq_xu";
                }
                else if (onlyBloom) {
                    blocksPerSm = 256;
                    tuneProfile = "priv_seq_bf";
                }
                else if (gUseBloom && gUseXorU) {
                    blocksPerSm = 64;
                    tuneProfile = "priv_seq_bf_xu";
                }
                else {
                    blocksPerSm = 72;
                    tuneProfile = "priv_seq";
                }
            }
            else if (isRandom) {
                blocksPerSm = 40;
                tuneProfile = "priv_random";
            }
            else if (fileInputMode) {
                blocksPerSm = 4;
                tuneProfile = "priv_file";
            }
            else {
                blocksPerSm = 4;
                tuneProfile = "priv_stream";
            }
        }
        else {
            blocksPerSm = 4;
            tuneProfile = (seqMode || isRandom) ? "brain_seq_random" :
                (fileInputMode ? "brain_file" : "brain_stream");
        }

        const unsigned int threadsPerBlock = (BLOCK_THREADS == 0u) ? 256u : BLOCK_THREADS;
        const unsigned int maxOutputThreads = 16u * 1024u * 1024u;
        unsigned int maxBlocks = maxOutputThreads / max(1u, threadsPerBlock);
        if (maxBlocks == 0u) {
            maxBlocks = 1u;
        }
        unsigned int maxBlocksPerSm = maxBlocks / max(1, props.multiProcessorCount);
        if (maxBlocksPerSm == 0u) {
            maxBlocksPerSm = 1u;
        }
        if (blocksPerSm > maxBlocksPerSm) {
            blocksPerSm = maxBlocksPerSm;
        }
        BLOCK_NUMBER = props.multiProcessorCount * blocksPerSm;
        fprintf(stderr, "[!] Auto-tuned blocks profile '%s': %u per SM\n",
            tuneProfile, blocksPerSm);
    }
    workSize = static_cast<uint64_t>(BLOCK_NUMBER) * BLOCK_THREADS *
        static_cast<uint64_t>(runKind == RunKind::Priv
            ? ((seqMode || isRandom) ? PRIV_SEQ_THREAD_STEPS : THREAD_STEPS)
            : THREAD_STEPS_BRAIN);
    fprintf(stderr, "[!] %s (%2d procs | Blocks: %d | Threads: %d)\n",
        props.name, props.multiProcessorCount, BLOCK_NUMBER, BLOCK_THREADS);
    return true;
}

cudaError_t CudaHashLookup::setTargetBloomFilter(vector<string>& targets)
{
    cudaError_t err = cudaSuccess;
    for (int i = 0; i < static_cast<int>(targets.size()); ++i) {
        if (i >= 100) {
            return cudaErrorInvalidValue;
        }
        uint8_t* filter = nullptr;
        try {
            filter = new uint8_t[BLOOM_SIZE];
        }
        catch (std::bad_alloc&) {
            return cudaErrorMemoryAllocation;
        }
        memset(filter, 0, BLOOM_SIZE);
        FILE* file = fopen(targets[i].c_str(), "rb");
        if (!file) {
            delete[] filter;
            return cudaErrorInvalidValue;
        }
        const size_t bytesRead = fread(filter, 1, BLOOM_SIZE, file);
        const bool readError = ferror(file) != 0;
        fclose(file);
        if (readError) {
            delete[] filter;
            return cudaErrorInvalidValue;
        }
        (void)bytesRead;

        err = cudaMalloc(&_bloomFilterPtr[i], BLOOM_SIZE);
        if (err != cudaSuccess) {
            delete[] filter;
            return err;
        }
        err = cudaMemcpy(_bloomFilterPtr[i], filter, BLOOM_SIZE, cudaMemcpyHostToDevice);
        delete[] filter;
        if (err != cudaSuccess) {
            cudaFree(_bloomFilterPtr[i]);
            _bloomFilterPtr[i] = nullptr;
            return err;
        }
        err = cudaMemcpyToSymbol_BLOOM_FILTER(_bloomFilterPtr[i], i);
        if (err != cudaSuccess) {
            cudaFree(_bloomFilterPtr[i]);
            _bloomFilterPtr[i] = nullptr;
            return err;
        }
    }
    return err;
}

cudaError_t CudaHashLookup::setTargets(vector<string>& targets)
{
    cleanup();
    return setTargetBloomFilter(targets);
}

void CudaHashLookup::cleanup()
{
    for (int i = 0; i < 100; ++i) {
        if (_bloomFilterPtr[i] != nullptr) {
            cudaFree(_bloomFilterPtr[i]);
            _bloomFilterPtr[i] = nullptr;
        }
    }
}

struct XorFilterCache32 {
    std::string path;
    std::vector<uint32_t> fp;
    size_t size, arrayLength, segmentCount, segmentCountLength, segmentLength, segmentLengthMask;
};
struct XorFilterCache16 {
    std::string path;
    std::vector<uint16_t> fp;
    size_t size, arrayLength, segmentCount, segmentCountLength, segmentLength, segmentLengthMask;
};
struct XorFilterCache8 {
    std::string path;
    std::vector<uint8_t> fp;
    size_t size, arrayLength, segmentCount, segmentCountLength, segmentLength, segmentLengthMask;
};

enum class XorLoadFailureKind {
    none,
    open_failed,
    header_read_failed,
    invalid_header,
    alloc_failed,
    body_read_failed
};

struct XorLoadDiagnostics {
    const char* filter_kind = nullptr;
    std::string path;
    XorLoadFailureKind failure = XorLoadFailureKind::none;
    size_t fingerprint_bytes = 0;
    size_t size = 0;
    size_t arrayLength = 0;
    size_t segmentCount = 0;
    size_t segmentCountLength = 0;
    size_t segmentLength = 0;
    size_t segmentLengthMask = 0;
    size_t expected_body_bytes = 0;
    int error_code = 0;
};

static const char* xor_load_failure_kind_to_string(const XorLoadFailureKind kind) {
    switch (kind) {
    case XorLoadFailureKind::open_failed: return "open_failed";
    case XorLoadFailureKind::header_read_failed: return "header_read_failed";
    case XorLoadFailureKind::invalid_header: return "invalid_header";
    case XorLoadFailureKind::alloc_failed: return "alloc_failed";
    case XorLoadFailureKind::body_read_failed: return "body_read_failed";
    default: return "none";
    }
}

#if defined(_WIN64) && !defined(__CYGWIN__) && !defined(__MINGW64__)
static std::wstring normalize_windows_extended_path(const std::wstring& path) {
    if (path.empty()) {
        return path;
    }
    if (path.rfind(L"\\\\?\\", 0) == 0 || path.rfind(L"\\\\.\\", 0) == 0) {
        return path;
    }
    if (path.size() < MAX_PATH) {
        return path;
    }
    if (path.rfind(L"\\\\", 0) == 0) {
        return L"\\\\?\\UNC\\" + path.substr(2);
    }
    if (path.size() >= 2 && path[1] == L':') {
        return L"\\\\?\\" + path;
    }
    return path;
}
#endif

static FILE* open_binary_file_utf8_for_read(const std::string& path, XorLoadDiagnostics& diag) {
#if defined(_WIN64) && !defined(__CYGWIN__) && !defined(__MINGW64__)
    FILE* file = nullptr;
    if (!path.empty()) {
        std::wstring wide_path = utf8_to_wstring(path);
        if (!wide_path.empty()) {
            errno_t err = _wfopen_s(&file, wide_path.c_str(), L"rb");
            if (err == 0 && file != nullptr) {
                return file;
            }
            diag.error_code = static_cast<int>(err);

            std::wstring extended_path = normalize_windows_extended_path(wide_path);
            if (extended_path != wide_path) {
                err = _wfopen_s(&file, extended_path.c_str(), L"rb");
                if (err == 0 && file != nullptr) {
                    return file;
                }
                diag.error_code = static_cast<int>(err);
            }
        }
    }

    errno_t err = fopen_s(&file, path.c_str(), "rb");
    diag.error_code = static_cast<int>(err);
    return (err == 0) ? file : nullptr;
#else
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        diag.error_code = errno;
    }
    return file;
#endif
}

static void log_xor_load_failure(const XorLoadDiagnostics& diag) {
    std::ostringstream oss;
    oss << "[!] " << (diag.filter_kind ? diag.filter_kind : "Unknown")
        << " XOR filter load failed"
        << " [file=" << diag.path << "]"
        << " [failure=" << xor_load_failure_kind_to_string(diag.failure) << "]"
        << " [fingerprint=" << diag.fingerprint_bytes << "B]";

    if (diag.arrayLength != 0) {
        oss << " [arrayLength=" << diag.arrayLength << "]";
    }
    if (diag.expected_body_bytes != 0) {
        oss << " [body_bytes=" << diag.expected_body_bytes << "]";
    }
    if (diag.segmentCount != 0 || diag.segmentLength != 0) {
        oss << " [segmentCount=" << diag.segmentCount
            << "] [segmentLength=" << diag.segmentLength
            << "] [segmentMask=" << diag.segmentLengthMask << "]";
    }
    if (diag.error_code != 0) {
        oss << " [errno=" << diag.error_code << "]";
        const char* text = std::strerror(diag.error_code);
        if (text != nullptr) {
            oss << " [" << text << "]";
        }
    }
    std::cerr << oss.str() << std::endl;
}

template <typename FingerprintType, typename CacheEntry>
static bool load_xor_filter_cache_entry(const std::string& path, const char* filter_kind, CacheEntry& entry) {
    XorLoadDiagnostics diag;
    diag.filter_kind = filter_kind;
    diag.path = path;
    diag.fingerprint_bytes = sizeof(FingerprintType);

    FILE* file = open_binary_file_utf8_for_read(path, diag);
    if (file == nullptr) {
        diag.failure = XorLoadFailureKind::open_failed;
        log_xor_load_failure(diag);
        return false;
    }

    size_t size = 0;
    size_t arrayLength = 0;
    size_t segmentCount = 0;
    size_t segmentCountLength = 0;
    size_t segmentLength = 0;
    size_t segmentLengthMask = 0;

    const bool header_ok =
        std::fread(&size, sizeof(size), 1, file) == 1 &&
        std::fread(&arrayLength, sizeof(arrayLength), 1, file) == 1 &&
        std::fread(&segmentCount, sizeof(segmentCount), 1, file) == 1 &&
        std::fread(&segmentCountLength, sizeof(segmentCountLength), 1, file) == 1 &&
        std::fread(&segmentLength, sizeof(segmentLength), 1, file) == 1 &&
        std::fread(&segmentLengthMask, sizeof(segmentLengthMask), 1, file) == 1;

    diag.size = size;
    diag.arrayLength = arrayLength;
    diag.segmentCount = segmentCount;
    diag.segmentCountLength = segmentCountLength;
    diag.segmentLength = segmentLength;
    diag.segmentLengthMask = segmentLengthMask;

    if (!header_ok) {
        diag.failure = XorLoadFailureKind::header_read_failed;
        std::fclose(file);
        log_xor_load_failure(diag);
        return false;
    }

    if (arrayLength == 0 ||
        arrayLength > XOR_MAX_SIZE ||
        arrayLength > (std::numeric_limits<size_t>::max() / sizeof(FingerprintType))) {
        diag.failure = XorLoadFailureKind::invalid_header;
        std::fclose(file);
        log_xor_load_failure(diag);
        return false;
    }

    diag.expected_body_bytes = arrayLength * sizeof(FingerprintType);

    try {
        entry.fp.assign(arrayLength, FingerprintType{});
    }
    catch (const std::bad_alloc&) {
        diag.failure = XorLoadFailureKind::alloc_failed;
        std::fclose(file);
        log_xor_load_failure(diag);
        return false;
    }

    if (arrayLength > 0 &&
        std::fread(entry.fp.data(), sizeof(FingerprintType), arrayLength, file) != arrayLength) {
        diag.failure = XorLoadFailureKind::body_read_failed;
        entry.fp.clear();
        std::fclose(file);
        log_xor_load_failure(diag);
        return false;
    }

    std::fclose(file);

    entry.path = path;
    entry.size = size;
    entry.arrayLength = arrayLength;
    entry.segmentCount = segmentCount;
    entry.segmentCountLength = segmentCountLength;
    entry.segmentLength = segmentLength;
    entry.segmentLengthMask = segmentLengthMask;

    std::fprintf(stderr,
        "[!] Initializing %s XOR Filter [size=%zu] [host=%zu] [device=%zu] [fp=%zuB] [file=%s]\n",
        filter_kind,
        diag.expected_body_bytes,
        diag.expected_body_bytes,
        diag.expected_body_bytes,
        sizeof(FingerprintType),
        path.c_str());
    return true;
}

template <typename FingerprintType, typename CacheEntry, typename SymbolWriter>
static cudaError_t upload_xor_cache_to_gpu(
    const std::vector<CacheEntry>& cache,
    const char* filter_kind,
    SymbolWriter&& write_symbol) {

    for (size_t i = 0; i < cache.size(); ++i) {
        const size_t device_bytes = sizeof(FingerprintType) * cache[i].arrayLength;
        FingerprintType* devPtr = nullptr;
        cudaError_t err = cudaMalloc(&devPtr, device_bytes);
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                "[!] %s XOR filter GPU upload failed [file=%s] [stage=cudaMalloc] [device=%zu] [%s]\n",
                filter_kind, cache[i].path.c_str(), device_bytes, cudaGetErrorString(err));
            return err;
        }
        err = cudaMemcpy(devPtr, cache[i].fp.data(), device_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                "[!] %s XOR filter GPU upload failed [file=%s] [stage=cudaMemcpy] [device=%zu] [%s]\n",
                filter_kind, cache[i].path.c_str(), device_bytes, cudaGetErrorString(err));
            cudaFree(devPtr);
            return err;
        }
        err = write_symbol(devPtr, static_cast<int>(i), cache[i]);
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                "[!] %s XOR filter GPU upload failed [file=%s] [stage=cudaMemcpyToSymbol] [device=%zu] [%s]\n",
                filter_kind, cache[i].path.c_str(), device_bytes, cudaGetErrorString(err));
            cudaFree(devPtr);
            return err;
        }
    }
    return cudaSuccess;
}

static cudaError_t setTargetXorFilter(std::vector<std::string>& targets) {
    static std::mutex s_mutex;
    static std::vector<XorFilterCache32> s_cache;
    static bool s_loaded = false;

    {
        std::lock_guard<std::mutex> lock(s_mutex);
        if (!s_loaded) {
            s_cache.clear();
            for (size_t i = 0; i < targets.size(); ++i) {
                XorFilterCache32 entry;
                if (!load_xor_filter_cache_entry<uint32_t>(targets[i], "Compressed", entry)) {
                    s_cache.clear();
                    return cudaErrorUnknown;
                }
                s_cache.push_back(std::move(entry));
            }
            s_loaded = true;
        }
    }

    return upload_xor_cache_to_gpu<uint32_t>(s_cache, "Compressed",
        [](uint32_t* devPtr, int index, const XorFilterCache32& entry) {
            return cudaMemcpyToSymbol_XOR(devPtr, index, entry.size, entry.arrayLength, entry.segmentCount, entry.segmentCountLength, entry.segmentLength, entry.segmentLengthMask);
        });
}

static cudaError_t setTargetXorUnFilter(std::vector<std::string>& targets) {
    static std::mutex s_mutex;
    static std::vector<XorFilterCache32> s_cache;
    static bool s_loaded = false;

    {
        std::lock_guard<std::mutex> lock(s_mutex);
        if (!s_loaded) {
            s_cache.clear();
            for (size_t i = 0; i < targets.size(); ++i) {
                XorFilterCache32 entry;
                if (!load_xor_filter_cache_entry<uint32_t>(targets[i], "Uncompressed", entry)) {
                    s_cache.clear();
                    return cudaErrorUnknown;
                }
                s_cache.push_back(std::move(entry));
            }
            s_loaded = true;
        }
    }

    return upload_xor_cache_to_gpu<uint32_t>(s_cache, "Uncompressed",
        [](uint32_t* devPtr, int index, const XorFilterCache32& entry) {
            return cudaMemcpyToSymbol_XORUn(devPtr, index, entry.size, entry.arrayLength, entry.segmentCount, entry.segmentCountLength, entry.segmentLength, entry.segmentLengthMask);
        });
}

static cudaError_t setTargetXorUcFilter(std::vector<std::string>& targets) {
    static std::mutex s_mutex;
    static std::vector<XorFilterCache16> s_cache;
    static bool s_loaded = false;

    {
        std::lock_guard<std::mutex> lock(s_mutex);
        if (!s_loaded) {
            s_cache.clear();
            for (size_t i = 0; i < targets.size(); ++i) {
                XorFilterCache16 entry;
                if (!load_xor_filter_cache_entry<uint16_t>(targets[i], "Ultra compressed", entry)) {
                    s_cache.clear();
                    return cudaErrorUnknown;
                }
                s_cache.push_back(std::move(entry));
            }
            s_loaded = true;
        }
    }

    return upload_xor_cache_to_gpu<uint16_t>(s_cache, "Ultra compressed",
        [](uint16_t* devPtr, int index, const XorFilterCache16& entry) {
            return cudaMemcpyToSymbol_XORUc(devPtr, index, entry.size, entry.arrayLength, entry.segmentCount, entry.segmentCountLength, entry.segmentLength, entry.segmentLengthMask);
        });
}

static cudaError_t setTargetXorHcFilter(std::vector<std::string>& targets) {
    static std::mutex s_mutex;
    static std::vector<XorFilterCache8> s_cache;
    static bool s_loaded = false;

    {
        std::lock_guard<std::mutex> lock(s_mutex);
        if (!s_loaded) {
            s_cache.clear();
            for (size_t i = 0; i < targets.size(); ++i) {
                XorFilterCache8 entry;
                if (!load_xor_filter_cache_entry<uint8_t>(targets[i], "Hyper compressed", entry)) {
                    s_cache.clear();
                    return cudaErrorUnknown;
                }
                s_cache.push_back(std::move(entry));
            }
            s_loaded = true;
        }
    }

    return upload_xor_cache_to_gpu<uint8_t>(s_cache, "Hyper compressed",
        [](uint8_t* devPtr, int index, const XorFilterCache8& entry) {
            return cudaMemcpyToSymbol_XORHc(devPtr, index, entry.size, entry.arrayLength, entry.segmentCount, entry.segmentCountLength, entry.segmentLength, entry.segmentLengthMask);
        });
}

struct FoundBuffers {
    char (*strings)[512] = nullptr;
    unsigned char (*privKeys)[64] = nullptr;
    uint32_t (*hashes)[20] = nullptr;
    uint32_t (*lens)[1] = nullptr;
    uint32_t (*iters)[1] = nullptr;
    uint8_t* types = nullptr;
    int64_t* rounds = nullptr;
};

static thread_local FoundBuffers gFoundBuffers;

static void freeFoundBuffers()
{
    cudaFree(gFoundBuffers.strings);
    cudaFree(gFoundBuffers.privKeys);
    cudaFree(gFoundBuffers.hashes);
    cudaFree(gFoundBuffers.lens);
    cudaFree(gFoundBuffers.iters);
    cudaFree(gFoundBuffers.types);
    cudaFree(gFoundBuffers.rounds);
    gFoundBuffers = FoundBuffers{};
}

static cudaError_t allocateFoundBuffers()
{
    freeFoundBuffers();
    size_t free0 = 0;
    size_t total0 = 0;
    size_t free1 = 0;
    size_t total1 = 0;
    cudaMemGetInfo(&free0, &total0);
    const size_t count = MAX_FOUNDS;
    cudaError_t err = cudaMalloc((void**)&gFoundBuffers.strings, count * sizeof(*gFoundBuffers.strings));
    if (err != cudaSuccess) return err;
    err = cudaMalloc((void**)&gFoundBuffers.privKeys, count * sizeof(*gFoundBuffers.privKeys));
    if (err != cudaSuccess) return err;
    err = cudaMalloc((void**)&gFoundBuffers.hashes, count * sizeof(*gFoundBuffers.hashes));
    if (err != cudaSuccess) return err;
    err = cudaMalloc((void**)&gFoundBuffers.lens, count * sizeof(*gFoundBuffers.lens));
    if (err != cudaSuccess) return err;
    err = cudaMalloc((void**)&gFoundBuffers.iters, count * sizeof(*gFoundBuffers.iters));
    if (err != cudaSuccess) return err;
    err = cudaMalloc((void**)&gFoundBuffers.types, count * sizeof(uint8_t));
    if (err != cudaSuccess) return err;
    err = cudaMalloc((void**)&gFoundBuffers.rounds, count * sizeof(int64_t));
    if (err != cudaSuccess) return err;

    cudaMemset(gFoundBuffers.strings, 0, count * sizeof(*gFoundBuffers.strings));
    cudaMemset(gFoundBuffers.privKeys, 0, count * sizeof(*gFoundBuffers.privKeys));
    cudaMemset(gFoundBuffers.hashes, 0, count * sizeof(*gFoundBuffers.hashes));
    cudaMemset(gFoundBuffers.lens, 0, count * sizeof(*gFoundBuffers.lens));
    cudaMemset(gFoundBuffers.iters, 0, count * sizeof(*gFoundBuffers.iters));
    cudaMemset(gFoundBuffers.types, 0, count * sizeof(uint8_t));
    cudaMemset(gFoundBuffers.rounds, 0, count * sizeof(int64_t));

    err = cudaMemcpyToSymbol(d_foundStrings, &gFoundBuffers.strings, sizeof(gFoundBuffers.strings));
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(d_foundPrvKeys, &gFoundBuffers.privKeys, sizeof(gFoundBuffers.privKeys));
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(d_foundHash160, &gFoundBuffers.hashes, sizeof(gFoundBuffers.hashes));
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(d_len, &gFoundBuffers.lens, sizeof(gFoundBuffers.lens));
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(d_iter, &gFoundBuffers.iters, sizeof(gFoundBuffers.iters));
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(d_type, &gFoundBuffers.types, sizeof(gFoundBuffers.types));
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(d_round, &gFoundBuffers.rounds, sizeof(gFoundBuffers.rounds));
    if (err != cudaSuccess) return err;

    setFoundSize<<<1, 1>>>(MAX_FOUNDS);
    err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        cudaMemGetInfo(&free1, &total1);
        printf("[!] GPU %d founds memory allocated: %.2f MiB [!]\n", DEVICE_NR, (free0 - free1) / 1024.0 / 1024.0);
    }
    return err;
}

static cudaError_t prepareCuda()
{
    cudaError_t cudaStatus = cudaSuccess;
    size_t free_gpu_start = 0;
    size_t total_gpu_start = 0;
    size_t free_gpu_end = 0;
    size_t total_gpu_end = 0;
    cudaMemGetInfo(&free_gpu_start, &total_gpu_start);

    bool gpu_bloom_loaded = false;
    bool gpu_xor_loaded = false;
    bool gpu_xor_un_loaded = false;
    bool gpu_xor_uc_loaded = false;
    bool gpu_xor_hc_loaded = false;
    static std::mutex s_cpu_filter_load_mutex;
    static bool s_cpu_filters_loaded = false;

    const bool secp = Compressed || Uncompressed || Segwit || Taproot ||
        Ethereum || Xpoint || Dot || Aptos || Sui ||
        Xrp || Iota || Icp || Fil || Xtz;
    const bool ed = Solana || Ton || Ton_all || Dot || Aptos ||
        Sui || Xrp || Iota || Icp || Xtz;

    cudaStatus = allocateFoundBuffers();
    if (cudaStatus != cudaSuccess) return cudaStatus;

    printf("[!] GPU %d: preparing cuda device... Please wait... \n", DEVICE_NR);
#ifdef ECMULT_BIG_TABLE
    unsigned int bits = PARAM_ECMULT_WINDOW_SIZE;
    unsigned int windows = (256 / bits) + 1;
    size_t windowSize = (size_t(1) << (bits - 1));
    secp256k1_gej* devGejTemp = nullptr;
    secp256k1_fe* devZRatio = nullptr;

    if (secp) {
        cudaStatus = cudaMallocPitch(&_dev_precomp, &pitch, sizeof(secp256k1_ge_storage) * windowSize, windows);
        if (cudaStatus != cudaSuccess) return cudaStatus;
        cudaStatus = cudaMalloc((void**)&devGejTemp, windowSize * sizeof(secp256k1_gej));
        if (cudaStatus != cudaSuccess) return cudaStatus;
        cudaStatus = cudaMalloc((void**)&devZRatio, windowSize * sizeof(secp256k1_fe));
        if (cudaStatus != cudaSuccess) return cudaStatus;
        cudaStatus = loadWindow(bits, windows);
        if (cudaStatus != cudaSuccess) return cudaStatus;
        ecmult_big_create<<<1, 1>>>(devGejTemp, devZRatio, _dev_precomp, pitch, bits);
        cudaStatus = cudaGetLastError();
        if (cudaStatus == cudaSuccess) {
            cudaStatus = cudaDeviceSynchronize();
        }
        cudaFree(devGejTemp);
        cudaFree(devZRatio);
        if (cudaStatus != cudaSuccess) return cudaStatus;
        printf("[!] GPU %d: preparing finished \n", DEVICE_NR);
    }
#else
    unsigned int bits = ECMULT_GEN_PREC_BITS;
    if (secp && bits != 8 && bits != 4 && bits != 2) {
        int g = ECMULT_GEN_PREC_G;
        int n = ECMULT_GEN_PREC_N;
        secp256k1_ge_storage* devTable = nullptr;
        secp256k1_ge* devPrec = nullptr;
        secp256k1_gej* devPrecj = nullptr;
        cudaStatus = cudaMalloc((void**)&devPrecj, g * n * sizeof(secp256k1_gej));
        if (cudaStatus != cudaSuccess) return cudaStatus;
        cudaStatus = cudaMalloc((void**)&devPrec, g * n * sizeof(secp256k1_ge));
        if (cudaStatus != cudaSuccess) return cudaStatus;
        cudaStatus = cudaMalloc((void**)&devTable, g * n * sizeof(secp256k1_ge_storage));
        if (cudaStatus != cudaSuccess) return cudaStatus;
        computeTable<<<1, 1>>>(devTable, devPrec, devPrecj);
        cudaStatus = cudaDeviceSynchronize();
        cudaFree(devPrecj);
        cudaFree(devPrec);
        cudaFree(devTable);
        if (cudaStatus != cudaSuccess) return cudaStatus;
        printf("[!] GPU %d: prec gen %d bit finished [!]\n", DEVICE_NR, bits);
    }
#endif

    if (IS_HEX) {
        setHEX<<<1, 1>>>();
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) return cudaStatus;
        printf("[!] using -hex --- input strings are interpreted as hex-encoded bytes [!]\n");
    }
    if (seqMode) {
        SetSeqStep<<<1, 1>>>(step);
    }

    if (gUseBloom) {
        printf("[!] loading [%lld] bloom filter(s):\n", static_cast<long long>(gBloomFiles.size()));
        cudaStatus = _targetLookup.setTargets(gBloomFiles);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\n[!] Bloom filter(s) loading failed on GPU %d, skipping bloom on this GPU: %s [!]\n", DEVICE_NR, cudaGetErrorString(cudaStatus));
        }
        else {
            gpu_bloom_loaded = true;
        }
    }
    if (gUseXorC) {
        printf("[!] loading [%lld] compressed xor filter(s):\n", static_cast<long long>(gXorCFiles.size()));
        cudaStatus = setTargetXorFilter(gXorCFiles);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\n[!] Compressed xor filters loading failed on GPU %d, skipping on this GPU: %s [!]\n", DEVICE_NR, cudaGetErrorString(cudaStatus));
        }
        else {
            gpu_xor_loaded = true;
        }
    }
    if (gUseXorU) {
        printf("[!] loading [%lld] uncompressed xor filter(s):\n", static_cast<long long>(gXorUFiles.size()));
        cudaStatus = setTargetXorUnFilter(gXorUFiles);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\n[!] Uncompressed xor filters loading failed on GPU %d, skipping on this GPU: %s [!]\n", DEVICE_NR, cudaGetErrorString(cudaStatus));
        }
        else {
            gpu_xor_un_loaded = true;
        }
    }
    if (gUseXorUc) {
        printf("[!] loading [%lld] ultra compressed xor filter(s):\n", static_cast<long long>(gXorUcFiles.size()));
        cudaStatus = setTargetXorUcFilter(gXorUcFiles);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\n[!] Ultra compressed xor filters loading failed on GPU %d, skipping on this GPU: %s [!]\n", DEVICE_NR, cudaGetErrorString(cudaStatus));
        }
        else {
            gpu_xor_uc_loaded = true;
        }
    }
    if (gUseXorH) {
        printf("[!] loading [%lld] hyper compressed xor filter(s):\n", static_cast<long long>(gXorHFiles.size()));
        cudaStatus = setTargetXorHcFilter(gXorHFiles);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\n[!] Hyper compressed xor filters loading failed on GPU %d, skipping on this GPU: %s [!]\n", DEVICE_NR, cudaGetErrorString(cudaStatus));
        }
        else {
            gpu_xor_hc_loaded = true;
        }
    }
    if (useBloomCPU || useXorCPU) {
        bool need_cpu_load = false;
        {
            std::lock_guard<std::mutex> lock(s_cpu_filter_load_mutex);
            if (!s_cpu_filters_loaded) {
                s_cpu_filters_loaded = true;
                need_cpu_load = true;
            }
        }

        if (need_cpu_load) {
            if (useBloomCPU) {
                printf("[!] loading [%lld] bloom filter(s) in CPU RAM:\n", static_cast<long long>(gCpuBloomFiles.size()));
                if (!loadBloomFiltersIntoSharedMemory(gCpuBloomFiles)) {
                    fprintf(stderr, "\n[!] Bloom filter(s) CPU loading failed [!]\n");
                    std::lock_guard<std::mutex> lock(s_cpu_filter_load_mutex);
                    s_cpu_filters_loaded = false;
                    return cudaErrorInvalidValue;
                }
            }
            if (useXorCPU) {
                printf("[!] loading [%lld] uncompressed xor filter(s) in CPU RAM:\n", static_cast<long long>(gCpuXorFiles.size()));
                if (!loadXorFilters(gCpuXorFiles)) {
                    fprintf(stderr, "\n[!] XOR's CPU loading failed [!]\n");
                    std::lock_guard<std::mutex> lock(s_cpu_filter_load_mutex);
                    s_cpu_filters_loaded = false;
                    return cudaErrorInvalidValue;
                }
            }
        }
    }

    cudaStatus = loadHashTarget(hashTargetWordsHost, hashTargetMasksHost, hashTargetLenBytes, useHashTarget);
    if (cudaStatus != cudaSuccess) return cudaStatus;
    if (useHashTarget) {
        printf("[!] GPU %d: loaded hash prefix matcher (%u byte(s), hex=%s)\n",
            DEVICE_NR, hashTargetLenBytes, hashTargetHex.c_str());
    }

    SetCurve<<<1, 1>>>(secp, ed, Compressed, Uncompressed, Segwit, Taproot,
        Ethereum, Xpoint, Solana, Ton, Ton_all,
        Dot, Aptos, Sui, Xrp, Iota,
        Icp, Fil, Xtz, Endomorphism);
    setFilterType<<<1, 1>>>(gpu_bloom_loaded, gpu_xor_loaded, gpu_xor_un_loaded, gpu_xor_uc_loaded, gpu_xor_hc_loaded);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) return cudaStatus;

    printf("[!] GPU %d: all preparation finished.\n", DEVICE_NR);
    cudaMemGetInfo(&free_gpu_end, &total_gpu_end);
    printf("[!] GPU %d total memory used: %.2f MiB [!]\n", DEVICE_NR, (free_gpu_start - free_gpu_end) / 1024.0 / 1024.0);
    printf("-----------------------------------------------------------------\n");
    printf("[!] Loaded bloom filters on GPU %d: %lld\n", DEVICE_NR, gpu_bloom_loaded ? static_cast<long long>(gBloomFiles.size()) : 0ll);
    printf("[!] Loaded uncompressed xor filters on GPU %d: %lld\n", DEVICE_NR, gpu_xor_un_loaded ? static_cast<long long>(gXorUFiles.size()) : 0ll);
    printf("[!] Loaded compressed xor filters on GPU %d: %lld\n", DEVICE_NR, gpu_xor_loaded ? static_cast<long long>(gXorCFiles.size()) : 0ll);
    printf("[!] Loaded ultra compressed xor filters on GPU %d: %lld\n", DEVICE_NR, gpu_xor_uc_loaded ? static_cast<long long>(gXorUcFiles.size()) : 0ll);
    printf("[!] Loaded hyper compressed xor filters on GPU %d: %lld\n", DEVICE_NR, gpu_xor_hc_loaded ? static_cast<long long>(gXorHFiles.size()) : 0ll);
    printf("[!] Loaded hash prefix matcher on GPU %d: %s\n", DEVICE_NR, useHashTarget ? "yes" : "no");
    return cudaSuccess;
}

static cudaError_t launchWorkerPrivSeq(dim3 grid, dim3 block, bool* isResult, bool* deviceResult, int walkMode, uint8_t* devStart, uint64_t seqStep, uint64_t* startxBuf, uint64_t* startyBuf)
{
    const bool hasOther = Solana || Ton || Ton_all || Dot || Aptos || Sui || Xrp || Iota || Icp || Fil || Xtz;
    if (Endomorphism && !hasOther && (Compressed || Uncompressed || Segwit || Taproot || Ethereum || Xpoint)) {
        workerPRIV_seq_vanity_cusr<<<grid, block>>>(isResult, deviceResult, _dev_precomp, pitch, walkMode, devStart, seqStep, PRIV_SEQ_THREAD_STEPS, startxBuf, startyBuf);
    }
    else if (!Endomorphism && !hasOther && !Ethereum && !Compressed && !Uncompressed && !Segwit && !Taproot && Xpoint) {
        workerPRIV_seq_vanity_x<<<grid, block>>>(isResult, deviceResult, _dev_precomp, pitch, walkMode, devStart, seqStep, PRIV_SEQ_THREAD_STEPS, startxBuf, startyBuf);
    }
    else if (!Endomorphism && !hasOther && !Ethereum && Compressed && !Uncompressed && !Segwit && !Taproot && !Xpoint) {
        workerPRIV_seq_vanity_c<<<grid, block>>>(isResult, deviceResult, _dev_precomp, pitch, walkMode, devStart, seqStep, PRIV_SEQ_THREAD_STEPS, startxBuf, startyBuf);
    }
    else if (!Endomorphism && !hasOther && !Ethereum && !Compressed && Uncompressed && !Segwit && !Taproot && !Xpoint) {
        workerPRIV_seq_vanity_u<<<grid, block>>>(isResult, deviceResult, _dev_precomp, pitch, walkMode, devStart, seqStep, PRIV_SEQ_THREAD_STEPS, startxBuf, startyBuf);
    }
    else if (!Endomorphism && !hasOther && !Ethereum && !Compressed && !Uncompressed && Segwit && !Taproot && !Xpoint) {
        workerPRIV_seq_vanity_s<<<grid, block>>>(isResult, deviceResult, _dev_precomp, pitch, walkMode, devStart, seqStep, PRIV_SEQ_THREAD_STEPS, startxBuf, startyBuf);
    }
    else if (!Endomorphism && !hasOther && !Ethereum && !Compressed && !Uncompressed && !Segwit && Taproot && !Xpoint) {
        workerPRIV_seq_vanity_r<<<grid, block>>>(isResult, deviceResult, _dev_precomp, pitch, walkMode, devStart, seqStep, PRIV_SEQ_THREAD_STEPS, startxBuf, startyBuf);
    }
    else if (!Endomorphism && !hasOther && Ethereum && !Compressed && !Uncompressed && !Segwit && !Taproot && !Xpoint) {
        workerPRIV_seq_vanity_e<<<grid, block>>>(isResult, deviceResult, _dev_precomp, pitch, walkMode, devStart, seqStep, PRIV_SEQ_THREAD_STEPS, startxBuf, startyBuf);
    }
    else if (!hasOther && !Ethereum && !Xpoint && (Compressed || Uncompressed || Segwit || Taproot)) {
        workerPRIV_seq_vanity_cusr<<<grid, block>>>(isResult, deviceResult, _dev_precomp, pitch, walkMode, devStart, seqStep, PRIV_SEQ_THREAD_STEPS, startxBuf, startyBuf);
    }
    else {
        workerPRIV_seq_128<<<grid, block>>>(isResult, deviceResult, _dev_precomp, pitch, walkMode, devStart, seqStep, PRIV_SEQ_THREAD_STEPS, startxBuf, startyBuf);
    }
    return cudaGetLastError();
}

static inline cudaError_t allocateResultBuffers(const uint64_t outputEntries, bool** outIsResult, bool** outDeviceResult)
{
    if (outIsResult == nullptr || outDeviceResult == nullptr) {
        return cudaErrorInvalidValue;
    }
    *outIsResult = nullptr;
    *outDeviceResult = nullptr;

    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(outIsResult), sizeof(bool));
    if (err != cudaSuccess || *outIsResult == nullptr) {
        return (err != cudaSuccess) ? err : cudaErrorMemoryAllocation;
    }
    err = cudaMemset(*outIsResult, 0, sizeof(bool));
    if (err != cudaSuccess) {
        cudaFree(*outIsResult);
        *outIsResult = nullptr;
        return err;
    }

    if (outputEntries == 0) {
        return cudaSuccess;
    }
    if (outputEntries > (numeric_limits<size_t>::max() / sizeof(bool))) {
        cudaFree(*outIsResult);
        *outIsResult = nullptr;
        return cudaErrorInvalidValue;
    }

    const size_t bytes = static_cast<size_t>(outputEntries) * sizeof(bool);
    err = cudaMalloc(reinterpret_cast<void**>(outDeviceResult), bytes);
    if (err != cudaSuccess || *outDeviceResult == nullptr) {
        cudaFree(*outIsResult);
        *outIsResult = nullptr;
        return (err != cudaSuccess) ? err : cudaErrorMemoryAllocation;
    }

    err = cudaMemset(*outDeviceResult, 0, bytes);
    if (err != cudaSuccess) {
        cudaFree(*outDeviceResult);
        cudaFree(*outIsResult);
        *outDeviceResult = nullptr;
        *outIsResult = nullptr;
    }
    return err;
}

static std::mutex g_result_flag_host_access_mutex;

static inline bool read_result_flag_host(bool* flag_ptr)
{
    if (flag_ptr == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_result_flag_host_access_mutex);
    bool host_value = false;
    const cudaError_t st = cudaMemcpy(&host_value, flag_ptr, sizeof(bool), cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) {
        return false;
    }
    return host_value;
}

static inline void clear_result_flag_host(bool* flag_ptr)
{
    if (flag_ptr == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_result_flag_host_access_mutex);
    cudaMemset(flag_ptr, 0, sizeof(bool));
}

class GpuRuntimeContext {
public:
    GpuRuntimeContext(int deviceOrdinal, int deviceCount, FILE* outputFile)
        : deviceOrdinal_(deviceOrdinal), deviceCount_(deviceCount), outputFile_(outputFile)
    {
        const uint64_t inputWorkSize = static_cast<uint64_t>(BLOCK_NUMBER) * BLOCK_THREADS *
            static_cast<uint64_t>(runKind == RunKind::Priv ? ((seqMode || isRandom) ? PRIV_SEQ_THREAD_STEPS : THREAD_STEPS) : THREAD_STEPS_BRAIN);
        const bool generatedPrivSeqPath = runKind == RunKind::Priv && (seqMode || isRandom);
        const uint64_t resultEntries = generatedPrivSeqPath ? 1ull : max<uint64_t>(workSize, inputWorkSize);
        cudaError_t err = allocateResultBuffers(resultEntries, &isResult_, &deviceResult_);
        if (err != cudaSuccess) {
            throw runtime_error(cudaGetErrorString(err));
        }
        if (generatedPrivSeqPath) {
            err = cudaMalloc((void**)&devPrivSeqStart_, 32);
            if (err != cudaSuccess || devPrivSeqStart_ == nullptr) {
                throw runtime_error(cudaGetErrorString(err));
            }
            useVanityStartBuffers_ = (PRIV_SEQ_THREAD_STEPS % 1024 == 0) && (BLOCK_THREADS <= 256u);
            if (useVanityStartBuffers_) {
                const size_t startBufSize = static_cast<size_t>(BLOCK_NUMBER) * BLOCK_THREADS * 4 * sizeof(uint64_t);
                err = cudaMalloc((void**)&devVanityStartX_, startBufSize);
                if (err == cudaSuccess) {
                    err = cudaMalloc((void**)&devVanityStartY_, startBufSize);
                }
                if (err != cudaSuccess) {
                    fprintf(stderr, "[!] GPU %d: vanity start buffers disabled: %s [!]\n", DEVICE_NR, cudaGetErrorString(err));
                    cudaFree(devVanityStartX_);
                    cudaFree(devVanityStartY_);
                    devVanityStartX_ = nullptr;
                    devVanityStartY_ = nullptr;
                    useVanityStartBuffers_ = false;
                }
            }
        }
        if (runKind == RunKind::Brain) {
            iterationsHost_.reserve(Iterations.size());
            for (int iter : Iterations) {
                iterationsHost_.push_back(static_cast<uint32_t>(iter));
            }
            err = cudaMalloc((void**)&devIterations_, max<size_t>(1, iterationsHost_.size() * sizeof(uint32_t)));
            if (err != cudaSuccess) {
                throw runtime_error(cudaGetErrorString(err));
            }
            err = cudaMemcpyAsync(devIterations_, iterationsHost_.data(),
                iterationsHost_.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                throw runtime_error(cudaGetErrorString(err));
            }
        }
    }

    ~GpuRuntimeContext()
    {
        wait_for_current_gpu_async_save_queue();
        for (int i = 0; i < 2; ++i) {
            cudaFree(devLines_[i]);
            cudaFree(devIndexes_[i]);
        }
        cudaFree(devIterations_);
        cudaFree(devPrivSeqStart_);
        cudaFree(devVanityStartX_);
        cudaFree(devVanityStartY_);
        if (isResult_ != nullptr) cudaFree(isResult_);
        if (deviceResult_ != nullptr) cudaFree(deviceResult_);
    }

    uint64_t processed() const
    {
        return processed_;
    }

    uint32_t found() const
    {
        return found_;
    }

    int deviceOrdinal() const
    {
        return deviceOrdinal_;
    }

    int deviceCount() const
    {
        return deviceCount_;
    }

    void uploadBatch(int slot, const std::string& combined, const std::vector<uint32_t>& indexes)
    {
        if (indexes.empty()) {
            return;
        }
        cudaError_t err = copy_to_device_grow((void**)&devLines_[slot], combined.data(), combined.size());
        if (err != cudaSuccess) {
            throw runtime_error(cudaGetErrorString(err));
        }
        err = copy_to_device_grow((void**)&devIndexes_[slot], indexes.data(), indexes.size() * sizeof(uint32_t));
        if (err != cudaSuccess) {
            throw runtime_error(cudaGetErrorString(err));
        }
    }

    void launchBatch(int slot, uint32_t count)
    {
        if (count == 0) {
            return;
        }
        if (runKind == RunKind::Priv) {
            workerPRIV<<<BLOCK_NUMBER, BLOCK_THREADS>>>(isResult_, deviceResult_, devLines_[slot], devIndexes_[slot],
                count, _dev_precomp, pitch, 0);
        }
        else {
            workerBrain<<<BLOCK_NUMBER, BLOCK_THREADS>>>(isResult_, deviceResult_, devLines_[slot], devIndexes_[slot],
                count, _dev_precomp, pitch, 0, static_cast<uint8_t>(brainHash),
                devIterations_, static_cast<uint32_t>(iterationsHost_.size()));
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw runtime_error(cudaGetErrorString(err));
        }
    }

    void syncAndSave(bool priv)
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw runtime_error(cudaGetErrorString(err));
        }
        if (read_result_flag_host(isResult_)) {
            clear_result_flag_host(isResult_);
            if (priv) {
                SaveResultPRIV(outputFile_, found_, save, Derivations_list);
            }
            else {
                SaveResultBrain(outputFile_, found_, save, Derivations_list);
            }
        }
    }

    void accountBatch(uint32_t count)
    {
        processed_ += count;
        const uint64_t counterMul = (runKind == RunKind::Brain) ? static_cast<uint64_t>(max<size_t>(1u, Iterations.size())) : 1ull;
        counterTotal.fetch_add(static_cast<uint64_t>(count) * counterMul, std::memory_order_relaxed);
    }

    void runPrivSeqChunk(const array<uint8_t, 32>& start, bool back, uint64_t seqStep, uint64_t count)
    {
        if (devPrivSeqStart_ == nullptr) {
            throw runtime_error("private sequential start buffer is not allocated");
        }

        cudaError_t err = cudaMemcpyAsync(devPrivSeqStart_, start.data(), 32, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw runtime_error(cudaGetErrorString(err));
        }

        const int walkMode = back ? -1 : 1;
        const bool secpActive = Compressed || Uncompressed || Segwit || Taproot ||
            Ethereum || Xpoint || Dot || Aptos || Sui || Xrp || Iota || Icp || Fil || Xtz;
        const bool hasOther = Solana || Ton || Ton_all || Dot || Aptos || Sui || Xrp || Iota || Icp || Fil || Xtz;
        const bool persistentCompressedOnly = useVanityStartBuffers_ && devVanityStartX_ != nullptr && devVanityStartY_ != nullptr &&
            !hasOther && !Ethereum && Compressed && !Uncompressed && !Segwit && !Taproot && !Xpoint && !Endomorphism;
        const bool canReusePersistent = persistentCompressedOnly && vanityStartBuffersReady_ &&
            vanityPersistentShiftReady_ && vanitySeqStep_ == seqStep && vanityWalkMode_ == walkMode;

        if (secpActive) {
            if (!canReusePersistent) {
                compute_P0_H_kernel<<<1, 1>>>(devPrivSeqStart_, seqStep, _dev_precomp, pitch, walkMode);
                err = cudaGetLastError();
                if (err == cudaSuccess) {
                    err = cudaDeviceSynchronize();
                }
                if (err == cudaSuccess) {
                    err = cuda_vanity_set_step_increment(seqStep, _dev_precomp, pitch);
                }
                if (err != cudaSuccess) {
                    throw runtime_error(cudaGetErrorString(err));
                }
                vanityStartBuffersReady_ = false;
                vanityPersistentShiftReady_ = false;
                vanitySeqStep_ = seqStep;
                vanityWalkMode_ = walkMode;

                if (persistentCompressedOnly) {
                    const uint64_t outputSize = static_cast<uint64_t>(BLOCK_NUMBER) *
                        static_cast<uint64_t>(BLOCK_THREADS) *
                        static_cast<uint64_t>(PRIV_SEQ_THREAD_STEPS);
                    vanityPersistentShift_ = (outputSize > static_cast<uint64_t>(PRIV_SEQ_THREAD_STEPS))
                        ? (outputSize - static_cast<uint64_t>(PRIV_SEQ_THREAD_STEPS))
                        : 0ull;
                    err = cuda_vanity_set_persistent_shift(vanityPersistentShift_);
                    if (err != cudaSuccess) {
                        throw runtime_error(cudaGetErrorString(err));
                    }
                    vanityPersistentShiftReady_ = true;
                }
            }

            if (useVanityStartBuffers_ && devVanityStartX_ != nullptr && devVanityStartY_ != nullptr && !vanityStartBuffersReady_) {
                precompute_vanity_starts_kernel<<<dim3(BLOCK_NUMBER), dim3(BLOCK_THREADS)>>>(
                    devVanityStartX_, devVanityStartY_, PRIV_SEQ_THREAD_STEPS);
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    throw runtime_error(cudaGetErrorString(err));
                }
                vanityStartBuffersReady_ = true;
            }
        }

        uint64_t* startX = (useVanityStartBuffers_ && devVanityStartX_ != nullptr && devVanityStartY_ != nullptr) ? devVanityStartX_ : nullptr;
        uint64_t* startY = (startX != nullptr) ? devVanityStartY_ : nullptr;
        err = launchWorkerPrivSeq(dim3(BLOCK_NUMBER), dim3(BLOCK_THREADS), isResult_, deviceResult_, walkMode, devPrivSeqStart_, seqStep, startX, startY);
        if (err != cudaSuccess) {
            throw runtime_error(cudaGetErrorString(err));
        }
        finishKernel(true);

        if (persistentCompressedOnly && vanityStartBuffersReady_) {
            if (vanityPersistentShift_ != 0ull) {
                err = cuda_vanity_apply_persistent_shift(devVanityStartX_, devVanityStartY_, dim3(BLOCK_NUMBER), dim3(BLOCK_THREADS));
                if (err != cudaSuccess) {
                    throw runtime_error(cudaGetErrorString(err));
                }
            }
        }
        else {
            vanityStartBuffersReady_ = false;
        }

        processed_ += count;
        counterTotal.fetch_add(count * privEndomorphismKeyMultiplier(), std::memory_order_relaxed);
    }

    void runBrainSeqChunk(const array<uint8_t, 256>& start, bool back, uint64_t seqStep, uint64_t count)
    {
        uint8_t* devStart = nullptr;
        cudaError_t err = cudaMalloc((void**)&devStart, 256);
        if (err != cudaSuccess) {
            throw runtime_error(cudaGetErrorString(err));
        }
        err = cudaMemcpy(devStart, start.data(), 256, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(devStart);
            throw runtime_error(cudaGetErrorString(err));
        }
        SetSeqStep<<<1, 1>>>(seqStep);
        for (int iter : Iterations) {
            workerBrain_seq<<<BLOCK_NUMBER, BLOCK_THREADS>>>(isResult_, deviceResult_, _dev_precomp, pitch, 0,
                static_cast<uint8_t>(brainHash), back ? 2 : 1, devStart, 0, static_cast<uint32_t>(iter));
            finishKernel(false);
        }
        cudaFree(devStart);
        processed_ += count;
        counterTotal.fetch_add(count * static_cast<uint64_t>(max<size_t>(1u, Iterations.size())), std::memory_order_relaxed);
    }

    void runMask(const BrainMaskSpec& spec, uint64_t allowed)
    {
        if (allowed == 0 || spec.len == 0) {
            return;
        }
        BrainMaskSpec* devSpec = nullptr;
        cudaError_t err = cudaMalloc((void**)&devSpec, sizeof(BrainMaskSpec));
        if (err != cudaSuccess) {
            throw runtime_error(cudaGetErrorString(err));
        }
        err = cudaMemcpy(devSpec, &spec, sizeof(BrainMaskSpec), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(devSpec);
            throw runtime_error(cudaGetErrorString(err));
        }

        const uint64_t maxIndexable = numeric_limits<uint32_t>::max() / spec.len;
        if (maxIndexable == 0) {
            cudaFree(devSpec);
            throw runtime_error("mask is too long");
        }
        const uint64_t launchChunk = max<uint64_t>(1, min<uint64_t>(workSize, maxIndexable));
        const uint64_t advance = launchChunk * static_cast<uint64_t>(deviceCount_);
        uint64_t roundBase = 0;
        while (roundBase < allowed) {
            const uint64_t localStart = roundBase + static_cast<uint64_t>(deviceOrdinal_) * launchChunk;
            if (localStart < allowed) {
                const uint64_t count64 = min<uint64_t>(launchChunk, allowed - localStart);
                runMaskChunk(devSpec, spec, localStart, static_cast<uint32_t>(count64));
            }
            if (advance == 0 || roundBase > numeric_limits<uint64_t>::max() - advance) {
                break;
            }
            roundBase += advance;
        }
        cudaFree(devSpec);
    }

private:
    void runMaskChunk(BrainMaskSpec* devSpec, const BrainMaskSpec& spec, uint64_t start, uint32_t count)
    {
        if (count == 0) {
            return;
        }
        if (spec.len > numeric_limits<uint32_t>::max() / count) {
            throw runtime_error("mask batch is too large");
        }
        const size_t lineBytes = static_cast<size_t>(spec.len) * count;
        char* devLines = nullptr;
        uint32_t* devIndexes = nullptr;
        cudaError_t err = cudaMalloc((void**)&devLines, max<size_t>(lineBytes, 1));
        if (err != cudaSuccess) {
            throw runtime_error(cudaGetErrorString(err));
        }
        err = cudaMalloc((void**)&devIndexes, static_cast<size_t>(count) * sizeof(uint32_t));
        if (err != cudaSuccess) {
            cudaFree(devLines);
            throw runtime_error(cudaGetErrorString(err));
        }

        buildMaskBatch<<<BLOCK_NUMBER, BLOCK_THREADS>>>(devSpec, start, devLines, devIndexes, count);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(devLines);
            cudaFree(devIndexes);
            throw runtime_error(cudaGetErrorString(err));
        }

        if (runKind == RunKind::Priv) {
            workerPRIV<<<BLOCK_NUMBER, BLOCK_THREADS>>>(isResult_, deviceResult_, devLines, devIndexes,
                count, _dev_precomp, pitch, 0);
            finishKernel(true);
        }
        else {
            vector<uint32_t> iterations;
            iterations.reserve(Iterations.size());
            for (int iter : Iterations) {
                iterations.push_back(static_cast<uint32_t>(iter));
            }
            uint32_t* devIterations = nullptr;
            err = cudaMalloc((void**)&devIterations, iterations.size() * sizeof(uint32_t));
            if (err != cudaSuccess) {
                cudaFree(devLines);
                cudaFree(devIndexes);
                throw runtime_error(cudaGetErrorString(err));
            }
            err = cudaMemcpy(devIterations, iterations.data(), iterations.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                cudaFree(devIterations);
                cudaFree(devLines);
                cudaFree(devIndexes);
                throw runtime_error(cudaGetErrorString(err));
            }
            workerBrain<<<BLOCK_NUMBER, BLOCK_THREADS>>>(isResult_, deviceResult_, devLines, devIndexes,
                count, _dev_precomp, pitch, 0, static_cast<uint8_t>(brainHash),
                devIterations, static_cast<uint32_t>(iterations.size()));
            finishKernel(false);
            cudaFree(devIterations);
        }

        processed_ += count;
        const uint64_t counterMul = (runKind == RunKind::Brain) ? static_cast<uint64_t>(max<size_t>(1u, Iterations.size())) : 1ull;
        counterTotal.fetch_add(static_cast<uint64_t>(count) * counterMul, std::memory_order_relaxed);
        cudaFree(devLines);
        cudaFree(devIndexes);
    }

    void finishKernel(bool priv)
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw runtime_error(cudaGetErrorString(err));
        }
        if (read_result_flag_host(isResult_)) {
            clear_result_flag_host(isResult_);
            if (priv) {
                SaveResultPRIV(outputFile_, found_, save, Derivations_list);
            }
            else {
                SaveResultBrain(outputFile_, found_, save, Derivations_list);
            }
        }
    }

    int deviceOrdinal_ = 0;
    int deviceCount_ = 1;
    FILE* outputFile_ = nullptr;
    bool* isResult_ = nullptr;
    bool* deviceResult_ = nullptr;
    char* devLines_[2] = { nullptr, nullptr };
    uint32_t* devIndexes_[2] = { nullptr, nullptr };
    uint32_t* devIterations_ = nullptr;
    uint8_t* devPrivSeqStart_ = nullptr;
    uint64_t* devVanityStartX_ = nullptr;
    uint64_t* devVanityStartY_ = nullptr;
    bool useVanityStartBuffers_ = false;
    bool vanityStartBuffersReady_ = false;
    bool vanityPersistentShiftReady_ = false;
    uint64_t vanityPersistentShift_ = 0;
    uint64_t vanitySeqStep_ = 0;
    int vanityWalkMode_ = 0;
    vector<uint32_t> iterationsHost_;
    uint64_t processed_ = 0;
    uint32_t found_ = 0;
};

template <size_t N>
static bool lessOrEqual(const array<uint8_t, N>& a, const array<uint8_t, N>& b)
{
    return lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()) || a == b;
}

template <size_t N>
static bool greaterOrEqual(const array<uint8_t, N>& a, const array<uint8_t, N>& b)
{
    return lexicographical_compare(b.begin(), b.end(), a.begin(), a.end()) || a == b;
}

template <size_t N>
static array<uint8_t, N> bytesFromHexFixed(string text)
{
    if (text.empty()) {
        text = "0";
    }
    if (text.size() >= 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
        text.erase(0, 2);
    }
    if ((text.size() & 1) != 0) {
        text.insert(text.begin(), '0');
    }
    if (text.size() > N * 2 || !all_of(text.begin(), text.end(), isHexChar)) {
        throw runtime_error("invalid range hex value");
    }
    string bytes;
    string error;
    if (!decodeHex(text, bytes, error)) {
        throw runtime_error(error);
    }
    array<uint8_t, N> out{};
    size_t copyLen = min<size_t>(bytes.size(), N);
    memcpy(out.data() + (N - copyLen), bytes.data() + (bytes.size() - copyLen), copyLen);
    return out;
}

template <size_t N>
static void addBig(array<uint8_t, N>& value, uint64_t step)
{
    uint64_t carry = step;
    for (size_t i = N; i-- > 0 && carry != 0;) {
        uint64_t v = static_cast<uint64_t>(value[i]) + (carry & 0xffu);
        value[i] = static_cast<uint8_t>(v & 0xffu);
        carry = (carry >> 8) + (v >> 8);
    }
}

template <size_t N>
static void subBig(array<uint8_t, N>& value, uint64_t step)
{
    uint64_t borrow = step;
    for (size_t i = N; i-- > 0 && borrow != 0;) {
        uint64_t b = borrow & 0xffu;
        if (value[i] >= b) {
            value[i] = static_cast<uint8_t>(value[i] - b);
            borrow >>= 8;
        }
        else {
            value[i] = static_cast<uint8_t>(256u + value[i] - b);
            borrow = (borrow >> 8) + 1;
        }
    }
}

static bool mulOverflow64(uint64_t a, uint64_t b)
{
    return b != 0 && a > (numeric_limits<uint64_t>::max() / b);
}

class SeqRoundBarrier {
public:
    explicit SeqRoundBarrier(size_t total) : total_(total) {}

    bool wait()
    {
        unique_lock<mutex> lock(mutex_);
        if (aborted_) {
            return false;
        }
        const size_t generation = generation_;
        if (++arrived_ == total_) {
            arrived_ = 0;
            ++generation_;
            cv_.notify_all();
            return !aborted_;
        }
        cv_.wait(lock, [&]() { return aborted_ || generation != generation_; });
        return !aborted_;
    }

    void abort()
    {
        lock_guard<mutex> lock(mutex_);
        aborted_ = true;
        ++generation_;
        arrived_ = 0;
        cv_.notify_all();
    }

private:
    size_t total_ = 0;
    size_t arrived_ = 0;
    size_t generation_ = 0;
    bool aborted_ = false;
    mutex mutex_;
    condition_variable cv_;
};

static SeqRoundBarrier* g_seq_round_barrier = nullptr;

static void waitSeqRoundBarrier()
{
    if (g_seq_round_barrier != nullptr && !g_seq_round_barrier->wait()) {
        throw runtime_error("sequential multi-GPU dispatch aborted");
    }
}

struct SeqLogEntry {
    bool valid = false;
    uint64_t step = 0;
    bool both = false;
    bool back = false;
    string point;
    string plus;
    string minus;
};

static std::mutex g_seq_log_mutex;
static vector<SeqLogEntry> g_seq_log_entries;

template <size_t N>
static string bytesToHexUpper(const array<uint8_t, N>& value)
{
    static const char hex[] = "0123456789ABCDEF";
    string out;
    out.resize(N * 2);
    for (size_t i = 0; i < N; ++i) {
        out[i * 2] = hex[value[i] >> 4];
        out[i * 2 + 1] = hex[value[i] & 0x0f];
    }
    return out;
}

static string hexStep(uint64_t value)
{
    std::ostringstream ss;
    ss << "0x" << std::hex << std::uppercase << value;
    return ss.str();
}

static void writeSeqLogSnapshotLocked()
{
    std::ofstream ofs(log_file, std::ios::trunc);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open log file: " << log_file << "\n";
        return;
    }
    for (const SeqLogEntry& entry : g_seq_log_entries) {
        if (!entry.valid) {
            continue;
        }
        if (entry.both) {
            ofs << "+" << entry.plus << " step " << hexStep(entry.step) << "\n";
            ofs << "-" << entry.minus << " step " << hexStep(entry.step) << "\n";
        }
        else {
            ofs << (entry.back ? "-" : "+") << entry.point << " step " << hexStep(entry.step) << "\n";
        }
    }
}

static void writeSeqLogPointLocked(uint64_t seqStep, const string& point, bool back)
{
    std::ofstream ofs(log_file, std::ios::trunc);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open log file: " << log_file << "\n";
        return;
    }
    ofs << (back ? "-" : "+") << point << " step " << hexStep(seqStep) << "\n";
}

static void writeSeqLogBothLocked(uint64_t seqStep, const string& plus, const string& minus)
{
    std::ofstream ofs(log_file, std::ios::trunc);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open log file: " << log_file << "\n";
        return;
    }
    ofs << "+" << plus << " step " << hexStep(seqStep) << "\n";
    ofs << "-" << minus << " step " << hexStep(seqStep) << "\n";
}

template <size_t N>
static void updateSeqLogPoint(size_t ordinal, uint64_t seqStep, const array<uint8_t, N>& point, bool back)
{
    if (!isLog) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_seq_log_mutex);
    if (g_seq_log_entries.size() <= ordinal) {
        g_seq_log_entries.resize(ordinal + 1);
    }
    SeqLogEntry& entry = g_seq_log_entries[ordinal];
    entry.valid = true;
    entry.step = seqStep;
    entry.both = false;
    entry.back = back;
    entry.point = bytesToHexUpper(point);
    entry.plus.clear();
    entry.minus.clear();
    writeSeqLogSnapshotLocked();
}

template <size_t N>
static void updateSeqLogCheckpointPoint(uint64_t seqStep, const array<uint8_t, N>& point, bool back)
{
    if (!isLog) {
        return;
    }
    lock_guard<mutex> lock(g_seq_log_mutex);
    writeSeqLogPointLocked(seqStep, bytesToHexUpper(point), back);
}

template <size_t N>
static void updateSeqLogBoth(size_t ordinal, uint64_t seqStep, const array<uint8_t, N>& plus, const array<uint8_t, N>& minus)
{
    if (!isLog) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_seq_log_mutex);
    if (g_seq_log_entries.size() <= ordinal) {
        g_seq_log_entries.resize(ordinal + 1);
    }
    SeqLogEntry& entry = g_seq_log_entries[ordinal];
    entry.valid = true;
    entry.step = seqStep;
    entry.both = true;
    entry.back = false;
    entry.point.clear();
    entry.plus = bytesToHexUpper(plus);
    entry.minus = bytesToHexUpper(minus);
    writeSeqLogSnapshotLocked();
}

template <size_t N>
static void updateSeqLogCheckpointBoth(uint64_t seqStep, const array<uint8_t, N>& plus, const array<uint8_t, N>& minus)
{
    if (!isLog) {
        return;
    }
    lock_guard<mutex> lock(g_seq_log_mutex);
    writeSeqLogBothLocked(seqStep, bytesToHexUpper(plus), bytesToHexUpper(minus));
}

template <size_t N>
static void advanceBig(array<uint8_t, N>& value, uint64_t amount, bool back)
{
    if (back) {
        subBig(value, amount);
    }
    else {
        addBig(value, amount);
    }
}

static inline void mul64x64To128Host(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi)
{
#if defined(_MSC_VER) && defined(_M_X64)
    lo = _umul128(a, b, &hi);
#else
    __uint128_t p = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
    lo = static_cast<uint64_t>(p);
    hi = static_cast<uint64_t>(p >> 64);
#endif
}

static inline uint8_t productByte128(uint64_t lo, uint64_t hi, size_t offset)
{
    return static_cast<uint8_t>((offset < 8)
        ? ((lo >> (offset * 8)) & 0xffu)
        : ((hi >> ((offset - 8) * 8)) & 0xffu));
}

template <size_t N>
static void advanceBigProduct(array<uint8_t, N>& value, uint64_t amount, uint64_t seqStep, bool back)
{
    uint64_t lo = 0;
    uint64_t hi = 0;
    mul64x64To128Host(amount, seqStep, lo, hi);

    if (back) {
        uint16_t borrow = 0;
        for (size_t offset = 0; offset < 16 && offset < N; ++offset) {
            const size_t idx = N - 1 - offset;
            const uint16_t sub = static_cast<uint16_t>(productByte128(lo, hi, offset)) + borrow;
            if (value[idx] >= sub) {
                value[idx] = static_cast<uint8_t>(value[idx] - sub);
                borrow = 0;
            }
            else {
                value[idx] = static_cast<uint8_t>(256u + value[idx] - sub);
                borrow = 1;
            }
        }
        for (size_t idx = (N > 16 ? N - 16 : 0); borrow != 0 && idx > 0;) {
            --idx;
            if (value[idx] >= borrow) {
                value[idx] = static_cast<uint8_t>(value[idx] - borrow);
                borrow = 0;
            }
            else {
                value[idx] = static_cast<uint8_t>(256u + value[idx] - borrow);
                borrow = 1;
            }
        }
    }
    else {
        uint16_t carry = 0;
        for (size_t offset = 0; offset < 16 && offset < N; ++offset) {
            const size_t idx = N - 1 - offset;
            const uint16_t sum = static_cast<uint16_t>(value[idx]) + productByte128(lo, hi, offset) + carry;
            value[idx] = static_cast<uint8_t>(sum & 0xffu);
            carry = static_cast<uint16_t>(sum >> 8);
        }
        for (size_t idx = (N > 16 ? N - 16 : 0); carry != 0 && idx > 0;) {
            --idx;
            const uint16_t sum = static_cast<uint16_t>(value[idx]) + carry;
            value[idx] = static_cast<uint8_t>(sum & 0xffu);
            carry = static_cast<uint16_t>(sum >> 8);
        }
    }
}

template <size_t N>
static bool inRange(const array<uint8_t, N>& value, const array<uint8_t, N>& end, bool back)
{
    return back ? greaterOrEqual(value, end) : lessOrEqual(value, end);
}

static void runRangeSeqChunk(GpuRuntimeContext& processor, const array<uint8_t, 32>& start, bool back, uint64_t seqStep, uint64_t count)
{
    processor.runPrivSeqChunk(start, back, seqStep, count);
}

static void runRangeSeqChunk(GpuRuntimeContext& processor, const array<uint8_t, 256>& start, bool back, uint64_t seqStep, uint64_t count)
{
    processor.runBrainSeqChunk(start, back, seqStep, count);
}

template <size_t N>
static void processRangeBoth(GpuRuntimeContext& processor)
{
    const bool hasRangeEnd = !end_point.empty();
    array<uint8_t, N> plus = bytesFromHexFixed<N>(start_point);
    array<uint8_t, N> minus = hasRangeEnd ? bytesFromHexFixed<N>(end_point) : plus;
    array<uint8_t, N> plusLimit = hasRangeEnd ? minus : array<uint8_t, N>{};
    array<uint8_t, N> minusLimit = plus;
    if (!hasRangeEnd) {
        plusLimit.fill(0xff);
        minusLimit.fill(0x00);
    }
    bool plusActive = !hasRangeEnd || inRange(plus, plusLimit, false);
    bool minusActive = !hasRangeEnd || inRange(minus, minusLimit, true);
    const uint64_t devCount = static_cast<uint64_t>(max(1, processor.deviceCount()));
    const uint64_t devOrdinal = static_cast<uint64_t>(max(0, processor.deviceOrdinal()));

    const uint64_t fullChunk = static_cast<uint64_t>(BLOCK_NUMBER) * static_cast<uint64_t>(BLOCK_THREADS) *
        static_cast<uint64_t>(runKind == RunKind::Priv ? PRIV_SEQ_THREAD_STEPS : THREAD_STEPS_BRAIN);
    if (fullChunk == 0 || mulOverflow64(fullChunk, devOrdinal) || mulOverflow64(fullChunk, devCount)) {
        throw runtime_error("range launch overflow");
    }

    const uint64_t deviceOffset = fullChunk * devOrdinal;
    const uint64_t roundOffset = fullChunk * devCount;
    while (isRun.load(std::memory_order_acquire) && (!hasRangeEnd || plusActive || minusActive)) {
        if (devOrdinal == 0) {
            updateSeqLogCheckpointBoth(step, plus, minus);
        }
        waitSeqRoundBarrier();

        array<uint8_t, N> localPlus = plus;
        array<uint8_t, N> localMinus = minus;
        advanceBigProduct(localPlus, deviceOffset, step, false);
        advanceBigProduct(localMinus, deviceOffset, step, true);
        if ((!hasRangeEnd || plusActive) && inRange(localPlus, plusLimit, false)) {
            runRangeSeqChunk(processor, localPlus, false, step, fullChunk);
        }
        if ((!hasRangeEnd || minusActive) && inRange(localMinus, minusLimit, true)) {
            runRangeSeqChunk(processor, localMinus, true, step, fullChunk);
        }

        waitSeqRoundBarrier();
        array<uint8_t, N> nextPlus = plus;
        array<uint8_t, N> nextMinus = minus;
        advanceBigProduct(nextPlus, roundOffset, step, false);
        advanceBigProduct(nextMinus, roundOffset, step, true);
        if (hasRangeEnd) {
            plusActive = plusActive && lessOrEqual(plus, nextPlus) && nextPlus != plus && inRange(nextPlus, plusLimit, false);
            minusActive = minusActive && lessOrEqual(nextMinus, minus) && nextMinus != minus && inRange(nextMinus, minusLimit, true);
        }
        plus = nextPlus;
        minus = nextMinus;
    }
}

template <size_t N>
static void processRangeTyped(GpuRuntimeContext& processor)
{
    array<uint8_t, N> start = bytesFromHexFixed<N>(start_point);
    array<uint8_t, N> end = end_point.empty()
        ? (backward ? array<uint8_t, N>{} : array<uint8_t, N>{})
        : bytesFromHexFixed<N>(end_point);
    if (end_point.empty() && !backward) {
        end.fill(0xff);
    }

    if (both) {
        processRangeBoth<N>(processor);
        return;
    }

    array<uint8_t, N> cur = start;
    array<uint8_t, N> rangeEnd = end;
    if (backward && !end_point.empty() && lessOrEqual(start, end) && start != end) {
        cur = end;
        rangeEnd = start;
    }
    const uint64_t devCount = static_cast<uint64_t>(max(1, processor.deviceCount()));
    const uint64_t devOrdinal = static_cast<uint64_t>(max(0, processor.deviceOrdinal()));
    const uint64_t fullChunk = static_cast<uint64_t>(BLOCK_NUMBER) * static_cast<uint64_t>(BLOCK_THREADS) *
        static_cast<uint64_t>(runKind == RunKind::Priv ? PRIV_SEQ_THREAD_STEPS : THREAD_STEPS_BRAIN);
    if (fullChunk == 0 || mulOverflow64(fullChunk, devOrdinal) || mulOverflow64(fullChunk, devCount)) {
        throw runtime_error("range launch overflow");
    }

    const uint64_t deviceOffset = fullChunk * devOrdinal;
    const uint64_t roundOffset = fullChunk * devCount;
    while (isRun.load(std::memory_order_acquire) && inRange(cur, rangeEnd, backward)) {
        if (devOrdinal == 0) {
            updateSeqLogCheckpointPoint(step, cur, backward);
        }
        waitSeqRoundBarrier();

        array<uint8_t, N> local = cur;
        advanceBigProduct(local, deviceOffset, step, backward);
        if (inRange(local, rangeEnd, backward)) {
            runRangeSeqChunk(processor, local, backward, step, fullChunk);
        }

        waitSeqRoundBarrier();
        array<uint8_t, N> next = cur;
        advanceBigProduct(next, roundOffset, step, backward);
        if (backward) {
            if (greaterOrEqual(next, cur) && next != cur) {
                break;
            }
        }
        else if (lessOrEqual(next, cur) && next != cur) {
            break;
        }
        cur = next;
    }
}

static void processRange(GpuRuntimeContext& processor)
{
    if (runKind == RunKind::Priv) {
        processRangeTyped<32>(processor);
    }
    else {
        processRangeTyped<256>(processor);
    }
}

static bool prepareMaskSpec(const string& mask, BrainMaskSpec& spec, string& error)
{
    static const string lower = "abcdefghijklmnopqrstuvwxyz";
    static const string upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    static const string digits = "0123456789";
    static const string hexLower = "0123456789abcdef";
    static const string hexUpper = "0123456789ABCDEF";
    static const string symbols = " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

    memset(&spec, 0, sizeof(spec));
    uint32_t charWrite = 0;
    auto addCharset = [&](const string& chars) -> bool {
        if (spec.len >= BRAIN_MASK_MAX_LEN || chars.empty()) {
            return false;
        }
        if (charWrite + chars.size() > BRAIN_MASK_MAX_CHARS) {
            return false;
        }
        spec.charset_offset[spec.len] = static_cast<uint16_t>(charWrite);
        spec.charset_len[spec.len] = static_cast<uint16_t>(chars.size());
        memcpy(spec.chars + charWrite, chars.data(), chars.size());
        charWrite += static_cast<uint32_t>(chars.size());
        ++spec.len;
        return true;
    };
    auto addLiteral = [&](char c) -> bool {
        return addCharset(string(1, c));
    };

    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] != '?') {
            if (!addLiteral(mask[i])) {
                error = "mask is too long";
                return false;
            }
            continue;
        }
        if (i + 1 >= mask.size()) {
            error = "dangling '?' in mask";
            return false;
        }
        const char code = mask[++i];
        string chars;
        switch (code) {
        case '?': chars = "?"; break;
        case 'l': chars = lower; break;
        case 'u': chars = upper; break;
        case 'd': chars = digits; break;
        case 'h': chars = hexLower; break;
        case 'H': chars = hexUpper; break;
        case 's': chars = symbols; break;
        case 'a':
            for (int c = 0x20; c <= 0x7e; ++c) {
                chars.push_back(static_cast<char>(c));
            }
            break;
        case '1':
        case '2':
        case '3':
        case '4':
            chars = CustomSets[code - '1'];
            if (chars.empty()) {
                error = string("-cs") + code + " is empty";
                return false;
            }
            break;
        default:
            error = string("unknown mask token ?") + code;
            return false;
        }
        if (!addCharset(chars)) {
            error = "mask char storage overflow";
            return false;
        }
    }

    if (spec.len == 0) {
        error = "empty mask";
        return false;
    }
    uint64_t total = 1;
    for (uint32_t i = 0; i < spec.len; ++i) {
        const uint64_t len = spec.charset_len[i];
        if (len == 0 || total > numeric_limits<uint64_t>::max() / len) {
            error = "mask candidate count overflows uint64";
            return false;
        }
        total *= len;
    }
    spec.total_charsets = spec.len;
    spec.total_candidates = total;
    return true;
}

static void processMaskText(const string& mask, GpuRuntimeContext& processor)
{
    BrainMaskSpec spec{};
    string error;
    if (!prepareMaskSpec(mask, spec, error)) {
        throw runtime_error("invalid mask '" + mask + "': " + error);
    }
    processor.runMask(spec, spec.total_candidates);
}

static void processMasks(GpuRuntimeContext& processor)
{
    for (const string& mask : MaskList) {
        processMaskText(mask, processor);
    }
    for (const string& file : MaskFileList) {
        ifstream in(file);
        if (!in) {
            throw runtime_error("failed to open mask file: " + file);
        }
        string line;
        while (getline(in, line)) {
            stripCr(line);
            line = trimCopy(line);
            if (line.empty()) continue;
            processMaskText(line, processor);
        }
    }
}

template <size_t N>
static array<uint8_t, N> makeRandomPoint()
{
    array<uint8_t, N> out{};
    if (seqMode) {
        if (end_point.empty()) {
            throw runtime_error("random sequential source requires -end");
        }
        array<uint8_t, N> start = bytesFromHexFixed<N>(start_point);
        array<uint8_t, N> end = bytesFromHexFixed<N>(end_point);
        random_bytes_seq(start.data(), end.data(), out.data(), N);
    }
    else {
        random_bytes(out.data(), N, static_cast<int>(N));
    }
    return out;
}

static void processPrivRandom(GpuRuntimeContext& processor)
{
    const uint64_t outputSizeB_T = static_cast<uint64_t>(BLOCK_NUMBER) *
        static_cast<uint64_t>(BLOCK_THREADS) *
        static_cast<uint64_t>(PRIV_SEQ_THREAD_STEPS);
    if (outputSizeB_T == 0) {
        throw runtime_error("invalid random launch size");
    }
    if (mulOverflow64(step, outputSizeB_T)) {
        throw runtime_error("random step overflow");
    }

    const uint64_t randomSpan = use_n_count ? n_number : 1ull;
    const uint64_t delta = step * outputSizeB_T;
    while (isRun.load(std::memory_order_acquire)) {
        array<uint8_t, 32> point = makeRandomPoint<32>();
        array<uint8_t, 32> plus = point;
        array<uint8_t, 32> minus = point;
        for (uint64_t local_n = 0; isRun.load(std::memory_order_acquire) && local_n <= randomSpan; local_n += outputSizeB_T) {
            updateSeqLogBoth(processor.deviceOrdinal(), step, plus, minus);
            processor.runPrivSeqChunk(plus, false, step, outputSizeB_T);
            advanceBig(plus, delta, false);
            if (numeric_limits<uint64_t>::max() - local_n < outputSizeB_T) {
                break;
            }
        }
        for (uint64_t local_n = 0; isRun.load(std::memory_order_acquire) && local_n <= randomSpan; local_n += outputSizeB_T) {
            updateSeqLogBoth(processor.deviceOrdinal(), step, plus, minus);
            processor.runPrivSeqChunk(minus, true, step, outputSizeB_T);
            advanceBig(minus, delta, true);
            if (numeric_limits<uint64_t>::max() - local_n < outputSizeB_T) {
                break;
            }
        }
    }
}

static void processBrainRandom(GpuRuntimeContext& processor)
{
    const uint64_t outputSizeB_T = static_cast<uint64_t>(BLOCK_NUMBER) *
        static_cast<uint64_t>(BLOCK_THREADS) *
        static_cast<uint64_t>(THREAD_STEPS_BRAIN);
    if (outputSizeB_T == 0) {
        throw runtime_error("invalid random launch size");
    }
    if (mulOverflow64(step, outputSizeB_T)) {
        throw runtime_error("random step overflow");
    }

    const uint64_t randomSpan = use_n_count ? n_number : 1ull;
    const uint64_t delta = step * outputSizeB_T;
    while (isRun.load(std::memory_order_acquire)) {
        array<uint8_t, 256> point = makeRandomPoint<256>();
        array<uint8_t, 256> plus = point;
        array<uint8_t, 256> minus = point;
        for (uint64_t local_n = 0; isRun.load(std::memory_order_acquire) && local_n <= randomSpan; local_n += outputSizeB_T) {
            updateSeqLogBoth(processor.deviceOrdinal(), step, plus, minus);
            processor.runBrainSeqChunk(plus, false, step, outputSizeB_T);
            advanceBig(plus, delta, false);
            if (numeric_limits<uint64_t>::max() - local_n < outputSizeB_T) {
                break;
            }
        }
        for (uint64_t local_n = 0; isRun.load(std::memory_order_acquire) && local_n <= randomSpan; local_n += outputSizeB_T) {
            updateSeqLogBoth(processor.deviceOrdinal(), step, plus, minus);
            processor.runBrainSeqChunk(minus, true, step, outputSizeB_T);
            advanceBig(minus, delta, true);
            if (numeric_limits<uint64_t>::max() - local_n < outputSizeB_T) {
                break;
            }
        }
    }
}

static void processRandom(GpuRuntimeContext& processor)
{
    if (runKind == RunKind::Priv) {
        processPrivRandom(processor);
    }
    else {
        processBrainRandom(processor);
    }
}

class SharedLineSource {
public:
    SharedLineSource(const vector<string>& entries, bool fileMode)
        : entries_(&entries), fileMode_(fileMode)
    {
    }

    explicit SharedLineSource(std::istream& stream)
        : stream_(&stream)
    {
    }

    bool next(string& line)
    {
        lock_guard<mutex> lock(mutex_);
        if (stream_ != nullptr) {
            if (!read_trimmed_line(*stream_, line, 512)) {
                return false;
            }
            emitted_++;
            return true;
        }
        if (!fileMode_) {
            if (entries_ == nullptr || entryIndex_ >= entries_->size()) {
                return false;
            }
            line = (*entries_)[entryIndex_++];
            stripCr(line);
            emitted_++;
            return true;
        }
        while (entries_ != nullptr && entryIndex_ < entries_->size()) {
            if (!current_.is_open()) {
                current_.open((*entries_)[entryIndex_], ios::binary);
                tune_ifstream_buffer(current_);
                if (!current_) {
                    throw runtime_error("failed to open input file: " + (*entries_)[entryIndex_]);
                }
            }
            if (read_trimmed_line(current_, line, 512)) {
                emitted_++;
                return true;
            }
            current_.close();
            current_.clear();
            entryIndex_++;
        }
        return false;
    }

private:
    const vector<string>* entries_ = nullptr;
    bool fileMode_ = false;
    std::istream* stream_ = nullptr;
    mutex mutex_;
    size_t entryIndex_ = 0;
    uint64_t emitted_ = 0;
    ifstream current_;
};

static inline bool read_process_line(std::istream& stream, std::string& buffer, uint64_t& seen)
{
    if (!read_trimmed_line(stream, buffer, 512)) {
        return false;
    }
    seen++;
    return true;
}

static inline bool read_process_line(SharedLineSource& source, std::string& buffer, uint64_t&)
{
    return source.next(buffer);
}

template <typename Source>
static cudaError_t processCudaPRIV(Source& stream, GpuRuntimeContext& processor)
{
    cudaError_t cudaStatus = cudaSuccess;
    const uint64_t outputSizeB_T = static_cast<uint64_t>(BLOCK_NUMBER) * BLOCK_THREADS * THREAD_STEPS;
    if (outputSizeB_T > numeric_limits<uint32_t>::max()) {
        return cudaErrorInvalidConfiguration;
    }

    std::string buffer;
    std::string combined1;
    std::vector<uint32_t> indexes1;
    std::string combined2;
    std::vector<uint32_t> indexes2;
    reserve_batch_buffers(combined1, indexes1, outputSizeB_T, 64);
    reserve_batch_buffers(combined2, indexes2, outputSizeB_T, 64);

    bool isData = true;
    bool data1ready = false;
    bool data2ready = false;
    uint32_t nr1 = 0;
    uint32_t nr2 = 0;
    string nr1Last = "";
    string nr2Last = "";
    uint64_t seen = 0;

    while (isData) {
        combined1.clear();
        indexes1.clear();
        nr1 = 0;
        data1ready = false;
        while (read_process_line(stream, buffer, seen))
        {
            if (buffer.length() == 0) {
                continue;
            }
            if (!append_normalized_priv_line(buffer, combined1, indexes1, nr1, IS_HEX)) {
                continue;
            }
            nr1Last = buffer;
            if (nr1 < outputSizeB_T) {
                continue;
            }
            data1ready = true;
            break;
        }
        if (!data1ready) {
            isData = false;
        }
        if (nr1 > 0) {
            processor.uploadBatch(0, combined1, indexes1);
        }
        if (nr2 > 0) {
            processor.syncAndSave(true);
        }
        if (nr1 > 0) {
            processor.launchBatch(0, nr1);
        }
        processor.accountBatch(nr1);

        combined2.clear();
        indexes2.clear();
        nr2 = 0;
        data2ready = false;
        while (read_process_line(stream, buffer, seen))
        {
            if (buffer.length() == 0) {
                continue;
            }
            if (!append_normalized_priv_line(buffer, combined2, indexes2, nr2, IS_HEX)) {
                continue;
            }
            nr2Last = buffer;
            if (nr2 < outputSizeB_T) {
                continue;
            }
            data2ready = true;
            break;
        }
        if (!data2ready) {
            isData = false;
        }
        if (nr2 > 0) {
            processor.uploadBatch(1, combined2, indexes2);
        }
        if (nr1 > 0) {
            processor.syncAndSave(true);
        }
        if (nr2 > 0) {
            processor.launchBatch(1, nr2);
        }
        processor.accountBatch(nr2);
    }
    processor.syncAndSave(true);
    return cudaStatus;
}

template <typename Source>
static cudaError_t processCudaBrain(Source& stream, GpuRuntimeContext& processor)
{
    cudaError_t cudaStatus = cudaSuccess;
    const uint64_t outputSizeB_T64 = static_cast<uint64_t>(BLOCK_NUMBER) * BLOCK_THREADS * THREAD_STEPS_BRAIN;
    if (outputSizeB_T64 > numeric_limits<uint32_t>::max()) {
        return cudaErrorInvalidConfiguration;
    }
    const uint32_t outputSizeB_T = static_cast<uint32_t>(outputSizeB_T64);

    std::string buffer;
    std::string combined1;
    std::vector<uint32_t> indexes1;
    std::string combined2;
    std::vector<uint32_t> indexes2;
    reserve_batch_buffers(combined1, indexes1, outputSizeB_T, 2048);
    reserve_batch_buffers(combined2, indexes2, outputSizeB_T, 2048);

    bool isData = true;
    bool data1ready = false;
    bool data2ready = false;
    uint32_t nr1 = 0;
    uint32_t nr2 = 0;
    string nr1Last = "";
    string nr2Last = "";
    uint64_t seen = 0;

    while (isData) {
        combined1.clear();
        indexes1.clear();
        nr1 = 0;
        data1ready = false;
        while (read_process_line(stream, buffer, seen))
        {
            if (buffer.length() == 0) {
                continue;
            }
            if (IS_HEX)
            {
                string unhexed;
                unhexed.resize(buffer.length() / 2);
                unhex_host((uint8_t*)buffer.data(), buffer.length(), (uint8_t*)unhexed.data(), buffer.length() / 2);
                buffer.swap(unhexed);
            }
            combined1 += buffer;
            nr1Last = buffer;
            indexes1.emplace_back(static_cast<uint32_t>(combined1.size()));
            nr1++;
            if (nr1 < outputSizeB_T) {
                continue;
            }
            data1ready = true;
            break;
        }
        if (!data1ready) {
            isData = false;
        }
        if (nr1 > 0) {
            processor.uploadBatch(0, combined1, indexes1);
        }
        if (nr2 > 0) {
            processor.syncAndSave(false);
        }
        if (nr1 > 0) {
            processor.launchBatch(0, nr1);
        }
        processor.accountBatch(nr1);

        combined2.clear();
        indexes2.clear();
        nr2 = 0;
        data2ready = false;
        while (read_process_line(stream, buffer, seen))
        {
            if (buffer.length() == 0) {
                continue;
            }
            if (IS_HEX)
            {
                string unhexed;
                unhexed.resize(buffer.length() / 2);
                unhex_host((uint8_t*)buffer.data(), buffer.length(), (uint8_t*)unhexed.data(), buffer.length() / 2);
                buffer.swap(unhexed);
            }
            combined2 += buffer;
            nr2Last = buffer;
            indexes2.emplace_back(static_cast<uint32_t>(combined2.size()));
            nr2++;
            if (nr2 < outputSizeB_T) {
                continue;
            }
            data2ready = true;
            break;
        }
        if (!data2ready) {
            isData = false;
        }
        if (nr2 > 0) {
            processor.uploadBatch(1, combined2, indexes2);
        }
        if (nr1 > 0) {
            processor.syncAndSave(false);
        }
        if (nr2 > 0) {
            processor.launchBatch(1, nr2);
        }
        processor.accountBatch(nr2);
    }
    processor.syncAndSave(false);
    return cudaStatus;
}

static void processStream(istream& in, GpuRuntimeContext& processor)
{
    cudaError_t st = (runKind == RunKind::Priv) ? processCudaPRIV(in, processor) : processCudaBrain(in, processor);
    if (st != cudaSuccess) {
        throw runtime_error(cudaGetErrorString(st));
    }
}

static void processSharedInputs(SharedLineSource& source, GpuRuntimeContext& processor)
{
    cudaError_t st = (runKind == RunKind::Priv) ? processCudaPRIV(source, processor) : processCudaBrain(source, processor);
    if (st != cudaSuccess) {
        throw runtime_error(cudaGetErrorString(st));
    }
}

static void processInputs(GpuRuntimeContext& processor, bool deleteAfter)
{
    if (isRandom) {
        processRandom(processor);
        return;
    }
    if (seqMode) {
        processRange(processor);
        return;
    }
    if (!MaskList.empty() || !MaskFileList.empty()) {
        processMasks(processor);
        return;
    }
    if (!inputFiles.empty()) {
        for (const string& file : inputFiles) {
            ifstream in(file, ios::binary);
            tune_ifstream_buffer(in);
            if (!in) {
                throw runtime_error("failed to open input file: " + file);
            }
            processStream(in, processor);
            if (deleteAfter && deleteFile) {
                if (std::remove(file.c_str()) != 0) {
                    cerr << "[!] failed to delete input file: " << file << " [!]\n";
                }
            }
        }
        return;
    }
    processStream(cin, processor);
}

struct DeviceRunResult {
    bool ok = false;
    cudaError_t cudaStatus = cudaSuccess;
    string error;
    uint64_t processed = 0;
    uint32_t found = 0;
};

class DeviceInitBarrier {
public:
    explicit DeviceInitBarrier(size_t total) : total_(total) {}

    bool arriveAndWait()
    {
        unique_lock<mutex> lock(mutex_);
        if (failed_) {
            return false;
        }
        ++ready_;
        if (ready_ == total_) {
            if (!printed_) {
                printSequentialRangeBounds();
                printed_ = true;
            }
            cv_.notify_all();
            return true;
        }
        cv_.wait(lock, [&]() { return failed_ || ready_ >= total_; });
        return !failed_;
    }

    void fail()
    {
        lock_guard<mutex> lock(mutex_);
        failed_ = true;
        cv_.notify_all();
    }

private:
    mutex mutex_;
    condition_variable cv_;
    size_t total_ = 0;
    size_t ready_ = 0;
    bool failed_ = false;
    bool printed_ = false;
};

static DeviceRunResult runDevice(size_t ordinal, size_t deviceCount, SharedLineSource* sharedLines, FILE* outputFile, DeviceInitBarrier* initBarrier)
{
    DeviceRunResult result;
    auto cleanup = []() {
        if (_dev_precomp != nullptr) {
            cudaFree(_dev_precomp);
            _dev_precomp = nullptr;
        }
        freeFoundBuffers();
    };
    try {
        if (!checkDevice(DeviceList[ordinal])) {
            result.cudaStatus = cudaErrorInvalidDevice;
            result.error = "device initialization failed";
            if (initBarrier != nullptr) {
                initBarrier->fail();
            }
            cleanup();
            return result;
        }
        cudaError_t cudaStatus = prepareCuda();
        if (cudaStatus != cudaSuccess) {
            result.cudaStatus = cudaStatus;
            result.error = string("CUDA prepare failed: ") + cudaGetErrorString(cudaStatus);
            if (initBarrier != nullptr) {
                initBarrier->fail();
            }
            cleanup();
            return result;
        }

        GpuRuntimeContext processor(static_cast<int>(ordinal), static_cast<int>(deviceCount), outputFile);
        if (initBarrier != nullptr && !initBarrier->arriveAndWait()) {
            result.cudaStatus = cudaErrorUnknown;
            result.error = "device initialization failed";
            cleanup();
            return result;
        }
        if (sharedLines != nullptr) {
            processSharedInputs(*sharedLines, processor);
        }
        else {
            processInputs(processor, false);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            result.cudaStatus = cudaStatus;
            result.error = cudaGetErrorString(cudaStatus);
            cleanup();
            return result;
        }
        wait_for_current_gpu_async_save_queue();
        result.processed = processor.processed();
        result.found = processor.found();
        result.ok = true;
    }
    catch (const exception& ex) {
        result.cudaStatus = cudaErrorUnknown;
        result.error = ex.what();
        if (g_seq_round_barrier != nullptr) {
            g_seq_round_barrier->abort();
        }
        if (initBarrier != nullptr) {
            initBarrier->fail();
        }
    }
    cleanup();
    return result;
}

int main(int argc, char** argv)
{
#if defined(_WIN32) || defined(_WIN64)
    SetConsoleOutputCP(CP_UTF8);
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    setlocale(LC_ALL, "en_US.UTF-8");
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    bool help = false;
    string error;
    if (!parseArgs(argc, argv, help, error)) {
        cerr << "[!] " << error << " [!]\n\n";
        printHelp();
        return 1;
    }
    if (help) {
        printHelp();
        return 0;
    }
	cerr << "[!] ================== Brainflayer CUDA by @XopMC ================== [!]" << "\n";

#ifndef _DEBUG
    PARAM_ECMULT_WINDOW_SIZE = (runKind == RunKind::Priv) ? 18u : 16u;
#endif

    FILE* outputFile = nullptr;
    try {
        for (const string& dir : FolderList) {
            vector<string> files = scanDir(dir, all_file);
            inputFiles.insert(inputFiles.end(), files.begin(), files.end());
        }
        if (!inputFiles.empty()) {
            useStdin = false;
        }
        auto start = chrono::steady_clock::now();
        std::time_t startWall = chrono::system_clock::to_time_t(chrono::system_clock::now());

        if (silent) {
            setSilentMode();
        }
        g_save_output_with_gpu_prefix = DeviceList.size() > 1;
        outputFile = fopen(fileResult.c_str(), "a");
        if (outputFile == nullptr) {
            cerr << "[!] failed to open output file: " << fileResult << " [!]\n";
            return 1;
        }
        cout << "[!] Program started at: " << std::ctime(&startWall);
        printBranchStartup();
        cout.flush();
        fflush(stdout);
        fflush(stderr);
        counterTotal.store(0, std::memory_order_release);
        g_runtime_founds.store(0, std::memory_order_release);
        false_positive.store(0, std::memory_order_release);
        isRun.store(true, std::memory_order_release);
        speedThread = thread(SpeedThreadFunc);

        vector<DeviceRunResult> results(DeviceList.size());
        vector<thread> workers;
        workers.reserve(DeviceList.size());
        unique_ptr<SharedLineSource> sharedLines;
        DeviceInitBarrier initBarrier(DeviceList.size());
        unique_ptr<SeqRoundBarrier> seqRoundBarrier;
        if (seqMode && DeviceList.size() > 1) {
            seqRoundBarrier = make_unique<SeqRoundBarrier>(DeviceList.size());
            g_seq_round_barrier = seqRoundBarrier.get();
        }
        const bool streamMode = !isRandom && !seqMode && MaskList.empty() && MaskFileList.empty();
        if (streamMode && DeviceList.size() > 1) {
            if (!inputFiles.empty()) {
                sharedLines = make_unique<SharedLineSource>(inputFiles, true);
            }
            else if (useStdin) {
                sharedLines = make_unique<SharedLineSource>(cin);
            }
        }
        for (size_t i = 0; i < DeviceList.size(); ++i) {
            workers.emplace_back([&, i]() {
                results[i] = runDevice(i, DeviceList.size(), sharedLines.get(), outputFile, &initBarrier);
            });
        }
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        g_seq_round_barrier = nullptr;
        isRun.store(false, std::memory_order_release);
        if (speedThread.joinable()) {
            speedThread.join();
        }
        flush_all_async_save_queues();
        shutdown_async_save_queues();
        auto finish = chrono::steady_clock::now();
        const double sec = chrono::duration<double>(finish - start).count();
        const double finalSpeed = sec > 0.0 ? static_cast<double>(counterTotal.load(std::memory_order_relaxed)) / sec : 0.0;
        printSpeed(finalSpeed);
        printf("\n");
        if (outputFile != nullptr) {
            fflush(outputFile);
            fclose(outputFile);
            outputFile = nullptr;
        }

        bool ok = true;
        uint64_t processedTotal = 0;
        uint64_t foundTotal = 0;
        for (size_t i = 0; i < results.size(); ++i) {
            processedTotal += results[i].processed;
            foundTotal += results[i].found;
            if (!results[i].ok) {
                ok = false;
                cerr << "[!] GPU " << DeviceList[i] << " failed: "
                    << (results[i].error.empty() ? cudaGetErrorString(results[i].cudaStatus) : results[i].error)
                    << " [!]\n";
            }
        }
        if (!ok) {
            return 1;
        }

        if (deleteFile) {
            for (const string& file : inputFiles) {
                if (std::remove(file.c_str()) != 0) {
                    cerr << "[!] failed to delete input file: " << file << " [!]\n";
                }
            }
        }

        std::time_t finishWall = chrono::system_clock::to_time_t(chrono::system_clock::now());
        if (runKind == RunKind::Priv) {
            std::cout << "\n[!] Processed " << counterTotal.load(std::memory_order_relaxed) << " Keys."
                << " Found: " << g_runtime_founds.load(std::memory_order_relaxed)
                << ". Program finished at " << std::ctime(&finishWall);
        }
        else {
            std::cout << "\n[!] Processed " << counterTotal.load(std::memory_order_relaxed) << " lines."
                << " Found: " << g_runtime_founds.load(std::memory_order_relaxed)
                << ". Program finished at " << std::ctime(&finishWall);
        }

        (void)processedTotal;
        (void)foundTotal;
    }
    catch (const exception& ex) {
        isRun.store(false, std::memory_order_release);
        if (speedThread.joinable()) {
            speedThread.join();
        }
        flush_all_async_save_queues();
        shutdown_async_save_queues();
        if (outputFile != nullptr) {
            fflush(outputFile);
            fclose(outputFile);
            outputFile = nullptr;
        }
        cerr << "[!] " << ex.what() << " [!]\n";
        return 1;
    }
    return 0;
}

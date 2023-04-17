// #include "cuda.h"
// #include "common/errors.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <charconv>
#include <algorithm>
#include <numeric>


namespace {

// Graph traversal for graph orientation method
// 1 𝑛𝑢𝑚𝐶𝑙𝑖𝑞𝑢𝑒𝑠 = 0
// 2 procedure 𝑡𝑟𝑎𝑣𝑒𝑟𝑠𝑒𝑆𝑢𝑏𝑡𝑟𝑒𝑒 (𝐺, 𝑘, ℓ, 𝐼 ) : (G: Graph, k: clique_size, l: current_level, I: set_of_vertices)
// 3 for 𝑣 ∈ 𝐼
// 4    𝐼 ′ = 𝐼 ∩ 𝐴𝑑𝑗_𝐺 (𝑣)
// 5    if ℓ + 1 == 𝑘
// 6        𝑛𝑢𝑚𝐶𝑙𝑖𝑞𝑢𝑒𝑠 + = |𝐼 ′ |
// 7    else if |𝐼 ′ | > 0
// 8        𝑡𝑟𝑎𝑣𝑒𝑟𝑠𝑒𝑆𝑢𝑏𝑡𝑟𝑒𝑒 (𝐺, 𝑘, ℓ + 1, 𝐼 ′ )

    using Edge = std::pair<int, int>;

    int find_max_vertex(std::vector<Edge> const& edges) {
        int max_vertex = 0;
        for (auto const [v1, v2]: edges) {
            if (v1 > max_vertex) {
                max_vertex = v1;
            }
            if (v2 > max_vertex) {
                max_vertex = v2;
            }
        }
        // std::cerr << "Max vertex found: " << max_vertex << std::endl;
        return max_vertex;
    }

    struct CSR {
        std::vector<int> col_idx;
        std::vector<int> row_ptr;
        int max_v;
        int n;

        CSR(std::vector<Edge> const& edges) : max_v{find_max_vertex(edges)}, n{max_v + 1} {
            col_idx.resize(edges.size());
            row_ptr.resize(n + 1);

            int col_i = 0;
            for (int row = 0; row <= max_v; ++row) {
                row_ptr[row] = col_i;
                while (col_i < edges.size() && edges[col_i].first == row) {
                    // std::cerr << "Col_i: " << col_i << std::endl;
                    col_idx[col_i] = edges[col_i].second;
                    ++col_i;
                }
            }
            row_ptr[n] = col_idx.size();
        }
    };

    std::ostream& operator<<(std::ostream &os, CSR const& csr) {
        os << "Col_idx: [ ";
        for (int col: csr.col_idx) {
            os << col << ", ";
        }
        os << " ]\n";
        os << "Row_ptr: [ ";
        for (int row: csr.row_ptr) {
            os << row << ", ";
        }
        os << "]\n";

        return os;
    }

    Edge parse_edge(std::string const& buf) {
        char const* ptr = buf.data();
        int v1, v2;
        auto res1 = std::from_chars(ptr, ptr + buf.size(), v1);
        if (res1.ec != std::errc()) {
            std::cerr << "Error while parsing int!\n";
            exit(EXIT_FAILURE);
        }
        ptr = res1.ptr;
        while (std::isspace(*ptr)) ++ptr;

        auto res2 = std::from_chars(ptr, buf.data() + buf.size(), v2);
        if (res2.ec != std::errc()) {
            std::cerr << "Error while parsing int!\n";
            exit(EXIT_FAILURE);
        }
        return {v1, v2};
    }

    struct VertexSet {
        std::vector<bool> vertices;
    private:
        VertexSet(size_t n) {
            vertices.resize(static_cast<std::vector<bool>::size_type>(n));
        }
    public:
        bool is_empty() {
            for (auto v: vertices) {
                if (v)
                    return false;
            }
            return true;
        }

        static VertexSet empty(size_t n) {
            return {n};
        }
        static VertexSet full(size_t n) {
            VertexSet full{VertexSet::empty(n)};
            full.vertices.flip();
            return full;
        }

        bool operator[](size_t idx) {
            return vertices[idx];
        }

        VertexSet intersect_adjacent(CSR const& adjacency, int vertex) const {
            VertexSet set{this->vertices.size()};
            // std::cerr << "vertex: " << vertex << std::endl;
            int const row_beg = adjacency.row_ptr[vertex];
            int const row_end = adjacency.row_ptr[vertex + 1];
            // std::cerr << "row_beg: " << row_beg << ", row_end: " << row_end << std::endl;
            for (auto it = adjacency.col_idx.begin() + row_beg; it < adjacency.col_idx.begin() + row_end; ++it) {
                int const neighbour = *it;
                // std::cerr << "neighbour: " << neighbour << std::endl;
                if (this->vertices[neighbour])
                    set.vertices[neighbour] = true;
            }
            return set;
        }
    };

    struct StackEntry {
        VertexSet vertices;
        std::vector<int> clique_counter;
        int stack_vertex;
        int chosen_vertex;
        int level;
        int k;
        StackEntry(CSR const& graph, VertexSet vertices, int k, int stack_vertex, int vertex, int level)
        : chosen_vertex{vertex}, /* vertices{VertexSet::empty(static_cast<size_t>(graph.max_v))} */
          vertices{std::move(vertices)}, k{k}, level{level}, stack_vertex{stack_vertex}
        {
            clique_counter.resize(k);
        }
    };


    struct Stack {
        std::vector<StackEntry> entries;

        void push(StackEntry entry) {
            entries.push_back(std::move(entry));
        }

        bool is_empty() const {
            return entries.empty();
        }

        template <typename... Args>
        void emplace(Args&& ...args) {
            entries.emplace_back(args...);
        }

        StackEntry pop() {
            StackEntry entry {std::move(entries.back())};
            entries.pop_back();
            return entry;
        }
    };


    struct CPUAlgorithm {
        CSR edges;
        int k;
        std::vector<Stack> stacks;

        CPUAlgorithm(CSR edges, int k) : edges{edges}, stacks{static_cast<size_t>(edges.n)}, k{k} {}

// Graph traversal for graph orientation method
// 1 𝑛𝑢𝑚𝐶𝑙𝑖𝑞𝑢𝑒𝑠 = 0
// 2 procedure 𝑡𝑟𝑎𝑣𝑒𝑟𝑠𝑒𝑆𝑢𝑏𝑡𝑟𝑒𝑒 (𝐺, 𝑘, ℓ, 𝐼 ) : (G: Graph, k: clique_size, l: current_level, I: set_of_vertices)
// 3 for 𝑣 ∈ 𝐼
// 4    𝐼 ′ = 𝐼 ∩ 𝐴𝑑𝑗_𝐺 (𝑣)
// 5    if ℓ + 1 == 𝑘
// 6        𝑛𝑢𝑚𝐶𝑙𝑖𝑞𝑢𝑒𝑠 + = |𝐼 ′ |
// 7    else if |𝐼 ′ | > 0
// 8        𝑡𝑟𝑎𝑣𝑒𝑟𝑠𝑒𝑆𝑢𝑏𝑡𝑟𝑒𝑒 (𝐺, 𝑘, ℓ + 1, 𝐼 ′ )
        std::vector<int> count_cliques() {
            std::vector<int> count;
            count.resize(k);
            count[0] = edges.n;
            for (int v = 0; v <= edges.max_v; ++v) {
                // std::cerr << "First for v=" << v << '\n';
                auto& stack = stacks[v];
                stack.emplace(edges, VertexSet::full(edges.n), k, v, v, 1);
            }
            int i = 0;
            for (auto& stack: stacks) {
                std::cerr << "\nSecond for stack " << i++ << "\n";
                while (!stack.is_empty()) {
                    auto entry = stack.pop();
                    std::cerr << "Entry{level=" << entry.level << ", stack_vertex=" << entry.stack_vertex
                        << ", vertex=" << entry.chosen_vertex << "}" << std::endl;
                    auto new_vertices = entry.vertices.intersect_adjacent(edges, entry.chosen_vertex);
                    for (int v = 0; v <= edges.max_v; ++v) {
                        if (new_vertices[v]) {
                            std::cerr << "There exist an edge from " << entry.chosen_vertex << " to " << v
                                << " on level " << entry.level << '.' << std::endl;
                            // if (entry.level + 1 >= 4) {
                            ++count[entry.level];
                            // }
                            if (entry.level + 1 < k && !new_vertices.is_empty()) {
                                auto& stack = stacks[entry.stack_vertex];
                                stack.emplace(edges, new_vertices, k, entry.stack_vertex, v, entry.level + 1);
                            }
                        }
                    }
                }
            }
            return count;
        }
    };
}

int main(int argc, char const* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Bad arg num (expected 4, got %d)\n", argc);
        return EXIT_FAILURE;
    }

    char const* input_filename = argv[1];
    char const* k_str = argv[2];
    char const* output_filename = argv[3];

    std::ifstream input_file{input_filename, std::ios::in};
    if (!input_file.is_open()) {
        std::cerr << "Could not open input file '" << input_filename << "'!\n";
        return EXIT_FAILURE;
    }

    int k;
    try {
        k = std::stoi(k_str);
    } catch (std::invalid_argument&) {
        std::cerr << "Non integer k: " << k_str << "'!\n";
        return EXIT_FAILURE;
    } catch (std::out_of_range&) {
        std::cerr << "k too big for int type: " << k_str << "'!\n";
        return EXIT_FAILURE;
    }

    std::ofstream output_file{output_filename, std::ios::out};
    if (!output_file.is_open()) {
        std::cerr << "Could not open output file '" << output_filename << "'!\n";
        return EXIT_FAILURE;
    }

    std::vector<Edge> edges;
    std::string buffer;

    while (input_file.good() && !input_file.eof()) {
        std::getline(input_file, buffer);
        auto const edge = parse_edge(buffer);
        edges.push_back(edge);
    }

    if (input_file.bad()) {
        std::cerr << "Error while reading from file!\n";
        return EXIT_FAILURE;
    }

    std::sort(edges.begin(), edges.end());
    for (auto const [v1, v2]: edges) {
        std::cout << "(" << v1 << ", " << v2 << ")\n";
    }

    CSR graph{edges};
    std::cout << graph << "\n";

    CPUAlgorithm algo{edges, k};
    auto count = algo.count_cliques();
    std::cout << "count: [ ";
    for (auto c: count) {
        output_file << c << ' ';
        std::cout << c << ' ';
    }
    std::cout << "]\n";



    return EXIT_SUCCESS;
}
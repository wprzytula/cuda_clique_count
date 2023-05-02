// #include "cuda.h"
// #include "common/errors.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <utility>
#include <charconv>
#include <algorithm>
#include <numeric>
#include <cassert>

#define NSTACKS 1

#ifdef PRINT
constexpr bool const debug = true;
#else
constexpr bool const debug = false;
#endif

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

    int compute_max_deg(std::vector<Edge> const& edges, int max_vertex) {
        std::vector<int> deg;
        deg.resize(max_vertex + 1);

        for (auto const [v1, v2]: edges) {
            ++deg[v1];
            ++deg[v2];
        }

        int const max_deg = *std::max_element(deg.cbegin(), deg.cend());
        std::cerr << "Max deg found: " << max_deg << std::endl;
        return max_deg;
    }

    std::vector<int> compute_degs(std::vector<Edge> const& edges, int max_vertex) {
        std::vector<int> deg;
        deg.resize(max_vertex + 1);

        for (auto const [v1, v2]: edges) {
            ++deg[v1];
            ++deg[v2];
        }

        return deg;
    }

    void orient_graph(std::vector<Edge>& edges, std::vector<int> const& deg) {
        for (Edge& edge: edges) {
            auto [v1, v2] = edge;
            int const deg_v1 = deg[v1];
            int const deg_v2 = deg[v2];
            if (deg_v1 > deg_v2 || (deg_v1 == deg_v2 && v1 > v2)) {
                // Revert the edge
                edge.first = v2;
                edge.second = v1;
            }
        }
    }

    struct CSR {
        std::vector<int> col_idx;
        std::vector<int> row_ptr;
        int max_v;
        int n;

        CSR(std::vector<Edge> const& edges) : max_v{find_max_vertex(edges)}, n{max_v + 1} {
            assert(std::is_sorted(edges.cbegin(), edges.cend()));

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
            std::cerr << "Error while parsing first vertex int!\n";
            std::cerr << "(problematic line: " << ptr << ")\n";
            exit(EXIT_FAILURE);
        }
        ptr = res1.ptr;
        while (std::isspace(*ptr)) ++ptr;

        auto res2 = std::from_chars(ptr, buf.data() + buf.size(), v2);
        if (res2.ec != std::errc()) {
            std::cerr << "Error while parsing second vertex int!\n";
            std::cerr << "(problematic line: " << ptr << ")\n";
            exit(EXIT_FAILURE);
        }
        return {v1, v2};
    }

    struct InducedSubgraph {
        std::vector<int> const mapping;
        std::vector<std::vector<bool>> const adjacency_matrix;
private:
        InducedSubgraph(std::vector<int> mapping, std::vector<std::vector<bool>> adjacency_matrix)
            : mapping{std::move(mapping)}, adjacency_matrix{std::move(adjacency_matrix)} {}

public:
        static InducedSubgraph extract(CSR const& graph, int vertex) {
            int const i = vertex;

/* Build subgraph mapping: new_vertex [0..1024] -> old_vertex [0..|V|] */
            std::vector<int> subgraph_mapping;
            int const start = graph.row_ptr[i];
            int const end = graph.row_ptr[i + 1];
            for (int j = start; j < end; ++j) {
                // put neighbours in mapping.
                int const neighbour = graph.col_idx[j];
                subgraph_mapping.push_back(neighbour);
            }

/* Build adjacency matrix  */
            std::vector<std::vector<bool>> adjacency_matrix;

            // It has k rows, where k = |induced subgraph vertices|
            adjacency_matrix.resize(subgraph_mapping.size());

            auto old = [&subgraph_mapping](int new_v){/* std::cout << "old(" << new_v << ")\n";  */return subgraph_mapping[new_v];};
            auto neigh = [&graph](int col_i){return graph.col_idx[col_i];};

            // For each row
            for (int i = 0; i < subgraph_mapping.size(); ++i) {
                // Retrieve old id of the vertex
                int const old_v1 = subgraph_mapping[i];
                // std::cout << "Row with new id: " << i << ", old id: " << old_v1 << "\n";

                // Operate on this row
                auto& row = adjacency_matrix[i];
                // Resize it to k
                row.resize(subgraph_mapping.size());

                int csr_idx = graph.row_ptr[old_v1];
                int const csr_idx_end = graph.row_ptr[old_v1 + 1];

                // For each cell in this row
                for (int adj_idx = 0; adj_idx < subgraph_mapping.size(); ++adj_idx) {
                    // std::cout << "Incremented adj_idx to " << adj_idx << ", now points to " << old(adj_idx) << "\n";

                    if (csr_idx >= csr_idx_end) {
                            // std::cout << "csr_idx went out of bounds.\n";
                            goto end_row;
                    }

                    while (neigh(csr_idx) < old(adj_idx)) {
                        // std::cout << "Incremented csr_idx to " << csr_idx << "\n";
                        ++csr_idx;
                        if (csr_idx >= csr_idx_end) {
                            // std::cout << "csr_idx went out of bounds.\n";
                            goto end_row;
                        }
                        // std::cout << "csr_idx now points to " << neigh(csr_idx) << "\n";
                    }

                    // printf("Deciding edge between %d and %d based on value in csr_idx under %d: %d\n",
                    //      old_v1, old(adj_idx), csr_idx, neigh(csr_idx));
                    row[adj_idx] = neigh(csr_idx) == old(adj_idx);
end_row:            ;
                }
            }
            return InducedSubgraph{subgraph_mapping, adjacency_matrix};
        }
        InducedSubgraph operator=(InducedSubgraph const&) = delete;
    };
    std::ostream& operator<<(std::ostream &os, InducedSubgraph const& subgraph) {
            os << "Subgraph mapping: [ ";
            for (int old_v: subgraph.mapping) {
                os << old_v << " ";
            }
            os << "]\n";
            os << "Adjacency matrix:\n";
            os << "  ";
            for (int old_v: subgraph.mapping) {
                os << old_v << " ";
            }
            os << "\n";
            for (auto const& row: subgraph.adjacency_matrix) {
                os << "[";
                for (bool exists: row) {
                    os << ' ' << (exists ? 'x' : ' ');
                }
                os << " ]\n";
            }

            return os;
        }

    struct VertexSet;
    std::ostream& operator<<(std::ostream &os, VertexSet const& vertex_set);

    struct VertexSet {
        std::vector<bool> vertices;
        int elems;
    private:
        VertexSet(int n) {
            vertices.resize(n);
        }
        VertexSet(std::vector<bool> vertices) : vertices{std::move(vertices)} {
            elems = std::accumulate(vertices.cbegin(), vertices.cend(), 0);
        }
        static VertexSet empty(int n) {
            return VertexSet{n};
        }
        void remove(int v) {
            assert(vertices[v]);
            assert(elems > 0);
            vertices[v] = false;
            --elems;
        }
    public:
        bool is_empty() {
            for (auto v: vertices) {
                if (v) {
                    assert(elems > 0);
                    return false;
                }
            }
            assert(elems == 0);
            return true;
        }

        // static VertexSet full_row(std::vector<bool> const& row) {
        //     return VertexSet{row};
        // }

        static VertexSet full(int n) {
            // printf("VertexSet::full(%d)\n", n);
            VertexSet set{n};
            set.vertices.flip();
            set.elems = n;
            return set;
        }

        bool contains(int vertex) const {
            return vertices[vertex];
            // assert(std::is_sorted(vertices.cbegin(), vertices.cend()));
            // return std::binary_search(vertices.cbegin(), vertices.cend(), vertex);
        }

        VertexSet intersect_adjacent(InducedSubgraph const& subgraph, int vertex) const {
            if (debug) std::cerr << "Intersecting with neighbours set of vertex: " << vertex << std::endl;
            // if (vertex >= adjacency.row_ptr.size()) {
            //     return set; // empty
            // }
            VertexSet set{*this};
            if (debug) {
                std::cout << set << '\n';
            }
            // int const row_beg = adjacency.row_ptr[vertex];
            // int const row_end = adjacency.row_ptr[vertex + 1];
            // std::cerr << "row_beg: " << row_beg << ", row_end: " << row_end << std::endl;
            // for (auto it = adjacency.col_idx.begin() + row_beg; it < adjacency.col_idx.begin() + row_end; ++it) {
            //     int const neighbour = *it;
            //     // std::cerr << "neighbour: " << neighbour << std::endl;
            //     if (contains(neighbour))
            //         set.vertices.push_back(neighbour);
            // }
            auto const& row = subgraph.adjacency_matrix[vertex];
            // std::cout << "row size: " << row.size() << "\n";
            for (int i = 0; i < row.size(); ++i) {
                if (set.contains(i) && !row[i]) {
                    set.remove(i);
                }
            }
            if (debug && set.elems > 0) std::cerr << "Returning VertexSet with len=" << set.elems << "\n";
            return set;
        }
    };

    std::ostream& operator<<(std::ostream &os, VertexSet const& vertex_set) {
        os << "VertexSet[ ";
        for (int i = 0; i < vertex_set.vertices.size(); ++i) {
            if (vertex_set.vertices[i]) {
                os << i << " ";
            }
        }
        os << "]";
        return os;
    }

    struct StackEntry {
        VertexSet vertices;
        std::vector<int> clique_counter;
        int stack_vertex;
        int level;
        int k;
        StackEntry(VertexSet vertices, int k, int stack_vertex, int level)
        : /* chosen_vertex{vertex}, */ /* vertices{VertexSet::empty(static_cast<size_t>(graph.max_v))} */
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
        std::vector<InducedSubgraph> subgraphs;

        CPUAlgorithm(CSR edges, int k) : edges{edges}, stacks{NSTACKS}, k{k} {
            for (int v = 0; v <= edges.max_v; ++v) {
                auto subgraph = InducedSubgraph::extract(edges, v);
                if (debug) std::cout << "In relation to vertex " << v << ":\n" << subgraph << "\n";
                subgraphs.push_back(std::move(subgraph));
            }
        }

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
                auto& stack = stacks[v % NSTACKS];
                stack.emplace(VertexSet::full(subgraphs[v].mapping.size()), k, v, 1);
            }
            int i = 0;
            for (auto& stack: stacks) {
                // std::cerr << "\nSecond for stack " << i++ << "\n";
                while (!stack.is_empty()) {
                    auto entry = stack.pop();
                    if (debug) std::cerr << "Entry{level=" << entry.level << ", stack_vertex=" << entry.stack_vertex
                        <<   ", vertex set=" << entry.vertices << "}" << std::endl;
                    auto const& subgraph = subgraphs[entry.stack_vertex];
                    for (int v = 0; v < subgraph.mapping.size(); ++v) {
                        if (entry.vertices.contains(v)) {
                            // We've found a `level`-level clique.
                            ++count[entry.level];

                            // Let's explore deeper.
                            if (entry.level + 1 < k) {
                                auto new_vertices = entry.vertices
                                    .intersect_adjacent(subgraph, v);
                                if (!new_vertices.is_empty()) {
                                    stack.emplace(new_vertices, k, entry.stack_vertex, entry.level + 1);
                                }
                            }
                            // if (!new_vertices.is_empty()) {
                                // std::cerr << "There exists an edge from " << entry.chosen_vertex << " to " << v
                                //         << " on level " << entry.level << ".\n";
                            // }
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

    int max_v = 0;
    while (input_file.good() && !input_file.eof()) {
        std::getline(input_file, buffer);
        if (!buffer.empty()) {
            auto const edge = parse_edge(buffer);
            max_v = std::max({max_v, edge.first, edge.second});
            edges.push_back(edge);
        }
    }

    if (input_file.bad()) {
        std::cerr << "Error while reading from file!\n";
        return EXIT_FAILURE;
    }

    std::sort(edges.begin(), edges.end());
    if (debug) {
        std::cout << "unoriented sorted edges:\n";
        for (auto const [v1, v2]: edges) {
            std::cout << "(" << v1 << ", " << v2 << ")\n";
        }
        std::cout << "max_v=" << max_v << ")\n";
    }

    if (debug) { // debug
        CSR unoriented_graph{edges};
        std::cout << "unoriented graph:\n";
        std::cout << unoriented_graph << "\n";
    }

    auto degs = compute_degs(edges, max_v);
    orient_graph(edges, degs);
    std::sort(edges.begin(), edges.end());

    if (debug) {
        std::cout << "oriented sorted edges:\n";
        for (auto const [v1, v2]: edges) {
            std::cout << "(" << v1 << ", " << v2 << ")\n";
        }
    }

    CSR graph{edges};
    if (debug) {
        std::cout << "oriented graph:\n";
        std::cout << graph << "\n";
    }

    // {
    //     auto subgraph = InducedSubgraph::extract(graph, 0);
    //     std::cout << subgraph << "\n";
    // }
    // {
    //     auto subgraph = InducedSubgraph::extract(graph, 1);
    //     std::cout << subgraph << "\n";
    // }
    // {
    //     auto subgraph = InducedSubgraph::extract(graph, 4);
    //     std::cout << subgraph << "\n";
    // }
    // {
    //     auto subgraph = InducedSubgraph::extract(graph, 5);
    //     std::cout << subgraph << "\n";
    // }

    CPUAlgorithm algo{edges, k};
    auto count = algo.count_cliques();
    if (debug) std::cout << "count: [ ";
    auto it = count.cbegin();
    if (it != count.cend())
        output_file << *it;
    ++it;
    for (; it != count.cend(); ++it) {
        output_file << ' ' << *it;
        if (debug) std::cout << ' ' << *it;
    }
    if (debug) std::cout << " ]\n";


    return EXIT_SUCCESS;
}
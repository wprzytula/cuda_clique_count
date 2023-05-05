#include "cuda.h"
#include "common/errors.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <utility>
#include <charconv>
#include <algorithm>
#include <numeric>
#include <cassert>

#define PRINT

#ifdef PRINT
constexpr bool const debug = true;
#else
constexpr bool const debug = false;
#endif

#define MAX_K 12
#define MAX_DEG 1024
#define BLOCK_SIZE 256
#define NUM_BLOCKS 8

// dim3 grid(BLOCK_SIZE);

namespace cpu { namespace {

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
        std::vector<std::vector<int>> const adjacency_matrix;
private:
        InducedSubgraph(std::vector<int> mapping, std::vector<std::vector<int>> adjacency_matrix)
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
            std::vector<std::vector<int>> adjacency_matrix;

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
}} // namespace


struct CSR {
    int vs;
    int* row_ptr;
    int row_len;
    int* col_idx;
    int col_len;
};

struct InducedSubgraph {
    int len;
    int* mapping; // len
    int* adjacency_matrix; // len * len
};

__device__ void print_subgraph(InducedSubgraph const& subgraph) {
        printf("Subgraph mapping: [ ");
        for (int i = 0; i < subgraph.len; ++i) {
            int const old_v = subgraph.mapping[i];
            printf("%i ", old_v);
        }
        printf("]\n");
        printf("Adjacency matrix:\n");
        printf("  ");
        for (int i = 0; i < subgraph.len; ++i) {
            int const old_v = subgraph.mapping[i];
            printf("%i ", old_v);
        }
        printf("\n");
        for (int i = 0; i < subgraph.len; ++i) {
            printf("[");
            for (int j = 0; j < subgraph.len; ++j) {
                bool exists = subgraph.adjacency_matrix[i * subgraph.len + j];
                printf(" %c", exists ? 'x' : ' ');
            }
            printf(" ]\n");
        }
    }

struct Stack {
    int top;
    // VertexSet
    bool *vertices; // 2-level array [[true, true, false], [false, false, false]]
    int *elems; // how many elements are there in vertex set in i-th stack frame

    bool* done;

    int* level; // len: stack_entries_num
};

struct Data {
    int const k;
    int* next_vertex;
    Stack stacks[NUM_BLOCKS];
    CSR csr;
    InducedSubgraph* subgraphs;

    Data(cpu::CSR const& edges, int const k) : k{k} {
        csr.vs = edges.n;
        csr.row_len = edges.row_ptr.size();
        HANDLE_ERROR(cudaMalloc(&csr.row_ptr, edges.row_ptr.size() * sizeof(int)));
        csr.col_len = edges.col_idx.size();
        HANDLE_ERROR(cudaMalloc(&csr.col_idx, edges.n * sizeof(int)));

        HANDLE_ERROR(cudaMalloc(&subgraphs, edges.n * sizeof(InducedSubgraph)));

        auto tmp_subgraphs = std::make_unique<InducedSubgraph[]>(edges.n);
        for (int v = 0; v < edges.n; ++v) {
            auto cpu_subgraph = cpu::InducedSubgraph::extract(edges, v);
            printf("Filling subgraph with idx %i\n", v);
            auto& subgraph = tmp_subgraphs[v];
            subgraph.len = cpu_subgraph.mapping.size();
            if (cpu_subgraph.mapping.size() > 0) {
                HANDLE_ERROR(cudaMalloc(&subgraph.mapping, cpu_subgraph.mapping.size() * sizeof(int)));
                // Get back the output data
                HANDLE_ERROR(cudaMemcpy(subgraph.mapping,
                        cpu_subgraph.mapping.data(),
                        cpu_subgraph.mapping.size() * sizeof(int),
                        cudaMemcpyHostToDevice)
                );
                HANDLE_ERROR(cudaMalloc(&subgraph.adjacency_matrix,
                                cpu_subgraph.adjacency_matrix.size() * cpu_subgraph.adjacency_matrix[0].size() * sizeof(int)));
                for (int r = 0; r < cpu_subgraph.adjacency_matrix.size(); ++r) {
                    printf("cudaMemcpy(dst=%p, src=%p, count=%li)\n",
                            subgraph.adjacency_matrix + r * cpu_subgraph.adjacency_matrix[r].size(),
                            cpu_subgraph.adjacency_matrix[r].data(),
                            cpu_subgraph.adjacency_matrix[r].size()
                    );
                    HANDLE_ERROR(cudaMemcpy(subgraph.adjacency_matrix + r * cpu_subgraph.adjacency_matrix[r].size(),
                            cpu_subgraph.adjacency_matrix[r].data(),
                            cpu_subgraph.adjacency_matrix[r].size() * sizeof(int),
                            cudaMemcpyHostToDevice)
                    );
                }
            } else {
                subgraph.mapping = nullptr;
                subgraph.adjacency_matrix = nullptr;
            }
        }
        HANDLE_ERROR(cudaMemcpy(subgraphs,
                     tmp_subgraphs.get(),
                     edges.n * sizeof(InducedSubgraph),
                     cudaMemcpyHostToDevice)
        );

        // Initialise stacks
        int const max_entries = k * MAX_DEG;
        for (int i = 0; i < NUM_BLOCKS; ++i) {
            Stack& stack = stacks[i];
            HANDLE_ERROR(cudaMalloc(&stack.vertices, max_entries * MAX_DEG * sizeof(*stack.vertices)));
            HANDLE_ERROR(cudaMemset(stack.vertices, -1, MAX_DEG * sizeof(*stack.vertices))); // first stack entry

            HANDLE_ERROR(cudaMalloc(&stack.elems, max_entries * sizeof(*stack.elems)));
            // HANDLE_ERROR(cudaMemset(&stack.elems, 0, k * sizeof(int)));

            HANDLE_ERROR(cudaMalloc(&stack.level, max_entries * sizeof(*stack.level)));
            HANDLE_ERROR(cudaMalloc(&stack.done, max_entries * sizeof(*stack.done)));
        }

        // printf("Before cudaMalloc: next_vertex=%p\n", next_vertex);
        HANDLE_ERROR(cudaMalloc(&next_vertex, sizeof(*next_vertex)));
        printf("After cudaMalloc: next_vertex=%p\n", next_vertex);
        HANDLE_ERROR(cudaMemset(next_vertex, 0, sizeof(*next_vertex)));
    }
};

__device__ void intersect_adjacent(InducedSubgraph const& subgraph, bool const* vertex_set, int vertex, bool* out_vertex_set) {
        // if (debug) std::cerr << "Intersecting with neighbours set of vertex: " << vertex << std::endl;
        // if (vertex >= adjacency.row_ptr.size()) {
        //     return set; // empty
        // }
        // if (debug) {
            // std::cout << set << '\n';
        // }

        auto const* row = subgraph.adjacency_matrix + vertex * subgraph.len;

        for (int i = threadIdx.x; i < subgraph.len; i += blockDim.x) {
            printf("Thread %i: I'm intersecting %i-th vertex: vertex_set[%i]=true, row[%i]=false\n", threadIdx.x, i, i, i);
            out_vertex_set[i] = vertex_set[i] && row[i]; // set vertex as in or out of set
        }
        // if (debug && set.elems > 0) std::cerr << "Returning VertexSet with len=" << set.elems << "\n";
    }

__device__ bool vertex_set_nonempty(bool const* set) {
    __shared__ bool nonempty[BLOCK_SIZE];
    int const tid = threadIdx.x;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (tid < i) {
            // printf("Thread %i: reached reduction step i=%i\n", tid, i);
            nonempty[tid] |= nonempty[tid + i];
        }
        __syncthreads();
        i /= 2;
    }

    return nonempty[0];
}

__device__ int acquire_next_vertex(Data const& data) {
    int const thread_id = threadIdx.x;
    __shared__ int chosen_vertex;

    if (thread_id == 0)
        chosen_vertex = atomicAdd(data.next_vertex, 1);
    __syncthreads();
    return chosen_vertex;
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
__global__ void kernel(Data data, int *count) {
    int const block_id = blockIdx.x;
    int const thread_id = threadIdx.x;
    // int const warp_id = thread_id % 32;

    int chosen_vertex;

    Stack& stack = data.stacks[block_id];

    __shared__ int cliques[MAX_K];
    // Set counters to zeros.
    if (thread_id < MAX_K) {
        cliques[thread_id] = 0;
    }

    // debug
    if (thread_id == 0) {
        if (block_id == 0) {
            for (int i = 0; i < data.csr.vs; ++i) {
                print_subgraph(data.subgraphs[i]);
            }
        } else {
            clock_t start = clock();
            clock_t now;
            for (;;) {
                now = clock();
                clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
                if (cycles >= 1000000) {
                    break;
                }
            }
        }
    }
    __syncthreads();

    while ((chosen_vertex = acquire_next_vertex(data)) < data.csr.vs) {
        if (debug && thread_id == 0) {
            printf("Block %i has acquired vertex %i\n", block_id, chosen_vertex);
        }
        InducedSubgraph const& subgraph = data.subgraphs[chosen_vertex];

        // Initialise first stack frame.
        // stack.emplace(VertexSet::full(subgraphs[v].mapping.size()), k, v, 1);
        stack.top = 0;
        stack.level[0] = 1;
        stack.done[0] = false;

        while (stack.top >= 0) {
            int const current = stack.top;
            if (debug && thread_id == 0) {
                printf("Block %i vertex %i operating on stack entry with idx %i, done? %i\n",
                        block_id, chosen_vertex, current, stack.done[current]);
            }
            if (stack.done[current]) {
                if (thread_id == 0)
                    --stack.top;
                continue;
            }
            for (int v = 0; v < subgraph.len; ++v) {
                if (stack.vertices[MAX_DEG * current + v]) { // entry.vertices.contains(v)
                    // We've found a `level`-level clique.
                    if (thread_id == 0)
                        ++count[stack.level[current]];

                    // Let's explore deeper.
                    if (stack.level[current] + 1 < data.k) { // entry.level + 1 < k
                        if (thread_id == 0)
                            ++stack.top;
                        bool* new_vertices = stack.vertices + stack.top * MAX_DEG;
                        intersect_adjacent(subgraph, stack.vertices + current * MAX_DEG, v, new_vertices);

                        if (vertex_set_nonempty(new_vertices)) {
                            // stack.emplace(new_vertices, entry.level + 1);
                            if (thread_id == 0) {
                                stack.level[stack.top] = stack.level[current] + 1;
                                stack.done[stack.top] = false;
                            }
                        } else {
                            if (thread_id == 0)
                                --stack.top;
                        }
                    }
                }
            }
            if (thread_id == 0) {
                stack.done[current] = true;
                if (current == stack.top) /*leaf reached, go back*/{
                    printf("Vertex %i: Reached leaf in entry %i.\n", chosen_vertex, current);
                    --stack.top;
                }
            }
        }
    }

    if (thread_id < data.k) {
        atomicAdd(&count[thread_id], cliques[thread_id]);
    }
}

using cpu::Edge;
using cpu::parse_edge;
using cpu::compute_degs;
using cpu::orient_graph;

void count_cliques(std::vector<Edge>& edges, std::ofstream& output_file, int k, int max_v) {
    std::sort(edges.begin(), edges.end());
    if (debug) {
        std::cout << "unoriented sorted edges:\n";
        for (auto const [v1, v2]: edges) {
            std::cout << "(" << v1 << ", " << v2 << ")\n";
        }
        std::cout << "max_v=" << max_v << ")\n";
    }

    if (debug) { // debug
        cpu::CSR unoriented_graph{edges};
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

    cpu::CSR graph{edges};
    if (debug) {
        std::cout << "oriented graph:\n";
        std::cout << graph << "\n";
    }

    for (int v = 0; v <= max_v; ++v) {
        auto subgraph = cpu::InducedSubgraph::extract(graph, v);
        std::cout << subgraph << "\n";
    }

    // input data
    Data data{edges, k};

    // output data
    int *cliques_gpu, *cliques_cpu = new int[k];

    HANDLE_ERROR(cudaMalloc(&cliques_gpu, k * sizeof(int)));
    HANDLE_ERROR(cudaMemset(cliques_gpu, 0, k * sizeof(int)));

    cudaEvent_t kernel_run, stop;
    cudaEventCreate(&kernel_run);
    cudaEventCreate(&stop);

    cudaEventRecord(kernel_run, 0);

    // RUN KERNEL, RUN!
    kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(data, cliques_gpu);


    // Get back the output data
    HANDLE_ERROR(cudaMemcpy(cliques_cpu,
			   cliques_gpu,
			   k * sizeof(int),
			   cudaMemcpyDeviceToHost)
    );

    HANDLE_ERROR(cudaFree(cliques_gpu));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventDestroy(kernel_run);
    cudaEventDestroy(stop);

    if (debug) std::cout << "count: [ ";
    output_file << cliques_cpu[0];
    if (debug) std::cout << cliques_cpu[0];
    for (int i = 1; i < k; ++i) {
        output_file << ' ' << cliques_cpu[i];
        if (debug) std::cout << ' ' << cliques_cpu[i];
    }
    if (debug) std::cout << " ]\n";

    delete[] cliques_cpu;
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

    count_cliques(edges, output_file, k, max_v);

    return EXIT_SUCCESS;
}
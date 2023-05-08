#include "cuda.h"
#include "common/errors.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <vector>
#include <utility>
#include <charconv>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <unordered_map>

#ifdef PRINT
#define debug(x) x
#else
#define debug(x)
#endif

#define CEIL_DIV(x, y) ((x + y - 1) / y)
#define MAX_K 12
#define MAX_DEG 1024
#define MODULO 1'000'000'000;

#define BLOCK_SIZE 128
// #define BLOCK_SIZE 32
#define NUM_BLOCKS 128
// #define NUM_BLOCKS 1

#define WARP_SIZE 32
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define NUM_WARPS (NUM_BLOCKS * WARPS_PER_BLOCK)

namespace cpu { namespace {
    using Edge = std::pair<int, int>;

    // returns max_v
    int make_vertices_consecutive_natural_numbers(std::vector<Edge>& edges) {
        int next_num = 0;
        std::unordered_map<int, int> map;
        for (auto const [v1, v2]: edges) {
            if (map.find(v1) == map.cend()) {
                map[v1] = next_num++;
            }
            if (map.find(v2) == map.cend()) {
                map[v2] = next_num++;
            }
        }
        for (auto& edge: edges) {
            edge.first = map[edge.first];
            edge.second = map[edge.second];
        }
        return next_num - 1;
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

        CSR(std::vector<Edge> const& edges, int const max_v) : max_v{max_v}, n{max_v + 1} {
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

debug(
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
    })

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
}} // namespace


struct CSR {
    int vs;
    int* row_ptr;
    int row_len;
    int* col_idx;
    int col_len;
};

struct InducedSubgraph {
    int len_qwords;
    int vs;
    int mapping[MAX_DEG];
    unsigned long long adjacency_matrix[MAX_DEG * MAX_DEG / 64];

    __device__ void extract(CSR const& graph, int const v) {
        int const tid = threadIdx.x;
        int const start = graph.row_ptr[v];
        int const end = graph.row_ptr[v + 1];

        if (tid == 0) {
            vs = end - start;
            len_qwords = CEIL_DIV(vs, 64);
        }
        __syncthreads();

/* Build subgraph mapping: new_vertex [0..1024] -> old_vertex [0..|V|] */
        for (int j = tid; start + j < end; j += blockDim.x) {
            // put neighbours in mapping.
            int const neighbour = graph.col_idx[start + j];
            mapping[j] = neighbour;
        }

        __syncthreads();

/* Build adjacency matrix  */
        // It has k rows, where k = |induced subgraph vertices|
        auto const& mapping = this->mapping;
        auto old = [&mapping](int new_v){/* std::cout << "old(" << new_v << ")\n";  */return mapping[new_v];};
        auto neigh = [&graph](int col_i){return graph.col_idx[col_i];};

        // For each row
        for (int i = tid; i < vs; i += blockDim.x) {
            // Retrieve old id of the vertex
            int const old_v1 = mapping[i];

            // Operate on this row
            auto *const row = adjacency_matrix + i * vs;

            // Clear the row after previous subgraph (zeros are assumed in the algorithm)
            for (int j = 0; j < len_qwords; ++j) {
                row[j] = 0;
            }

            int csr_idx = graph.row_ptr[old_v1];
            int const csr_idx_end = graph.row_ptr[old_v1 + 1];

            // For each cell in this row
            for (int adj_idx = 0; adj_idx < vs; ++adj_idx) {
                // std::cout << "Incremented adj_idx to " << adj_idx << ", now points to " << old(adj_idx) << "\n";

                if (csr_idx >= csr_idx_end) {
                    // csr_idx went out of bounds.
                    goto end_row;
                }

                while (neigh(csr_idx) < old(adj_idx)) {
                    // std::cout << "Incremented csr_idx to " << csr_idx << "\n";
                    ++csr_idx;
                    if (csr_idx >= csr_idx_end) {
                        // csr_idx went out of bounds.
                        goto end_row;
                    }
                    // std::cout << "csr_idx now points to " << neigh(csr_idx) << "\n";
                }

                // printf("Deciding edge between %d and %d based on value in csr_idx under %d: %d\n",
                //      old_v1, old(adj_idx), csr_idx, neigh(csr_idx));
                row[adj_idx / 64] |= ((unsigned long long)(neigh(csr_idx) == old(adj_idx))) << (adj_idx % 64);
end_row:            ;
            }
        }
    }
};

__device__ void print_subgraph(InducedSubgraph const& subgraph) {
        printf("Subgraph mapping: [ ");
        for (int i = 0; i < subgraph.vs; ++i) {
            int const old_v = subgraph.mapping[i];
            printf("%i ", old_v);
        }
        printf("]\n");
        printf("Adjacency matrix:\n");
        printf("  ");
        for (int i = 0; i < subgraph.vs; ++i) {
            int const old_v = subgraph.mapping[i];
            printf("%i ", old_v);
        }
        printf("\n");
        for (int i = 0; i < subgraph.vs; ++i) {
            printf("[");
            for (int j = 0; j < subgraph.vs; ++j) {
                bool exists = subgraph.adjacency_matrix[i * subgraph.vs + j / 64] & (1ULL << j % 64);
                printf(" %c", exists ? 'x' : ' ');
            }
            printf(" ]\n");
        }
    }

struct Stack {
    // VertexSet
    unsigned long long *vertices; // 2-level array [[true, true, false], [false, false, false]]

    bool* done;

    int* level; // len: stack_entries_num
};

struct Data {
    int k;
    int* next_vertex;
    Stack stacks[NUM_WARPS];
    CSR csr;
    InducedSubgraph* subgraphs;

    void init(cpu::CSR const& edges, int const k) {
        this->k = k;
        csr.vs = edges.n;

        csr.row_len = edges.row_ptr.size();
        csr.row_ptr = nullptr;
        HANDLE_ERROR(cudaMalloc(&csr.row_ptr, edges.row_ptr.size() * sizeof(int)));
        HANDLE_ERROR(cudaMemcpy(
            csr.row_ptr,
            edges.row_ptr.data(),
            edges.row_ptr.size() * sizeof(int),
            cudaMemcpyHostToDevice)
        );

        csr.col_len = edges.col_idx.size();
        csr.col_idx = nullptr;
        HANDLE_ERROR(cudaMalloc(&csr.col_idx, edges.col_idx.size() * sizeof(int)));
        HANDLE_ERROR(cudaMemcpy(
            csr.col_idx,
            edges.col_idx.data(),
            edges.col_idx.size() * sizeof(int),
            cudaMemcpyHostToDevice)
        );

        subgraphs = nullptr;
        HANDLE_ERROR(cudaMalloc(&subgraphs, NUM_BLOCKS * sizeof(InducedSubgraph)));

        int const storage_units_per_vertex_set = MAX_DEG / 64;
        // Initialise stacks
        int const max_entries = k * MAX_DEG / WARPS_PER_BLOCK;
        for (int i = 0; i < NUM_WARPS; ++i) {
            Stack& stack = stacks[i];
            stack.vertices = nullptr;
            HANDLE_ERROR(cudaMalloc(&stack.vertices, max_entries * storage_units_per_vertex_set * sizeof(*stack.vertices)));
            HANDLE_ERROR(cudaMemset(stack.vertices, -1 /*1*/, storage_units_per_vertex_set * sizeof(*stack.vertices))); // first stack entry

            stack.level = nullptr;
            HANDLE_ERROR(cudaMalloc(&stack.level, max_entries * sizeof(*stack.level)));
            HANDLE_ERROR(cudaMalloc(&stack.done, max_entries * sizeof(*stack.done)));
        }

        next_vertex = nullptr;
        HANDLE_ERROR(cudaMalloc(&next_vertex, sizeof(*next_vertex)));
        HANDLE_ERROR(cudaMemset(next_vertex, 0, sizeof(*next_vertex)));
    }
};

__constant__ Data global_data;

// https://stackoverflow.com/a/3208376
#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  ((byte) & 0x80 ? '1' : '0'), \
  ((byte) & 0x40 ? '1' : '0'), \
  ((byte) & 0x20 ? '1' : '0'), \
  ((byte) & 0x10 ? '1' : '0'), \
  ((byte) & 0x08 ? '1' : '0'), \
  ((byte) & 0x04 ? '1' : '0'), \
  ((byte) & 0x02 ? '1' : '0'), \
  ((byte) & 0x01 ? '1' : '0')

#define QWORD_TO_BINARY_HIGHER(name, qword) \
printf(name ": Bytes 7, 6, 5, 4: " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN "\n",\
    BYTE_TO_BINARY(qword>>56), BYTE_TO_BINARY(qword>>48), BYTE_TO_BINARY(qword>>40), BYTE_TO_BINARY(qword>>32));

#define QWORD_TO_BINARY_LOWER(name, qword) \
printf(name ": Bytes 3, 2, 1, 0: " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN "\n",\
    BYTE_TO_BINARY(qword>>24), BYTE_TO_BINARY(qword>>16), BYTE_TO_BINARY(qword>>8), BYTE_TO_BINARY(qword));

__device__ void intersect_adjacent(InducedSubgraph const& subgraph, unsigned long long const* vertex_set, int vertex, unsigned long long* out_vertex_set) {
    auto const* row = subgraph.adjacency_matrix + vertex * subgraph.vs;

    for (int i = threadIdx.x % WARP_SIZE; i < subgraph.len_qwords; i += WARP_SIZE) {
        debug({
            printf("Block %i, Thread %i: I'm intersecting %i-th vertex slice: vertex_set[%i]=(%lli), row[%i] = (%llx)\n",
                blockIdx.x, threadIdx.x, i, i, vertex_set[i], i, row[i]);
            QWORD_TO_BINARY_HIGHER("Set", vertex_set[i]);
            QWORD_TO_BINARY_LOWER("Set", vertex_set[i]);
            QWORD_TO_BINARY_HIGHER("Row", row[i]);
            QWORD_TO_BINARY_LOWER("Row", row[i]);
        });
        out_vertex_set[i] = vertex_set[i] & row[i]; // set each vertex as in or out of set
    }
}

__device__ void copy_adjacent(InducedSubgraph const& subgraph, int vertex, unsigned long long* out_vertex_set) {
    auto const* row = subgraph.adjacency_matrix + vertex * subgraph.vs;

    for (int i = threadIdx.x; i < subgraph.len_qwords; i += WARP_SIZE) {
        debug({
            printf("Block %i, Thread %i: I'm copying %i-th vertex slice: row[%i] = (%llx)\n",
                blockIdx.x, threadIdx.x, i, i, row[i]);
            QWORD_TO_BINARY_HIGHER("Row", row[i]);
            QWORD_TO_BINARY_LOWER("Row", row[i]);
        });
        out_vertex_set[i] = row[i]; // set each vertex as in or out of row
    }
}

__device__ bool vertex_set_nonempty(unsigned long long const* set, int const vs) {
    int const tid = threadIdx.x % WARP_SIZE;
    int const warp_id = threadIdx.x / WARP_SIZE;

    int const len_qwords = CEIL_DIV(vs, 64);

    __shared__ bool nonempty[WARPS_PER_BLOCK];

    // for (int i = tid; i < len_qwords; i += WARP_SIZE) {
    //     if (i + 1 == len_qwords) { // if last, we have to only take into account the valid bits.

    //         // vs % 64
    //         // 0 -> 1 1 ... 1 1 1
    //         // 1 -> 0 0 ... 0 0 1
    //         // 2 -> 0 0 ... 0 1 1
    //         // ...
    //         // 63 -> 0 1 ... 1 1 1

    //         unsigned long long const mask = (-1ULL) >> ((64 - vs % 64) % 64);
    //         nonempty |= set[i] & mask;
    //     } else {
    //         nonempty |= set[i];
    //     }
    // }

    if (tid == 0) {
        for (int i = 0; i < len_qwords; ++i) {
            nonempty[warp_id] |= set[i];
        }
    }
    __syncwarp();

    debug(
        if (tid == 0) {
            printf("set_nonempty([ ");
            for (int i = 0; i < vs; ++i) {
                printf("%i ", !!(set[i / 64] & (1ULL << i % 64)));
            }
            printf("]) = %i\n", nonempty[warp_id]);
        }
    )

    return nonempty[warp_id];
}

__device__ int acquire_next_vertex(Data const& data) {
    int const thread_id = threadIdx.x;
    __shared__ int chosen_vertex;

    if (thread_id == 0) {
        chosen_vertex = atomicAdd(data.next_vertex, 1);
        // printf("Block %i: Acquired vertex %i.\n", blockIdx.x, chosen_vertex);
    }
    __syncthreads();
    return chosen_vertex;
}

__device__ bool vertex_set_contains(unsigned long long const* vertex_set, int const current_frame, int const v) {
    // if (debug && threadIdx.x == 0)
    //     printf("Thread %i: set: %p, current: %i, v: %i\n", threadIdx.x, vertex_set, current_frame, v);
    // __syncthreads();

    return vertex_set[MAX_DEG / 64 * current_frame + v / 64] & (1ULL << (v % 64));
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
__global__ void kernel(unsigned long long *count) {
    int const block_id = blockIdx.x;
    int const thread_id = threadIdx.x % WARP_SIZE;
    int const warp_id = threadIdx.x / WARP_SIZE;
    int const unique_warp_id = block_id * WARPS_PER_BLOCK + warp_id;

    int chosen_vertex;
    Data& data = global_data;
    int const all_vs = data.csr.vs;

    Stack& stack = data.stacks[unique_warp_id];
    __shared__ int stack_tops[WARPS_PER_BLOCK];
    int& stack_top = stack_tops[warp_id];

    debug(if (block_id == 0 && warp_id == 0 && thread_id == 0) printf("\n\n----- RUNNING KERNEL!!! ------\n\n"));

    __shared__ int cliques[WARPS_PER_BLOCK][MAX_K];
    // Set counters to zeros.
    if (thread_id < MAX_K) {
        cliques[warp_id][thread_id] = 0;
    }

    __syncthreads();

    while ((chosen_vertex = acquire_next_vertex(data)) < all_vs) {
        debug(if (thread_id == 0 && warp_id == 0) {
            printf("\n ACQUISITION: Block %i has acquired vertex %i\n", block_id, chosen_vertex);
        })

        // Compute InducedSubgraph
        {
            InducedSubgraph& subgraph = data.subgraphs[block_id];
            subgraph.extract(data.csr, chosen_vertex);
            debug(if (thread_id == 0 && warp_id == 0) print_subgraph(subgraph));
        }
        InducedSubgraph const& subgraph = data.subgraphs[block_id];
        int const vs = subgraph.vs;

        // Initialise empty stack.
        if (thread_id == 0) {
            stack_top = -1;
        }

        // First level
        __syncthreads();
        if (warp_id == 0) {
            // We've found a number=vs level 1 cliques.
            if (thread_id == 0) {
                debug(printf("Block %i vertex %i: found %i of 2-cliques.\n",
                             block_id, chosen_vertex, vs);
                );
                int* level_cliques = &cliques[0/*warp id*/][1 /*level*/];
                *level_cliques = (*level_cliques + vs) % MODULO;
            }
            if (2 < data.k) { // entry.level + 1 < k
                for (int v = 0; v < vs; ++v) {
                    int const assigned_warp = v % WARPS_PER_BLOCK;
                    int const unique_assigned_warp = block_id * WARPS_PER_BLOCK + assigned_warp;
                    int& assigned_stack_top = stack_tops[assigned_warp];
                    Stack& assigned_stack = data.stacks[unique_assigned_warp];
                    debug(if (thread_id == 0) printf("Block %i, Vertex %i, Assigned subgraph's vertex %i to warp %i.\n",
                                                        block_id, chosen_vertex, v, assigned_warp)
                    );
                    __syncwarp();

                    unsigned long long* new_vertices = assigned_stack.vertices + (assigned_stack_top + 1) * MAX_DEG / 64;
                    debug(if (thread_id == 0) printf("Block %i, Vertex %i, Warp %i: Copying subgraph's vertex %i.\n",
                                                        block_id, chosen_vertex, warp_id, v)
                    );
                    copy_adjacent(subgraph, v, new_vertices);

                    __syncwarp();

                    if (vertex_set_nonempty(new_vertices, vs)) {
                        // stack.emplace(new_vertices, entry.level + 1);
                        if (thread_id == 0) {
                            ++assigned_stack_top;
                            assigned_stack.level[assigned_stack_top] = 2;
                            assigned_stack.done[assigned_stack_top] = false;
                        }
                    }
                    __syncwarp();
                }
            }
        }

        __syncthreads();
        // Per-warp iteration
        debug(if (warp_id == 0 && thread_id == 0)
            printf("\n BEGINNING PER_WARP iteration: block %i, vertex %i.\n\n", block_id, chosen_vertex)
        );

        while (stack_top >= 0) {
            __syncwarp();
            int const current = stack_top;
            debug(if (thread_id == 0) printf("Warp %i: Stack top: %i\n", warp_id, current));
            debug(if (thread_id == 0) {
                printf("Block %i warp %i vertex %i operating on stack entry with idx %i, done? %i\n",
                        block_id, warp_id, chosen_vertex, current, stack.done[current]);
            })
            if (stack.done[current]) {
                if (thread_id == 0)
                    --stack_top;
                __syncwarp();
                continue;
            }
            for (int v = 0; v < vs; ++v) {
                __syncwarp();
                if (vertex_set_contains(stack.vertices, current, v)) { // entry.vertices.contains(v)
                    // We've found a `level`-level clique.
                    if (thread_id == 0) {
                        int* level_cliques = &cliques[warp_id][stack.level[current]];
                        *level_cliques = (*level_cliques + 1) % MODULO;
                        debug(
                            printf("Block %i, warp %i, vertex %i: found a %i-clique. Cliques now: %i\n",
                                      block_id, warp_id, chosen_vertex, stack.level[current] + 1, *level_cliques)
                        );
                    }

                    // Let's explore deeper.
                    if (stack.level[current] + 1 < data.k) { // entry.level + 1 < k
                        unsigned long long* new_vertices = stack.vertices + (stack_top + 1) * MAX_DEG / 64;
                        debug(if (thread_id == 0) printf("Block %i, Vertex %i, Warp %i: Intersecting with subgraph's vertex %i.\n",
                                                         block_id, chosen_vertex, warp_id, v)
                        );
                        intersect_adjacent(subgraph, stack.vertices + current * MAX_DEG / 64, v, new_vertices);

                        __syncwarp();

                        if (vertex_set_nonempty(new_vertices, vs)) {
                            // stack.emplace(new_vertices, entry.level + 1);
                            if (thread_id == 0) {
                                ++stack_top;
                                stack.level[stack_top] = stack.level[current] + 1;
                                stack.done[stack_top] = false;
                            }
                        }
                    }
                }
            }
            __syncwarp();

            if (thread_id == 0) {
                stack.done[current] = true;
                if (current == stack_top) /*leaf reached, go back*/{
                    debug(printf("Vertex %i warp %i: Reached leaf in entry %i.\n", chosen_vertex, warp_id, current));
                    --stack_top;
                } else {
                    debug(printf("Vertex %i warp %i: Finished work over node in entry %i.\n", chosen_vertex, warp_id, current));
                }
            }

            __syncwarp();
        }
        debug(if (thread_id == 0) {
            printf("Block %i, Vertex %i, warp %i: Finished stack iteration.\n", block_id, chosen_vertex, warp_id);
        });
    }

    __syncthreads();

    // if (thread_id == 0) { // DEBUG@@@R#@@#@T@@$T$@T@
    //     debug(printf("block %i warp %i: 2-Cliques num=%i\n", block_id, warp_id, cliques[0][1]));
    // }

    // if (warp_id != 0) {
    //     if (thread_id < data.k) {
    //         atomicAdd_block(&cliques[0][thread_id], cliques[warp_id][thread_id]);
    //     }
    // }

    __syncthreads();

    if (thread_id < data.k) {
        atomicAdd(&count[thread_id], (unsigned long long)cliques[warp_id][thread_id]);
    }
    // for (int i = thread_id; i < data.k; i += blockDim.x) {
    //     atomicAdd(&count[i], (unsigned long long)cliques[warp_id][i]);
    // }

    debug(
        __syncthreads();
        if (thread_id == 0 && warp_id == 0) printf("Block %i, Finished!\n", block_id);
    );
}

static void count_cliques(std::vector<cpu::Edge>& edges, std::ofstream& output_file, int k) {
    debug({
        std::cout << "unoriented sorted edges before making vertices consecutive:\n";
        for (auto const [v1, v2]: edges) {
            std::cout << "(" << v1 << ", " << v2 << ")\n";
        }
    })

    int const max_v = cpu::make_vertices_consecutive_natural_numbers(edges);

    debug({
        std::cout << "unoriented sorted edges with vertices made consecutive:\n";
        for (auto const [v1, v2]: edges) {
            std::cout << "(" << v1 << ", " << v2 << ")\n";
        }
        std::cout << "max_v=" << max_v << ")\n";
    });

    std::sort(edges.begin(), edges.end());
    debug({
        cpu::CSR unoriented_graph(edges, max_v);
        std::cout << "unoriented graph:\n";
        std::cout << unoriented_graph << "\n";
    });

    auto degs = cpu::compute_degs(edges, max_v);
    cpu::orient_graph(edges, degs);
    std::sort(edges.begin(), edges.end());

    debug({
        std::cout << "oriented sorted edges:\n";
        for (auto const [v1, v2]: edges) {
            std::cout << "(" << v1 << ", " << v2 << ")\n";
        }
    });

    cpu::CSR graph{edges, max_v};
    debug({
        std::cout << "oriented graph:\n";
        std::cout << graph << "\n";
    });

    auto cliques_cpu = std::make_unique<unsigned long long[]>(k);

    { // GPU section
        // input data
        Data data;
        data.init(graph, k);

        cudaMemcpyToSymbol(global_data, &data, sizeof(Data), 0, cudaMemcpyHostToDevice);

        // output data
        unsigned long long *cliques_gpu;

        HANDLE_ERROR(cudaMalloc(&cliques_gpu, k * sizeof(*cliques_gpu)));
        HANDLE_ERROR(cudaMemset(cliques_gpu, 0, k * sizeof(*cliques_gpu)));

        cudaEvent_t kernel_run, stop;
        cudaEventCreate(&kernel_run);
        cudaEventCreate(&stop);

        cudaEventRecord(kernel_run, 0);

        // RUN KERNEL, RUN!
        kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(cliques_gpu);


        // Get back the output data
        HANDLE_ERROR(cudaMemcpy(cliques_cpu.get(),
                cliques_gpu,
                k * sizeof(*cliques_gpu),
                cudaMemcpyDeviceToHost)
        );

        HANDLE_ERROR(cudaFree(cliques_gpu));

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float elapsed_kernel;
        HANDLE_ERROR(cudaEventElapsedTime(&elapsed_kernel, kernel_run, stop));
        printf("Elapsed kernel: %.3fms\n", elapsed_kernel);

        cudaEventDestroy(kernel_run);
        cudaEventDestroy(stop);
    }


    cliques_cpu[0] = max_v + 1;
    for (int i = 1; i < k; ++i) {
        cliques_cpu[i] = cliques_cpu[i] % MODULO;
    }

    std::stringstream s;

    s << "count: [ ";
    output_file << cliques_cpu[0];
    s << cliques_cpu[0];
    for (int i = 1; i < k; ++i) {
        output_file << ' ' << cliques_cpu[i];
        s << ' ' << cliques_cpu[i];
    }
    s << " ]\n";
    // if (debug)
    std::cout << s.str();
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

    std::vector<cpu::Edge> edges;
    std::string buffer;

    while (input_file.good() && !input_file.eof()) {
        std::getline(input_file, buffer);
        if (!buffer.empty()) {
            auto const edge = cpu::parse_edge(buffer);
            edges.push_back(edge);
        }
    }

    if (input_file.bad()) {
        std::cerr << "Error while reading from file!\n";
        return EXIT_FAILURE;
    }

    count_cliques(edges, output_file, k);

    return EXIT_SUCCESS;
}
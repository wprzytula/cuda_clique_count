#include "cuda.h"
#include "common/errors.h"
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

    int max_vertex(std::vector<Edge> const& edges) {
        int max_vertex = 0;
        for (auto const [v1, v2]: edges) {
            if (v1 > max_vertex) {
                max_vertex = v1;
            }
            if (v2 > max_vertex) {
                max_vertex = v2;
            }
        }
        return max_vertex;
    }

    struct CSR {
        std::vector<int> col_idx;
        std::vector<int> row_ptr;

        CSR(std::vector<Edge> const& edges) {
            col_idx.resize(edges.size());
            int max_v = max_vertex(edges);
            row_ptr.resize(max_v);

            int col_i = 0;
            for (int row = 0; row < max_v; ++row) {
                row_ptr[row] = col_i;
                while (edges[col_i].first == row) {
                    col_idx[col_i] = edges[col_i].second;
                    ++col_i;
                }
            }
            row_ptr.push_back(col_idx.size());
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

    CSR csr{edges};
    std::cout << csr << "\n";


    return EXIT_SUCCESS;
}
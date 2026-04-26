#include "inferedge_runtime/cli.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        const inferedge_runtime::RuntimeConfig config = inferedge_runtime::parse_args(argc, argv);

        if (config.show_help) {
            inferedge_runtime::print_help();
            return 0;
        }

        if (config.show_version) {
            inferedge_runtime::print_version();
            return 0;
        }

        return inferedge_runtime::run_cli(config);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

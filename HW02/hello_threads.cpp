#include <iostream>
#include <thread>
#include <vector>

void hello(int id, int total) {
    std::cout << "Hello from thread " << id << " of " << total << "\n";
}

int main() {

    const int N = 5;
    std::vector<std::thread> threads; // vector full of threads
    threads.reserve(N); // reserving enough space for all the threads in memory

    for (int i = 0; i < N; i++)
        threads.emplace_back(hello, i , N);

    for (auto &t : threads) t.join();
    return 0;
}
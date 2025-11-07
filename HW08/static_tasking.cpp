#include <taskflow/taskflow.hpp>

int main() {
    tf::Executor executor;
    tf::Taskflow taskflow("Static Taskflow Demo");

    auto A = taskflow.emplace([](){ printf("Task A\n"); });
    auto B = taskflow.emplace([](){ printf("Task B\n"); });
    auto C = taskflow.emplace([](){ printf("Task C\n"); });
    auto D = taskflow.emplace([](){ printf("Task D\n"); });

    A.precede(B, C);
    B.precede(D);
    C.precede(D);

    executor.run(taskflow).wait();
    
}
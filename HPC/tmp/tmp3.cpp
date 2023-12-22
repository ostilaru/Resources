#include<iostream>
using namespace std;

class CPU {
public:
    CPU(){cout << "构造了CPU" << endl;}
    CPU(CPU &cpu) {
        cpu_rank = cpu.cpu_rank;
        cout << "拷贝构造了CPU" << endl;
    }
    ~CPU(){cout << "析构了CPU" << endl;}
    int cpu_rank;
};

class Computer {
public:
    Computer(CPU _cpu, int _disk_size, int _memory_size) {
        cpu = _cpu; // 1
        disk_size = _disk_size;
        memory_size = _memory_size;
        cout << "构造了Computer" << endl;
    }
    ~Computer() {
        cout << "析构了Computer" << endl;
    }

private:
    CPU cpu; // 2
    int disk_size;
    int memory_size;
};

int main() {
    CPU cpu; // 3
    Computer computer(cpu, 100, 100);
    return 0;
}
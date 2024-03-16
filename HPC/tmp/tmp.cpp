#include <string.h>
#include <iostream>
using namespace std;

class Mammal {
public:
    Mammal(int h = 0) : h(h) {}
    virtual void print() {
        cout << h;
    }
    virtual int getH() { return h; }
private:
    int h;
};

class Human : public Mammal {
public:
    Human(int h = 0) : h(h), Mammal(h/2) {}
    virtual void print() {
        Mammal* mthis = this;
        cout << mthis->getH();
        Mammal::print();
    }
    virtual int getH() { return h; }
private:
    int h;
};

int main() {
    Human man(2);
    Mammal* mmal = &man;
    mmal->print();
    man.print();
    return 0;
}

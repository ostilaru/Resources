#include <string.h>
#include <iostream>
using namespace std;

struct Base1 { int x; };
struct Base2 { float y; };
struct Derived : Base1, Base2 { };

int main() {
    Derived *pd = new Derived;
    pd->x = 1; pd->y = 2.0f;
    Derived *pv = pd;
    Base2 *pb2 = static_cast<Base2*>(pv);
    // Base2 *pb2 = pd;
    cout << pd->y << " " << pb2->y << endl;
    return 0;
}

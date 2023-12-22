#include <string.h>
#include <iostream>
using namespace std;

int main() {
    char myString[] = "Hello, World!";
    string myString2("Hello, World!");
    size_t length = strlen(myString);
    std::cout << "The length of the string is " << length << std::endl;
    cout << myString2.length() << endl;
    // 'length' 现在包含 "Hello, World!" 字符串的长度，不包括末尾的 '\0'
    return 0;
}

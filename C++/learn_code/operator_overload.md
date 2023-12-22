# 运算符重载

## 1. 基本概念
重载的运算符是具有特殊名字的函数：它们的名字由关键字`operator`和其后要定义的运算符号共同组成。和其他函数一样，重载的运算符也包含返回类型、参数列表和函数体。

重载运算符函数的参数数量与该运算符作用的运算对象数量一样多：一元运算符有一个参数，二元运算符有两个参数。

对于二元运算符来说，左侧运算对象传递给第一个参数，右侧运算对象传递给第二个参数。

除了重载的函数调用运算符`operator()`之外，其他重载运算符不能含有默认实参。以下是一个运算符重载函数调用运算符 `operator()` 带有默认参数的示例：
```cpp
#include <iostream>

class MyCallable {
public:
    // 重载函数调用运算符 ()，带有默认参数
    int operator()(int x, int y = 0) const {
        return x + y;
    }
};

int main() {
    MyCallable myFunction;

    // 使用重载的函数调用运算符
    std::cout << myFunction(5) << std::endl;      // 输出 5
    std::cout << myFunction(5, 3) << std::endl;   // 输出 8

    return 0;
}

```


如果一个运算符函数是成员函数，则它的第一个（左侧）运算对象绑定到隐式的`this`指针上，因此，成员运算符函数的（显示）参数数量比运算符的运算对象数少一个。例如，如果你有一个类 `MyClass` 并想要重载加法运算符 `+`:
```cpp
#include <iostream>

class MyClass {
public:
    int value;

    MyClass(int val) : value(val) {}

    // 重载加法运算符，此时成员运算符函数的（显示）参数数量比运算符的运算对象数少一个
    MyClass operator+(const MyClass& other) const {
        MyClass result(value + other.value);
        return result;
    }
};

int main() {
    MyClass obj1(10);
    MyClass obj2(20);

    // 使用重载的加法运算符
    MyClass result = obj1 + obj2;

    // 输出结果
    std::cout << "Result: " << result.value << std::endl;

    return 0;
}

```


对于一个运算符函数来说，它要么是类的成员，要么至少含有一个类类型的参数：
```cpp
// 错误：不能为 int 重定义内置的运算符
int operator+(int, int);
```
这一约定意味着当运算符作用于内置类型对象时，我们不能改变运算符的含义。

### 直接调用一个重载的运算符函数
```cpp
data1 + data2           // 普通调用
operator+(data1, data2) // 等价的函数调用
```

### 选择作为成员函数还是非成员函数
当我们定义重载运算符时，必须首先决定将其声明为类的成员函数还是声明为一个普通的非成员函数。
1. 赋值`=`，下标`[ ]`，函数调用`()`，成员访问箭头`->`，这些运算符必须是成员函数。
2. 复合赋值运算符`+=`，`-=`，`*=`，`/=`，`%=`，一般来说应该是成员函数，但不是必须的。
3. 改变对象状态的运算符或者与给定类型密切相关的运算符，如递增`++`，递减`--`和解引用`*`，通常应该是成员函数，但也不是必须的。
4. 具有对称性的运算符可能转换任意一端的运算对象，例如算术、相等性、关系和位运算符等，它们通常是非成员函数。
例如，我们能求一个`int`和`double`的和，因为它们中的任意一个都可以是左侧运算对象和右侧运算对象，所以加法是**对称的**。如果我们想要提供含有类对象是混合类型表达式，则运算符必须定义成非成员函数。
```cpp
string s = "world";
string t = s + "!"; // 正确：我们能够把一个const char*加到一个string对象中
string u = "hi" + s; // 如果+是string的成员，则产生错误
```
如果`operator+`是`string`类的成员函数，则上面的加法相当于`s.operator+("!")`和`"hi".operator+(s)`，显然`"hi"`的类型是`const char*`，这是一种内置类型，根本没有成员函数。

因为`string`将`+`定义成了普通的非成员函数，所以`"hi"+s`等价于`operator+("hi",s)`。和其他任何函数调用一样，每个实参都能被转化为形参类型。唯一的要求是至少有一个运算对象是类类型，并且两个运算对象都能准确无误地转换成`string`。


## 2. 
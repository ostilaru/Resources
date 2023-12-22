#include <iostream>
#include <memory>
using namespace std;

// 箭头运算符必须是类的成员
// 解引用运算符通常也是类的成员，但不是必须的

template <typename T>
class MySmartPointer {
public:
    // 构造函数
    MySmartPointer(T* ptr = nullptr);

    // 拷贝构造函数
    MySmartPointer(const MySmartPointer& rhs);

    // 赋值运算符重载
    MySmartPointer& operator=(const MySmartPointer& rhs);

    // 解引用运算符重载
    T& operator*() const;

    // 箭头运算符重载
    T* operator->() const;

    // 隐式转换为 bool 的运算符重载
    explicit operator bool() const;

    // 交换两个对象
    void swap(MySmartPointer& t);

    // 获取引用计数
    size_t use_count() const;

    // 析构函数
    ~MySmartPointer();

    // 声明友元函数
    template <typename U>
    friend ostream& operator<<(ostream& os, const MySmartPointer<U>& sp);

private:
    T* ptr_;
    size_t* pcount_;
};

// 构造函数
template <typename T>
MySmartPointer<T>::MySmartPointer(T* ptr) : ptr_(ptr), pcount_(new size_t(0)) {
    cout << __FUNCTION__ << endl;
    if (ptr_) {
        *pcount_ = 1;
    }
}

// 拷贝构造函数
template <typename T>
MySmartPointer<T>::MySmartPointer(const MySmartPointer& rhs) : ptr_(rhs.ptr_), pcount_(rhs.pcount_) {
    cout << __FUNCTION__ << endl;
    if (ptr_) {
        ++*pcount_;
    }
}

// 赋值运算符重载
template <typename T>
MySmartPointer<T>& MySmartPointer<T>::operator=(const MySmartPointer& rhs) {
    cout << __FUNCTION__ << endl;
    MySmartPointer<T> temp(rhs);
    swap(temp);
    return *this;
}

// 解引用运算符重载
template <typename T>
T& MySmartPointer<T>::operator*() const {
    cout << __FUNCTION__ << endl;
    return *ptr_;
}

// 箭头运算符重载
template <typename T>
T* MySmartPointer<T>::operator->() const {
    cout << __FUNCTION__ << endl;
    return &operator*();
}

// 隐式转换为 bool 的运算符重载
template <typename T>
MySmartPointer<T>::operator bool() const {
    cout << __FUNCTION__ << endl;
    return !(ptr_ == nullptr && *pcount_ == 0);
}

// 交换两个对象
template <typename T>
void MySmartPointer<T>::swap(MySmartPointer& t) {
    cout << __FUNCTION__ << endl;
    std::swap(ptr_, t.ptr_);
    std::swap(pcount_, t.pcount_);
}

// 获取引用计数
template <typename T>
size_t MySmartPointer<T>::use_count() const {
    cout << __FUNCTION__ << endl;
    return *pcount_;
}

// 析构函数
template <typename T>
MySmartPointer<T>::~MySmartPointer() {
    cout << __FUNCTION__ << endl;
    if (--*pcount_ == 0) {
        delete ptr_;
        delete pcount_;
    }
}

// 定义友元函数
template <typename T>
ostream& operator<<(ostream& os, const MySmartPointer<T>& sp) {
    cout << __FUNCTION__ << endl;
    if (sp) {
        os << "SmartPointer: " << *sp;
    } else {
        os << "SmartPointer: nullptr";
    }
    return os;
}

int main() {
    MySmartPointer<string> spstr1(new string("hello"));
    cout << spstr1 << endl;
    MySmartPointer<string> spstr2(spstr1);
    cout << spstr1.use_count() << endl;
    cout << spstr2.use_count() << endl;
    cout << "------------------------------------------------------------------------" << endl;
    MySmartPointer<string> spstr3(new string("world"));
    spstr3 = spstr1;
    spstr1 = spstr1;
    cout << spstr3.use_count() << endl;
    cout << spstr1.use_count() << endl;
    cout << spstr2.use_count() << endl;
    cout << spstr1->size() << endl;
    if (spstr1) {
        cout << "spstr1 is not nullptr" << endl;
    }

    return 0;
}

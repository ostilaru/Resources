#include<vector>
#include<algorithm>
#include<random>
#include<iostream>
using namespace std;

class Solution {
public:
    vector<int> sortArray(vector<int>& nums) {
        QuickSort(nums, 0, nums.size() - 1);
        return nums;
    }

    void QuickSort(vector<int>& arr, int start, int end){
        if(start >= end) return;
        int index = rand()%(end-start+1) + start; // 随机产生一个下标
        int pivot = arr[index];   // 把它作为哨兵
        swap(arr[start], arr[index]); // 注意：先把它放在最前面
        
        // 循环不变量：这里是左闭右闭区间
        // 小于nums[pivot]区间：[left + 1, less]
        // 等于nums[pivot]区间：[less + 1, i]
        // 大于nums[pivot]区间：[more, right]
        int i=start, j=end;
        while(i<j){
            // 注意：先移动右边j，这样while结束后i处是比pivot小的数字，才可以与pivot交换
            while(i<j && arr[j]>=pivot) // 注意：大于等于的放右边
                j--;
            while(i<j && arr[i]<=pivot) // 小于等于的放左边
                i++;
            if(i!=j) // 最多i==j
                swap(arr[i], arr[j]);
        }
        swap(arr[start], arr[j]); // 再把start处的pivot放在中间
        QuickSort(arr, start, i-1);
        QuickSort(arr, i+1, end); // 注意：i已经排过序
        return ;
    }
};

int main() {
    std::vector<int> arr = {1, 3, 2, 4};
    // arr.push_back(1);
    // arr.push_back(3);
    // arr.push_back(2);
    // arr.push_back(4);
    Solution s;
    s.QuickSort(arr, 0, arr.size() - 1);

    for(int i=0; i<arr.size(); i++) {
        cout << arr[i] << " ";
    }

    return 0;
}
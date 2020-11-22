package leetcode;

import java.util.Arrays;

public class LeetCode {
    public static void main(String[] args) {
        Solution solution = new Solution();
        System.out.println(Arrays.toString(solution.twoSum(new int[]{3, 3}, 6)));
    }
}

class Solution {
    public int[] twoSum(int[] nums, int target) {
        int[] res = new int[2];
        for (int i = 0; i < nums.length; i++) {
            int remainValue = target - nums[i];
            for (int j = 0; j < nums.length; j++) {
                if (nums[j] == remainValue && i != j) {
                    res[0] = i;
                    res[1] = j;
                }
            }
        }
        return res;
    }
}
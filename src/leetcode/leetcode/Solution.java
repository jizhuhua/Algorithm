package leetcode;

import java.util.*;

public class Solution {

    //1两数之和
    public int[] twoSum(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return new int[]{-1, -1};
        }
        int[] result = new int[2];
        Map<Integer, Integer> targetIndex = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (!targetIndex.containsKey(nums[i])) {
                targetIndex.put(target - nums[i], i);
            } else {
                result[0] = targetIndex.get(nums[i]);
                result[1] = i;
            }
        }
        return result;
    }

    //2两数相加
    public class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode l3 = new ListNode(0);
        ListNode result = l3;
        int sum, val1, val2, sumWithoutCarry;
        int carry = 0;
        while (l1 != null || l2 != null) {
            val1 = l1 == null ? 0 : l1.val;
            val2 = l2 == null ? 0 : l2.val;
            sum = val1 + val2 + carry;
            sumWithoutCarry = sum % 10;
            carry = sum / 10;
            l3.next = new ListNode(sumWithoutCarry);
            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }
            l3 = l3.next;
        }
        if (carry != 0) {
            l3.next = new ListNode(carry);
        }
        return result;
    }

    //3无重复字符的最长子串
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        char[] chars = s.toCharArray();
        int start = 0;
        int ans = 0;
        Map<Character, Integer> charIndex = new HashMap<>();
        for (int i = 0; i < chars.length; i++) {
            if (charIndex.containsKey(chars[i])) {
                start = Math.max(start, charIndex.get(chars[i]) + 1);//"tmmzuxt"t指向0，m指向2，start=2，t的下一个为1小于start
                charIndex.put(chars[i], i);
            } else {
                charIndex.put(chars[i], i);
            }
            ans = Math.max(ans, i - start + 1);
        }
        return ans;
    }


    //最大子序和
    public int maxSubArray(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        int ans = nums[0];
        int sum = 0;
        for (int num : nums) {
            if (sum < 0) {
                sum = num;
            } else {
                sum += num;
            }
            ans = Math.max(sum, ans);
        }
        return ans;
    }

    //617合并二叉树
    public class TreeNode {
        int val;
        public TreeNode left;
        public TreeNode right;

        public TreeNode(int x) {
            val = x;
        }
    }

    //二叉树先序遍历:根左右
    public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
        //递归终止条件
        if (t1 == null) {
            return t2;
        }
        if (t2 == null) {
            return t1;
        }
        TreeNode newNode = new TreeNode(t1.val + t2.val);
        newNode.left = mergeTrees(t1.left, t2.left);
        newNode.right = mergeTrees(t1.right, t2.right);
        return newNode;
    }

    //461汉明距离
    public int hammingDistance(int x, int y) {
        //异或求值，对值求二进制1的个数
        return Integer.bitCount(x ^ y);
    }

    //226翻转二叉树
    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode tmp = root.left;
        root.left = root.right;
        root.right = tmp;
        invertTree(root.left);
        invertTree(root.right);
        return root;
    }

    //104二叉树最大深度
    public int maxDepth(TreeNode root) {
        //同样拿最小树举例
        if (root == null) {
            return 0;
        }
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    //206翻转链表
    public ListNode reverseList(ListNode head) {
        ListNode pre = null;
        ListNode cur = head;
        while (cur != null) {
            ListNode tmpCurNext = cur.next;
            cur.next = pre;
            pre = cur;
            cur = tmpCurNext;
        }
        return pre;
    }

    //136 只出现一次的数
    public int singleNumber(int[] nums) {
        //自己异或自己等于0
        int ans = nums[0];
        for (int i = 1; i < nums.length; i++) {
            ans = ans ^ nums[i];
        }
        return ans;
    }

    //169 多数元素
    public int majorityElement(int[] nums) {
        Arrays.sort(nums);
        return nums[nums.length / 2];
    }

    //283 移动零
    public void moveZeroes(int[] nums) {
        int j = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                int tmp = nums[i];
                nums[i] = nums[j];
                nums[j++] = tmp;
            }
        }
    }

    //538 把二叉搜索树转累加树
    int sum = 0;

    public TreeNode convertBST(TreeNode root) {
        //二叉搜索 左根右 累加右根左
        if (root != null) {
            convertBST(root.right);
            root.val += sum;
            sum = root.val;
            convertBST(root.left);
        }
        return root;//举最简单例子返回
    }

    //448 找到所有数组中消失的数字
    public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> ans = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            //用下标来记录遍历到的值
            int index = Math.abs(nums[i]) - 1;
            if (nums[index] > 0) {
                nums[index] *= -1;
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0) {
                ans.add(i + 1);
            }
        }
        return ans;
    }

    //437 路径总和
    public int pathSum(TreeNode root, int sum) {
        if (root == null) {
            return 0;
        }
        return helper(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum);
    }

    private int helper(TreeNode node, int sum) {
        if (node == null) {
            return 0;
        }
        sum -= node.val;
        return (sum == 0 ? 1 : 0) + helper(node.left, sum) + helper(node.right, sum);
    }

    //160 相交链表
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode pA = headA, pB = headB;
        while (pA != pB) {
            pA = pA == null ? headB : pA.next;
            pB = pB == null ? headA : pB.next;
        }
        return pA;
    }

    //121 买股票的最佳时机
    public int maxProfit(int[] prices) {
        int max = 0;
        int minPrice = Integer.MAX_VALUE;
        for (int price : prices) {
            minPrice = Math.min(minPrice, price);
            max = Math.max(max, price - minPrice);
        }
        return max;
    }

    //543 二叉树的直径
    int ans = 1;

    public int diameterOfBinaryTree(TreeNode root) {
        depth(root);
        return ans - 1;
    }

    private int depth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int L = depth(root.left);
        int R = depth(root.right);
        ans = Math.max(L + R + 1, ans);
        return Math.max(L, R) + 1;
    }

    //70
    public int climbStairs(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

    //198 大家劫舍
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[nums.length - 1];
        }
        //方案一：不包含最新一间
        int sum1 = nums[0];
        //方案二：包含最新一间
        int sum2 = nums[1];
        for (int i = 2; i < nums.length; i++) {
            int tmp = sum1;
            //方案一更新,取上把大的，依然不包含这把最新一间
            sum1 = Math.max(sum1, sum2);
            //方案二更新，取上一把没有包含新一间的加这一把要包含新一间
            sum2 = tmp + nums[i];
        }
        return Math.max(sum1, sum2);
    }

    //234回文链表
    public boolean isPalindrome(ListNode head) {
        if (head == null) {
            return true;
        }
        List<Integer> list = new ArrayList<>();
        while (head != null) {
            list.add(head.val);
            head = head.next;
        }
        for (int i = 0, j = list.size() - 1; i < list.size() / 2; i++, j--) {
            if (!list.get(i).equals(list.get(j))) {
                return false;
            }
        }
        return true;
    }

    //581 最短无序连续子数组
    public int findUnsortedSubarray(int[] nums) {
        int[] sortNums = nums.clone();
        Arrays.sort(sortNums);
        //start为找出第一个使序列不升序的坐标
        int start = nums.length - 1;
        //end为找出最后一个使序列不升序的坐标
        int end = 0;
        for (int i = 0; i < nums.length; i++) {
            if (sortNums[i] != nums[i]) {
                start = Math.min(i, start);
                end = Math.max(end, i);
            }
        }
        return end - start > 0 ? end - start + 1 : 0;
    }

    //21 合并两个有序链表
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode cur = new ListNode(0);
        ListNode ans = cur;
        while (l1 != null || l2 != null) {
            if (l1 == null) {
                cur.next = l2;
                break;
            }
            if (l2 == null) {
                cur.next = l1;
                break;
            }
            if (l1.val < l2.val) {
                cur.next = l1;
                l1 = l1.next;
            } else {
                cur.next = l2;
                l2 = l2.next;
            }
            cur = cur.next;
        }
        return ans.next;
    }

    //78 子集
    public List<List<Integer>> subsets(int[] nums) {
        
    }
}

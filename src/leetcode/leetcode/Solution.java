package leetcode;

import javafx.scene.transform.Rotate;

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
        List<List<Integer>> ans = new ArrayList<>();
        backtrack(ans, new ArrayList<Integer>(), 0, nums);
        return ans;
    }

    private void backtrack(List<List<Integer>> ans, ArrayList<Integer> tmp, int i, int[] nums) {
        ans.add(new ArrayList<>(tmp));//{1} {1 2} {1 2 3} //需remove
        for (int j = i; j < nums.length; j++) {
            tmp.add(nums[j]);
            backtrack(ans, tmp, j + 1, nums);
            tmp.remove(tmp.size() - 1);
        }
    }

    //46 全排列
    List<List<Integer>> res = new LinkedList<>();

    /* 主函数，输入一组不重复的数字，返回它们的全排列 */
    public List<List<Integer>> permute(int[] nums) {
        // 记录「路径」
        LinkedList<Integer> track = new LinkedList<>();
        backtrack(nums, track);
        return res;
    }

    // 路径：记录在 track 中
    // 选择列表：nums 中不存在于 track 的那些元素
    // 结束条件：nums 中的元素全都在 track 中出现
    void backtrack(int[] nums, LinkedList<Integer> track) {
        // 触发结束条件
        if (track.size() == nums.length) {
            res.add(new LinkedList(track));
            return;
        }

        for (int i = 0; i < nums.length; i++) {
            // 排除不合法的选择
            if (track.contains(nums[i]))
                continue;
            // 做选择
            track.add(nums[i]);
            // 进入下一层决策树
            backtrack(nums, track);
            // 取消选择([1,2，]取消2，回到for循环的下一个数生成[1,3,])
            track.removeLast();
        }
    }


    //338.比特位计数
    public int[] countBits(int num) {
        int[] res = new int[num + 1];
        for (int i = 0; i <= num; i++) {
            res[i] = Integer.bitCount(i);
        }
        return res;
    }

    //二叉树中序遍历
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        inorder(res, root);
        return res;
    }

    private void inorder(List<Integer> res, TreeNode root) {
        if (root == null) {
            return;
        }
        inorder(res, root.left);
        res.add(root.val);
        inorder(res, root.right);
    }


    //除自身以外数组的乘积
    public int[] productExceptSelf(int[] nums) {
        int len = nums.length;
        int[] ans = new int[len];
        int tmpL = 1;
        int tmpR = 1;
        for (int i = 0; i < len; i++) {
            tmpL = tmpR = 1;
            for (int j = 0; j < i; j++) {
                tmpL *= nums[j];
            }
            for (int j = len - 1; j > i; j--) {
                tmpR *= nums[j];
            }
            ans[i] = tmpL * tmpR;
        }
        return ans;
    }

    public void flatten(TreeNode root) {
        //根节点开始迭代
        while (root != null) {
            //左节点没有，根节点直接到下一个根
            if (root.left == null) {
                root = root.right;
            } else {
                //右节点放到左节点的最右
                TreeNode tmp = root.left;
                while (tmp.right != null) {
                    tmp = tmp.right;
                }
                tmp.right = root.right;
                //左节点放到根的右节点
                root.right = root.left;
                //根的左节点置null
                root.left = null;
                //根节点下移
                root = root.right;
            }

        }
    }

    public void rotate(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return;
        }
        int n = matrix.length - 1;
        int[][] res = new int[n + 1][n + 1];
        for (int j = 0; j <= n; j++) {
            for (int i = 0; i <= n; i++) {
                int matrix1 = matrix[i][j];
                res[j][n - i] = matrix1;
            }
        }
        for (int i = 0; i <= n; i++) {
            System.arraycopy(res[i], 0, matrix[i], 0, n + 1);
        }
    }


    public int numTrees(int n) {
        int[] res = new int[n + 1];
        res[0] = 1;
        res[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 0; j <= i - 1; j++) {
                res[i] = res[j] * res[i - 1 - j];
            }
        }
        return res[n];
    }

    //218 前缀树
    class TrieNode {

        // R links to node children
        private TrieNode[] links;

        private final int R = 26;

        private boolean isEnd;

        public TrieNode() {
            links = new TrieNode[R];
        }

        public boolean containsKey(char ch) {
            return links[ch - 'a'] != null;
        }

        public TrieNode get(char ch) {
            return links[ch - 'a'];
        }

        public void put(char ch, TrieNode node) {
            links[ch - 'a'] = node;
        }

        public void setEnd() {
            isEnd = true;
        }

        public boolean isEnd() {
            return isEnd;
        }
    }


    public class Trie {
        private TrieNode root;

        public Trie() {
            root = new TrieNode();
        }

        // Inserts a word into the trie.
        public void insert(String word) {
            TrieNode curNode = root;
            for (int i = 0; i < word.length(); i++) {
                char currentChar = word.charAt(i);
                if (!curNode.containsKey(currentChar)) {
                    curNode.put(currentChar, new TrieNode());
                }
                curNode = curNode.get(currentChar);
            }
            curNode.setEnd();
        }


        // search a prefix or whole key in trie and
        // returns the node where search ends
        private TrieNode searchPrefix(String word) {
            TrieNode node = root;
            for (int i = 0; i < word.length(); i++) {
                char curLetter = word.charAt(i);
                if (node.containsKey(curLetter)) {
                    node = node.get(curLetter);
                } else {
                    return null;
                }
            }
            return node;
        }

        // Returns if the word is in the trie.
        public boolean search(String word) {
            TrieNode node = searchPrefix(word);
            return node != null && node.isEnd();
        }

        public boolean startsWith(String word) {
            TrieNode node = searchPrefix(word);
            return node != null;
        }
    }

    //12. 整数转罗马数字
    public String intToRoman(int num) {
        String thoudsands[] = {"", "M", "MM", "MMM"};
        String hunbuns[] = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
        String tens[] = {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
        String ones[] = {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};
        return thoudsands[(num / 1000)] + hunbuns[(num % 1000) / 100] + tens[(num % 1000) % 100 / 10] + ones[num % 1000 % 100 % 10];
    }

    public int romanToInt(String s) {
        Map<String, Integer> map = new HashMap<>();
        {

            map.put("I", 1);
            map.put("V", 5);
            map.put("X", 10);
            map.put("L", 50);
            map.put("C", 100);
            map.put("D", 500);
            map.put("M", 1000);
            map.put("IV", 4);
            map.put("IX", 9);
            map.put("XL", 40);
            map.put("XC", 90);
            map.put("CD", 400);
            map.put("CM", 900);

        }
        int ans = 0;
        for (int i = 0; i < s.length(); ) {
            if (i + 1 < s.length() && map.containsKey(s.substring(i, i + 2))) {
                ans += map.get(s.substring(i, i + 2));//右区间取不到
                i += 2;
            } else {
                ans += map.get(s.substring(i, i + 1));
                i++;
            }
        }
        return ans;
    }

    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) {
            return "";
        }
        if (strs[0].equals("") || strs[0].length() == 0) {
            return "";
        }
        String prefix = strs[0];
        for (int i = 1; i < strs.length; i++) {
            prefix = shrink(prefix, strs[i]);
            if (prefix.equals("")) {
                return "";
            }
        }
        return prefix;
    }

    private String shrink(String prefix, String str) {
        int i = 0;
        int end = str.length() - 1;
        for (; i <= end && i <= prefix.length() - 1; i++) {
            if (prefix.charAt(i) != str.charAt(i)) {
                break;
            }
        }
        return prefix.substring(0, i);
    }

}
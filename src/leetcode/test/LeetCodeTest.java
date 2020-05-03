package test;

import leetcode.Solution;
import org.junit.Before;
import org.junit.Test;

public class LeetCodeTest {
    Solution solution;

    @Before
    public void setUp() {
        solution = new Solution();
    }

    @Test
    public void test1() {
        int ans = solution.maxSubArray(new int[]{-2, 1, -3, 4, -1, 2, 1, -5, 4});
        System.out.println(ans);
    }

    @Test
    public void test2() {
        Solution.TreeNode treeNode1 = solution.new TreeNode(0);
        treeNode1.left = solution.new TreeNode(1);
        treeNode1.right = solution.new TreeNode(2);
        Solution.TreeNode treeNode2 = solution.new TreeNode(0);
        treeNode2.left = solution.new TreeNode(2);
        treeNode2.right = solution.new TreeNode(3);
        solution.mergeTrees(treeNode1, treeNode2);
    }

    @Test
    public void test3() {
        solution.hammingDistance(2, 3);
    }

    @Test
    public void test4() {
        Solution.TreeNode treeNode1 = solution.new TreeNode(0);
        treeNode1.left = solution.new TreeNode(1);
        treeNode1.right = solution.new TreeNode(2);
        solution.invertTree(treeNode1);
    }

    @Test
    public void test5() {
        solution.moveZeroes(new int[]{1});
    }

    @Test
    public void test06() {
        System.out.println(1 ^ 4);
    }

}
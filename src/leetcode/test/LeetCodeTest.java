package test;

import leetcode.Solution;
import org.junit.Before;
import org.junit.Test;

import javax.sound.midi.Soundbank;

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

}

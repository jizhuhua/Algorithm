package leetcode;

import java.util.*;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.PriorityBlockingQueue;

public class LeetCode {
    public static void main(String[] args) {
        Solution solution = new Solution();
        ListNode list1 = new ListNode(2);
        ListNode listNode1 = list1;
        listNode1.next = new ListNode(4);
        listNode1 = listNode1.next;
        listNode1.next = new ListNode(3);
        ListNode list2 = new ListNode(5);
        ListNode listNode2 = list2;
        listNode2.next = new ListNode(6);
        listNode2 = listNode2.next;
        listNode2.next = new ListNode(4);
        solution.addTwoNumbers(list1, list2);
    }
}


class ListNode {
    int val;
    ListNode next;

    ListNode() {
    }

    ListNode(int val) {
        this.val = val;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}

class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        Queue<Integer> queue = new LinkedList<>();  //LinkedList实现queue 先入先出  priorityQueue会根据值来判断
        int carry = 0;
        while (l1 != null || l2 != null) {
            int val1 = l1 == null ? 0 : l1.val;
            int val2 = l2 == null ? 0 : l2.val;
            int sum = val1 + val2 + carry;
            queue.add(sum % 10);
            carry = sum / 10;
            if (l1 != null) {
                l1 = l1.next;   //理解链表指向：一个= 不是===
            }
            if (l2 != null) {
                l2 = l2.next;
            }
        }
        if (carry != 0) {
            queue.add(carry);
        }
        ListNode res = new ListNode();
        ListNode tmp = res;
        while (queue.size() != 0) {
            tmp.next = new ListNode(queue.poll());
            tmp = tmp.next;
        }
        return res.next;
    }
}
use std::{
    cell::RefCell,
    cmp::{max, min, Ordering},
    collections::{HashMap, HashSet, VecDeque},
    i32,
    iter::once,
    rc::Rc,
    usize,
};

// Definition for singly-linked list.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    fn new(val: i32) -> Self {
        ListNode { next: None, val }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: TNode,
    pub right: TNode,
}

impl TreeNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None,
        }
    }
}

type TNode = Option<Rc<RefCell<TreeNode>>>;

struct Solution;

#[allow(dead_code)]
impl Solution {
    // 130. Surrounded Regions
    pub fn solve130(board: &mut Vec<Vec<char>>) {
        let mut stack = Vec::new();
        let (m, n) = (board.len(), board[0].len());
        for (r, c) in (0..n)
            .map(|i| (0, i))
            .chain((0..n).map(|i| (m - 1, i)))
            .chain((0..m).map(|i| (i, 0)))
            .chain((0..m).map(|i| (i, n - 1)))
        {
            if board[r][c] == 'O' {
                stack.push((r, c));
                while let Some((r, c)) = stack.pop() {
                    if r < m && c < n && board[r][c] == 'O' {
                        board[r][c] = 'M';
                        for rc in [0, 1, 0, !0, 0].windows(2) {
                            stack.push((r.wrapping_add(rc[0]), c.wrapping_add(rc[1])));
                        }
                    }
                }
            }
        }
        for r in board.iter_mut() {
            for c in r.iter_mut() {
                *c = if c == &'M' { 'O' } else { 'X' }
            }
        }
    }

    // 200. Number of Islands
    pub fn num_islands(mut grid: Vec<Vec<char>>) -> i32 {
        fn dfs(grid: &mut Vec<Vec<char>>, i: usize, j: usize) {
            if i < grid.len() && j < grid[i].len() && grid[i][j] == '1' {
                grid[i][j] = '0';
                for rc in [0, 1, 0, !0, 0].windows(2) {
                    dfs(grid, i.wrapping_add(rc[0]), j.wrapping_add(rc[1]));
                }
            }
        }
        let mut count = 0;
        for i in 0..grid.len() {
            for j in 0..grid[0].len() {
                if grid[i][j] == '1' {
                    count += 1;
                    dfs(&mut grid, i, j);
                }
            }
        }
        count
    }

    // 98. Validate Binary Search Tree
    pub fn is_valid_bst(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        fn dfs(root: TNode, nums: &mut Vec<i64>) {
            if let Some(root) = root {
                let mut node = root.borrow_mut();
                dfs(node.left.take(), nums);
                nums.push(node.val as i64);
                dfs(node.right.take(), nums);
            }
        }
        let mut nums = Vec::new();
        dfs(root, &mut nums);
        if nums.len() <= 1 {
            return true;
        }
        nums.into_iter()
            .fold((true, i32::MIN as i64 - 1), |(valid, y), x| {
                (valid && x > y, x)
            })
            .0
    }

    // 230. Kth Smallest Element in a BST
    pub fn kth_smallest(root: Option<Rc<RefCell<TreeNode>>>, k: i32) -> i32 {
        fn dfs(root: TNode, nums: &mut Vec<i32>) {
            if let Some(root) = root {
                let mut node = root.borrow_mut();
                dfs(node.left.take(), nums);
                nums.push(node.val);
                dfs(node.right.take(), nums);
            }
        }
        let mut nums = Vec::new();
        dfs(root, &mut nums);
        nums[k as usize - 1]
    }

    // 530. Minimum Absolute Difference in BST
    pub fn get_minimum_difference(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        fn dfs(root: TNode, nums: &mut Vec<i32>) {
            if let Some(root) = root {
                let mut node = root.borrow_mut();
                dfs(node.left.take(), nums);
                nums.push(node.val);
                dfs(node.right.take(), nums);
            }
        }
        let mut nums = Vec::new();
        dfs(root, &mut nums);
        nums.into_iter()
            .fold((-100000, i32::MAX), |(last, m), x| (x, m.min(x - last)))
            .1
    }

    // 103. Binary Tree Zigzag Level Order Traversal
    pub fn zigzag_level_order(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
        let mut res: Vec<Vec<i32>> = Vec::new();
        let mut q: VecDeque<_> = once(root).flatten().collect();
        let mut positive = true;
        while !q.is_empty() {
            let mut nodes = Vec::with_capacity(q.len());
            for _ in 0..q.len() {
                let rc = q.pop_front().unwrap();
                let mut node = rc.borrow_mut();
                nodes.push(node.val);
                q.extend([node.left.take(), node.right.take()].into_iter().flatten());
            }
            res.push(if positive {
                nodes
            } else {
                nodes.reverse();
                nodes
            });
            positive = !positive;
        }
        res
    }

    // 102. Binary Tree Level Order Traversal
    // once
    pub fn level_order(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
        let mut res: Vec<Vec<i32>> = Vec::new();
        let mut q: VecDeque<_> = once(root).flatten().collect();
        while !q.is_empty() {
            let mut nodes = Vec::with_capacity(q.len());
            for _ in 0..q.len() {
                let rc = q.pop_front().unwrap();
                let mut node = rc.borrow_mut();
                nodes.push(node.val);
                q.extend([node.left.take(), node.right.take()].into_iter().flatten());
            }
            res.push(nodes);
        }
        res
    }

    // 637. Average of Levels in Binary Tree
    pub fn average_of_levels(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<f64> {
        let mut res: Vec<f64> = Vec::with_capacity(100);
        let mut queue: VecDeque<_> = once(root).flatten().collect();
        while !queue.is_empty() {
            let (mut sum, n) = (0.0, queue.len());
            for _ in 0..n {
                let rc = queue.pop_front().unwrap();
                let mut node = rc.borrow_mut();
                sum += node.val as f64;
                queue.extend([node.left.take(), node.right.take()].into_iter().flatten());
            }
            res.push(sum / n as f64);
        }
        res
    }

    // 199. Binary Tree Right Side View
    pub fn right_side_view(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        let mut res = vec![0; 1 << 8];
        fn dfs(root: &TNode, res: &mut Vec<i32>, index: usize) {
            if let Some(root) = root {
                let node = root.borrow();
                if index == res.len() {
                    res.push(node.val);
                }
                dfs(&node.right, res, index + 1);
                dfs(&node.left, res, index + 1);
            }
        }
        dfs(&root, &mut res, 0);
        res
    }

    // 236. Lowest Common Ancestor of a Binary Tree
    // once
    pub fn lowest_common_ancestor(
        root: Option<Rc<RefCell<TreeNode>>>,
        p: Option<Rc<RefCell<TreeNode>>>,
        q: Option<Rc<RefCell<TreeNode>>>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        let mut stack_p = Vec::new();
        let mut stack_q = Vec::new();
        let p = p.unwrap().borrow().val;
        let q = q.unwrap().borrow().val;
        fn find(
            root: TNode,
            p: i32,
            q: i32,
            stack_p: &mut Vec<TNode>,
            stack_q: &mut Vec<TNode>,
            found_p: &mut bool,
            found_q: &mut bool,
        ) {
            if *found_p && *found_q {
                return;
            }
            if let Some(root) = root {
                let node = root.borrow();
                if !*found_p {
                    stack_p.push(Some(root.clone()));
                    if node.val == p {
                        *found_p = true;
                    }
                }
                if !*found_q {
                    stack_q.push(Some(root.clone()));
                    if node.val == q {
                        *found_q = true;
                    }
                }
                find(node.left.clone(), p, q, stack_p, stack_q, found_p, found_q);
                find(node.right.clone(), p, q, stack_p, stack_q, found_p, found_q);
                if !*found_p {
                    stack_p.pop();
                }
                if !*found_q {
                    stack_q.pop();
                }
            }
        }
        find(
            root,
            p,
            q,
            &mut stack_p,
            &mut stack_q,
            &mut false,
            &mut false,
        );
        stack_p
            .into_iter()
            .zip(stack_q)
            .filter(|(p, q)| p == q)
            .last()
            .map(|(p, _)| p)
            .unwrap()
    }

    // 124. Binary Tree Maximum Path Sum
    pub fn max_path_sum(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        fn max_path(root: &TNode, max_sum: &mut i32) -> i32 {
            match root {
                Some(root) => {
                    let node = root.borrow();
                    let val = node.val;
                    let left = max_path(&node.left, max_sum).max(0);
                    let right = max_path(&node.right, max_sum).max(0);
                    *max_sum = (left + right + val).max(*max_sum);
                    val + max(left, right)
                }
                None => 0,
            }
        }
        let mut res = root.as_ref().unwrap().borrow().val;
        max_path(&root, &mut res);
        res
    }

    // 129. Sum Root to Leaf Numbers
    pub fn sum_numbers(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        fn sum(root: TNode, num: i32) -> i32 {
            match root {
                Some(root) => {
                    let val = root.borrow().val + num * 10;
                    match (root.borrow().left.clone(), root.borrow().right.clone()) {
                        (None, None) => val,
                        (left, right) => sum(left, val) + sum(right, val),
                    }
                }
                None => 0,
            }
        }
        sum(root, 0)
    }

    // 112. Path Sum
    pub fn has_path_sum(root: Option<Rc<RefCell<TreeNode>>>, target_sum: i32) -> bool {
        if let Some(root) = root {
            let target = target_sum - root.borrow().val;
            let left = root.borrow().left.clone();
            let right = root.borrow().right.clone();
            if target == 0 && left.is_none() && right.is_none() {
                return true;
            }
            return Self::has_path_sum(left, target) || Self::has_path_sum(right, target);
        }
        false
    }

    // 114. Flatten Binary Tree to Linked List
    pub fn flatten(root: &mut TNode) {
        fn flat(right: TNode, after: TNode) -> TNode {
            match right {
                Some(n) => {
                    let mut b = n.borrow_mut();
                    let right = flat(b.right.take(), after);
                    b.right = flat(b.left.take(), right);
                    drop(b);
                    Some(n)
                }
                None => after,
            }
        }
        *root = flat(root.take(), None);
    }

    // 106. Construct Binary Tree from Inorder and Postorder Traversal
    pub fn build_tree2(inorder: Vec<i32>, postorder: Vec<i32>) -> TNode {
        fn build(inorder: &[i32], postorder: &[i32]) -> TNode {
            if inorder.is_empty() || postorder.is_empty() {
                return None;
            }
            let i = inorder.len() - 1;
            let val = postorder[i];
            let m = inorder.iter().position(|&v| v == val).unwrap();
            Some(Rc::new(RefCell::new(TreeNode {
                val,
                left: build(&inorder[0..m], &postorder[0..m]),
                right: build(&inorder[m + 1..i + 1], &postorder[m..i]),
            })))
        }
        build(&inorder[..], &postorder[..])
    }

    // 105. Construct Binary Tree from Preorder and Inorder Traversal
    pub fn build_tree(preorder: Vec<i32>, inorder: Vec<i32>) -> TNode {
        fn dfs(
            pre: &mut std::slice::Iter<i32>,
            map: &mut HashMap<i32, usize>,
            left: usize,
            right: usize,
        ) -> TNode {
            if left > right || right > map.len() {
                return None;
            }
            match pre.next() {
                Some(val) => {
                    let i = *map.get(val).unwrap();
                    Some(Rc::new(RefCell::new(TreeNode {
                        val: *val,
                        left: dfs(pre, map, left, i - 1),
                        right: dfs(pre, map, i + 1, right),
                    })))
                }
                _ => None,
            }
        }
        let mut map: HashMap<i32, usize> = inorder
            .into_iter()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();
        dfs(&mut preorder.iter(), &mut map, 0, preorder.len() - 1)
    }

    // 101. Symmetric Tree
    pub fn is_symmetric(root: TNode) -> bool {
        fn dfs(left: TNode, right: TNode) -> bool {
            match (left, right) {
                (None, None) => true,
                (Some(left), Some(right)) => {
                    left.borrow().val == right.borrow().val
                        && dfs(left.borrow().left.clone(), right.borrow().right.clone())
                        && dfs(left.borrow().right.clone(), right.borrow().left.clone())
                }
                _ => false,
            }
        }
        match root {
            Some(root) => dfs(root.borrow().left.clone(), root.borrow().right.clone()),
            None => true,
        }
    }

    // 226. Invert Binary Tree
    pub fn invert_tree(root: TNode) -> TNode {
        match root {
            Some(root) => {
                let mut node = root.borrow_mut();
                let tmp = node.left.clone();
                node.left = node.right.clone();
                node.right = tmp;
                Self::invert_tree(node.right.clone());
                Self::invert_tree(node.left.clone());
                drop(node);
                Some(root)
            }
            None => None,
        }
    }

    // 100. Same Tree
    pub fn is_same_tree(p: TNode, q: TNode) -> bool {
        match (p, q) {
            (None, None) => true,
            (_, None) | (None, _) => false,
            (Some(p), Some(q)) => {
                p.borrow().val == q.borrow().val
                    && Solution::is_same_tree(p.borrow().left.clone(), q.borrow().left.clone())
                    && Solution::is_same_tree(p.borrow().right.clone(), q.borrow().right.clone())
            }
        }
    }

    // 104. Maximum Depth of Binary TreeNode
    pub fn max_depth(root: TNode) -> i32 {
        fn dp(root: TNode, mut depth: i32) -> i32 {
            match root {
                Some(root) => max(
                    dp(root.borrow().left.clone(), depth + 1),
                    dp(root.borrow().right.clone(), depth + 1),
                ),
                None => depth,
            }
        }
        dp(root, 0)
    }

    // 21. Merge Two Sorted Lists
    // once
    pub fn merge_two_lists(
        list1: Option<Box<ListNode>>,
        list2: Option<Box<ListNode>>,
    ) -> Option<Box<ListNode>> {
        match (list1, list2) {
            (None, None) => None,
            (Some(n), None) | (None, Some(n)) => Some(n),
            (Some(n1), Some(n2)) => {
                if n1.val >= n2.val {
                    Some(Box::new(ListNode {
                        val: n2.val,
                        next: Solution::merge_two_lists(Some(n1), n2.next),
                    }))
                } else {
                    Some(Box::new(ListNode {
                        val: n1.val,
                        next: Solution::merge_two_lists(n1.next, Some(n2)),
                    }))
                }
            }
        }
    }

    // 2. Add Two Numbers
    pub fn add_two_numbers(
        mut l1: Option<Box<ListNode>>,
        mut l2: Option<Box<ListNode>>,
    ) -> Option<Box<ListNode>> {
        let mut dummy = ListNode::new(0);
        let mut head = &mut dummy;
        let mut carry = 0;
        while l1.is_some() || l2.is_some() || carry == 1 {
            let mut sum = l1.as_ref().map_or(0, |node| node.val)
                + l2.as_ref().map_or(0, |node| node.val)
                + carry;
            if sum >= 10 {
                carry = 1;
                sum -= 10;
            } else {
                carry = 0;
            }
            let node = ListNode::new(sum);
            head.next = Some(Box::new(node));
            head = head.next.as_mut().unwrap();
            l1 = l1.and_then(|node| node.next);
            l2 = l2.and_then(|node| node.next);
        }
        dummy.next
    }

    // 224. Basic Calculator
    pub fn calculate(s: String) -> i32 {
        let mut stack: Vec<i32> = vec![1];
        let mut ans = 0;
        let mut num = 0;
        let mut sign = 1;

        for ch in s.chars() {
            match ch {
                ' ' => continue,
                '0'..='9' => {
                    num = num * 10 + (ch as i32 - '0' as i32);
                }
                '+' | '-' => {
                    ans += sign * num;
                    num = 0;
                    sign = *stack.last().unwrap() * if ch == '+' { 1 } else { -1 };
                }
                '(' => {
                    stack.push(sign);
                }
                ')' => {
                    stack.pop();
                }
                _ => {}
            }
        }
        ans + sign * num
    }

    // 150. Evaluate Reverse Polish Notation
    // once
    pub fn eval_rpn(tokens: Vec<String>) -> i32 {
        let mut stack = Vec::with_capacity(tokens.len());
        for str in tokens {
            match str {
                s if s == "+" => {
                    let val = stack.pop().unwrap() + stack.pop().unwrap();
                    stack.push(val);
                }
                s if s == "-" => {
                    let val = 0 - stack.pop().unwrap() + stack.pop().unwrap();
                    stack.push(val);
                }
                s if s == "*" => {
                    let val = stack.pop().unwrap() * stack.pop().unwrap();
                    stack.push(val);
                }
                s if s == "/" => {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(a / b);
                }
                _ => stack.push(str.parse().unwrap()),
            }
        }
        stack.pop().unwrap()
    }

    // 71. Simplify Path
    // once
    pub fn simplify_path(path: String) -> String {
        let root = "/".to_owned();
        let mut stack = vec![];
        for directory in path.split("/") {
            match directory {
                "" | "." => continue,
                ".." => {
                    stack.pop();
                }
                directory => stack.push(directory),
            }
        }

        root + stack.join("/").as_str()
    }

    // 20. Valid Parentheses
    pub fn is_valid(s: String) -> bool {
        let mut stack = Vec::new();
        fn pair(b: &u8) -> u8 {
            match b {
                b')' => b'(',
                b'}' => b'{',
                b']' => b'[',
                _ => b' ',
            }
        }

        for c in s.as_bytes().iter() {
            match c {
                b'(' | b'{' | b'[' => stack.push(c),
                b')' | b'}' | b']' => match stack.pop() {
                    Some(pop) if *pop == pair(c) => (),
                    _ => return false,
                },
                _ => (),
            }
            println!("{:?}", stack);
        }
        stack.is_empty()
    }

    // 452. Minimum Number of Arrows to Burst Balloons
    pub fn find_min_arrow_shots(mut points: Vec<Vec<i32>>) -> i32 {
        let mut count = 1;
        points.sort_unstable_by_key(|v| v[1]);
        let mut shoot = points[0][1];
        for ele in points.into_iter() {
            if ele[0] > shoot {
                count += 1;
                shoot = ele[1];
            }
        }
        count
    }

    // 57. Insert Interval
    pub fn insert(intervals: Vec<Vec<i32>>, mut new_interval: Vec<i32>) -> Vec<Vec<i32>> {
        let mut res = Vec::with_capacity(intervals.len());
        let mut pushed = false;
        for curr in intervals.into_iter() {
            if pushed {
                res.push(curr);
                continue;
            }
            if new_interval[1] < curr[0] {
                res.push(new_interval.clone());
                res.push(curr);
                pushed = true;
                continue;
            }
            if new_interval[0] <= curr[1] && new_interval[1] >= curr[0] {
                new_interval = vec![min(new_interval[0], curr[0]), max(new_interval[1], curr[1])];
                if new_interval[1] <= curr[1] {
                    res.push(new_interval.clone());
                } else {
                    continue;
                }
                pushed = true;
            } else {
                res.push(curr);
            }
        }
        if !pushed {
            res.push(new_interval);
        }
        res
    }

    // 228. Summary Ranges
    pub fn summary_ranges(nums: Vec<i32>) -> Vec<String> {
        if nums.is_empty() {
            return vec![];
        }
        let mut res = vec![];
        let mut left = nums[0];
        let mut right = None;
        for ele in nums.into_iter().skip(1) {
            if ele == right.unwrap_or(left) {
                continue;
            }
            if ele == right.unwrap_or(left) + 1 {
                right = Some(ele);
                continue;
            }
            res.push(if right.is_some() {
                format!("{}->{}", left, right.unwrap())
            } else {
                left.to_string()
            });
            left = ele;
            right = None;
        }
        res.push(if right.is_some() {
            format!("{}->{}", left, right.unwrap())
        } else {
            left.to_string()
        });
        res
    }

    // 128. Longest Consecutive Sequence
    pub fn longest_consecutive(mut nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        nums.sort_unstable();
        // nums.dedup();
        let mut longest = 1;
        let mut count = 1;
        let mut prev = nums[0];
        for ele in nums.into_iter().skip(1) {
            if ele == prev {
                continue;
            }
            if ele == prev + 1 {
                count += 1;
            } else {
                longest = max(longest, count);
                count = 1;
            }
            prev = ele;
        }
        max(longest, count)
    }

    // 56. Merge Intervals
    pub fn merge(mut intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        intervals.sort();
        let (mut result, mut cur) = (vec![], vec![]);
        for interval in intervals {
            if cur.is_empty() {
                cur = interval;
            } else if cur[1] >= interval[0] {
                cur[1] = cur[1].max(interval[1])
            } else {
                result.push(cur);
                cur = interval;
            }
        }
        result.push(cur);
        result
    }

    // 219. Contains Duplicate II
    pub fn contains_nearby_duplicate(nums: Vec<i32>, k: i32) -> bool {
        let mut map: HashMap<i32, usize> = HashMap::with_capacity(nums.len());
        nums.into_iter().enumerate().any(|(i, v)| {
            let j = map.insert(v, i);
            match j {
                Some(j) => (i - j) as i32 <= k,
                None => false,
            }
        })
    }

    // 202. Happy Number
    pub fn is_happy(mut n: i32) -> bool {
        let map: HashMap<i32, i32> = (0..=9).map(|i| (i, i * i)).collect();
        loop {
            let mut tmp = n;
            n = 0;
            let mut digit = tmp % 10;
            while digit != 0 || tmp >= 10 {
                n += *map.get(&digit).unwrap();
                tmp /= 10;
                digit = tmp % 10;
            }
            if n == 1 {
                return true;
            }
            if n < 10 {
                return false;
            }
        }
    }

    // 205. Isomorphic Strings
    pub fn is_isomorphic(s: String, t: String) -> bool {
        let mut s2t: HashMap<char, char> = HashMap::with_capacity(26);
        let mut t2s: HashMap<char, char> = HashMap::with_capacity(26);
        s.chars()
            .into_iter()
            .zip(t.chars().into_iter())
            .all(|(c1, c2)| {
                let v1 = s2t.entry(c1).or_insert(c2);
                let v2 = t2s.entry(c2).or_insert(c1);
                *v1 == c2 && *v2 == c1
            })
    }

    // 289. Game of Life
    pub fn game_of_life(board: &mut Vec<Vec<i32>>) {}
    // 51. N-Queens
    pub fn solve_n_queens(n: i32) -> Vec<Vec<String>> {
        let n: usize = n as usize;
        let mut cols: Vec<bool> = vec![false; n];
        let mut downward_diagonal: Vec<bool> = vec![false; n + n];
        let mut upward_diagonal: Vec<bool> = vec![false; n + n];

        let mut dp: Vec<String> = vec![];
        let mut res: Vec<Vec<String>> = vec![];
        fn backtrack(
            x: usize,
            y: usize,
            n: &usize,
            cols: &mut Vec<bool>,
            downward_diagonal: &mut Vec<bool>,
            upward_diagonal: &mut Vec<bool>,
            dp: &mut Vec<String>,
            res: &mut Vec<Vec<String>>,
        ) {
            if cols[y] || downward_diagonal[x + n - y] || upward_diagonal[x + y] {
                return;
            }
            println!(
                "{} {}, {:?}, {:?}, {:?}",
                x, y, cols, downward_diagonal, upward_diagonal
            );
            let str: String = (0..*n).map(|i| if i == y { 'Q' } else { '.' }).collect();
            dp.push(str);
            cols[y] = true;
            downward_diagonal[x + n - y] = true;
            upward_diagonal[x + y] = true;
            if x == n - 1 {
                res.push(dp.clone());
                dp.pop();
                cols[y] = false;
                downward_diagonal[x + n - y] = false;
                upward_diagonal[x + y] = false;
                return;
            }
            for yy in 0..*n {
                backtrack(
                    x + 1,
                    yy,
                    n,
                    cols,
                    downward_diagonal,
                    upward_diagonal,
                    dp,
                    res,
                );
            }
            dp.pop();
            cols[y] = false;
            downward_diagonal[x + n - y] = false;
            upward_diagonal[x + y] = false;
        }
        for y in 0..n {
            backtrack(
                0,
                y,
                &n,
                &mut cols,
                &mut downward_diagonal,
                &mut upward_diagonal,
                &mut dp,
                &mut res,
            );
        }
        res
    }

    // 48. Rotate Image
    pub fn rotate_matrix(matrix: &mut [Vec<i32>]) {
        if matrix.is_empty() {
            return;
        }
        let mut n = matrix.len() - 1;
        let mut i = 0;
        while i < n {
            for j in i..n {
                let left_top = matrix[i][j];
                matrix[i][j] = matrix[n - j + i][i];
                matrix[n - j + i][i] = matrix[n][n - j + i];
                matrix[n][n - j + i] = matrix[j][n];
                matrix[j][n] = left_top;
            }
            i += 1;
            n -= 1;
        }
    }

    // 54. Spiral Matrix
    pub fn spiral_order(matrix: Vec<Vec<i32>>) -> Vec<i32> {
        let mut res = vec![];
        if matrix.is_empty() {
            return res;
        }
        let (mut top, mut bottom, mut left, mut right) = (
            0i32,
            matrix.len() as i32 - 1,
            0i32,
            matrix[0].len() as i32 - 1,
        );
        while top <= bottom && left <= right {
            for i in left..=right {
                res.push(matrix[top as usize][i as usize]);
            }
            top += 1;
            for i in top..=bottom {
                res.push(matrix[i as usize][right as usize]);
            }
            right -= 1;
            if top <= bottom {
                for i in (left..=right).rev() {
                    res.push(matrix[bottom as usize][i as usize]);
                }
                bottom -= 1;
            }
            if left <= right {
                for i in (top..=bottom).rev() {
                    res.push(matrix[i as usize][left as usize]);
                }
                left += 1;
            }
        }
        res
    }

    // 37. Sudoku Solver
    pub fn solve_sudoku(board: &mut Vec<Vec<char>>) {
        const BITS: u32 = 0x1FF;
        let mut row = std::collections::HashSet::new();
        let mut col = std::collections::HashSet::new();
        let mut boxs = std::collections::HashSet::new();
        for i in 0..9 {
            for j in 0..9 {
                if board[i][j] == '.' {
                    continue;
                }
                let num = board[i][j] as usize - '1' as usize;
                row.insert((i, num));
                col.insert((j, num));
                boxs.insert((i / 3 * 3 + j / 3, num));
            }
        }

        #[allow(clippy::needless_range_loop)]
        fn is_invalid(
            i: usize,
            j: usize,
            c: char,
            row: &std::collections::HashSet<(usize, usize)>,
            col: &std::collections::HashSet<(usize, usize)>,
            boxs: &std::collections::HashSet<(usize, usize)>,
        ) -> bool {
            let num = c as usize - '1' as usize;
            row.contains(&(i, num))
                || col.contains(&(j, num))
                || boxs.contains(&(i / 3 * 3 + j / 3, num))
        }

        fn backtrack(
            board: &mut Vec<Vec<char>>,
            row: &mut std::collections::HashSet<(usize, usize)>,
            col: &mut std::collections::HashSet<(usize, usize)>,
            boxs: &mut std::collections::HashSet<(usize, usize)>,
        ) -> bool {
            for i in 0..9 {
                for j in 0..9 {
                    if board[i][j] != '.' {
                        continue;
                    }
                    for c in '1'..='9' {
                        if is_invalid(i, j, c, row, col, boxs) {
                            continue;
                        }
                        board[i][j] = c;
                        let num = c as usize - '1' as usize;
                        row.insert((i, num));
                        col.insert((j, num));
                        boxs.insert((i / 3 * 3 + j / 3, num));

                        if backtrack(board, row, col, boxs) {
                            return true;
                        }
                        row.remove(&(i, num));
                        col.remove(&(j, num));
                        boxs.remove(&(i / 3 * 3 + j / 3, num));
                        board[i][j] = '.';
                    }
                    return false;
                }
            }
            true
        }
        backtrack(board, &mut row, &mut col, &mut boxs);
    }

    // 36. Valid Sudoku
    pub fn is_valid_sudoku(board: Vec<Vec<char>>) -> bool {
        let mut rows = vec![vec![false; 9]; 9];
        let mut cols = vec![vec![false; 9]; 9];
        let mut boxes = vec![vec![false; 9]; 9];
        for i in 0..9 {
            for j in 0..9 {
                if board[i][j] == '.' {
                    continue;
                }
                let num = board[i][j] as usize - '1' as usize;
                let box_index = i / 3 * 3 + j / 3;
                if rows[i][num] || cols[j][num] || boxes[box_index][num] {
                    return false;
                }
                rows[i][num] = true;
                cols[j][num] = true;
                boxes[box_index][num] = true;
            }
        }
        true
    }

    // 76. Minimum Window Substring
    pub fn min_window(s: String, t: String) -> String {
        let s_bytes = s.as_bytes();
        let t_bytes = t.as_bytes();
        let mut map = std::collections::HashMap::new();
        for c in t_bytes.iter() {
            *map.entry(c).or_insert(0) += 1;
        }
        let mut left = 0;
        let mut count = 0;
        let mut min_len = std::usize::MAX;
        let mut min_left = 0;
        for (right, c) in s_bytes.iter().enumerate() {
            if let Some(&v) = map.get(&c) {
                if v > 0 {
                    count += 1;
                }
                map.insert(c, v - 1);
            }
            while count == t.len() {
                if right - left + 1 < min_len {
                    min_len = right - left + 1;
                    min_left = left;
                }
                if let Some(&v) = map.get(&s_bytes[left]) {
                    if v == 0 {
                        count -= 1;
                    }
                    map.insert(&s_bytes[left], v + 1);
                }
                left += 1;
            }
        }
        if min_len == std::usize::MAX {
            "".to_string()
        } else {
            s[min_left..min_left + min_len].to_string()
        }
    }

    // 30. Substring with Concatenation of All Words
    pub fn find_substring(s: String, words: Vec<String>) -> Vec<i32> {
        if s.is_empty() || words.is_empty() {
            return vec![];
        }
        let mut res = vec![];
        let words_len = words.len();
        let mut map = std::collections::HashMap::with_capacity(words_len);
        let word_len = words[0].len();
        let s_len = s.len();
        for word in words.iter() {
            *map.entry(word.as_str()).or_insert(0) += 1;
        }
        for i in 0..word_len {
            let mut left = i;
            let mut right = i;
            let mut count = 0;
            let mut window = std::collections::HashMap::with_capacity(words_len);
            while right + word_len <= s_len {
                let word = &s[right..right + word_len];
                right += word_len;
                if !map.contains_key(word) {
                    count = 0;
                    left = right;
                    window.clear();
                } else {
                    *window.entry(word).or_insert(0) += 1;
                    count += 1;
                    while window.get(word).unwrap() > map.get(word).unwrap() {
                        let left_word = &s[left..left + word_len];
                        left += word_len;
                        *window.get_mut(left_word).unwrap() -= 1;
                        count -= 1;
                    }
                    if count == words_len {
                        res.push(left as i32);
                    }
                }
            }
        }
        res
    }

    // 3. Longest Substring Without Repeating Characters
    pub fn length_of_longest_substring(s: String) -> i32 {
        let mut max = 0;
        let mut left = 0;
        let mut map = std::collections::HashMap::new();
        for (right, c) in s.chars().enumerate() {
            if let Some(&index) = map.get(&c) {
                left = left.max(index + 1);
            }
            map.insert(c, right);
            max = max.max(right - left + 1);
        }
        max as i32
    }

    // 209. Minimum Size Subarray Sum
    pub fn min_sub_array_len(target: i32, nums: Vec<i32>) -> i32 {
        let mut min_len = std::i32::MAX;
        let mut sum = 0;
        let mut left = 0;
        for (right, &num) in nums.iter().enumerate() {
            sum += num;
            while sum >= target {
                min_len = min_len.min((right - left + 1) as i32);
                sum -= nums[left];
                left += 1;
            }
        }
        if min_len == std::i32::MAX {
            0
        } else {
            min_len
        }
    }

    // 131. Palindrome Partitioning
    pub fn partition_palindrome(s: String) -> Vec<Vec<String>> {
        let mut res: Vec<Vec<String>> = vec![];
        let s: Vec<char> = s.chars().collect();

        #[inline]
        fn is_palindrome((mut l, mut r): (usize, usize), cs: &Vec<char>) -> bool {
            while l < r {
                if cs[l] != cs[r] {
                    return false;
                }
                l += 1;
                r -= 1;
            }
            true
        }

        fn palindrome(
            mut dp: Vec<String>,
            start: usize,
            cs: &Vec<char>,
            res: &mut Vec<Vec<String>>,
        ) {
            if start == cs.len() {
                res.push(dp.clone());
                return;
            }
            for end in start..cs.len() {
                if is_palindrome((start, end), cs) {
                    dp.push(cs[start..=end].iter().collect());
                    palindrome(dp.clone(), end + 1, cs, res);
                    dp.pop();
                }
            }
        }

        palindrome(vec![], 0, &s, &mut res);
        res
    }

    // 15. 3Sum
    pub fn three_sum(mut nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut nums = nums.clone();
        nums.sort();
        let mut res = Vec::new();
        let length = nums.len();

        for i in 0..length - 2 {
            if i > 0 && nums[i] == nums[i - 1] {
                continue;
            }

            if nums[i] > 0 {
                break;
            }

            let (mut l, mut r) = (i + 1, length - 1);

            while l < r {
                let total = nums[i] + nums[l] + nums[r];

                if total < 0 {
                    l += 1;
                } else if total > 0 {
                    r -= 1;
                } else {
                    res.push(vec![nums[i], nums[l], nums[r]]);

                    while l < r && nums[l] == nums[l + 1] {
                        l += 1;
                    }

                    while l < r && nums[r] == nums[r - 1] {
                        r -= 1;
                    }

                    l += 1;
                    r -= 1;
                }
            }
        }
        res
    }

    // 11. Container With Most Water
    // once
    pub fn max_area(height: Vec<i32>) -> i32 {
        let mut i = 0;
        let mut j = height.len() - 1;
        let mut max_val = 0;
        while i < j {
            let w = (j - i) as i32;
            max_val = max(min(height[j], height[i]) * w, max_val);
            if height[j] < height[i] {
                j -= 1;
            } else {
                i += 1;
            }
        }
        max_val
    }

    // 167. Two Sum II - Input Array Is Sorted
    pub fn two_sum(numbers: Vec<i32>, target: i32) -> Vec<i32> {
        let mut i = 0;
        let mut j = numbers.len() - 1;
        loop {
            match (numbers[i] + numbers[j]).cmp(&target) {
                Ordering::Less => i += 1,
                Ordering::Greater => j -= 1,
                Ordering::Equal => return vec![(i + 1) as i32, (j + 1) as i32],
            }
        }
    }

    // 392. Is Subsequence
    pub fn is_subsequence(s: String, t: String) -> bool {
        let mut t_iter = t.chars();
        s.chars()
            .all(|char_s| t_iter.find(|&char_t| char_t == char_s).is_some())
    }

    // 125. Valid Palindrome
    pub fn is_palindrome(s: String) -> bool {
        let s: Vec<_> = s
            .chars()
            .filter_map(|c| c.is_ascii_alphanumeric().then(|| c.to_ascii_lowercase()))
            .collect();

        s.iter().eq(s.iter().rev())
    }

    // 68. Text Justification
    pub fn full_justify(words: Vec<String>, max_width: i32) -> Vec<String> {
        let max_width = max_width as usize;
        let mut res = Vec::new();
        let mut tmp_vec: Vec<String> = Vec::new();
        let mut total_len = 0;
        for word in words.into_iter() {
            if total_len + word.len() + tmp_vec.len() > max_width {
                for i in 0..(max_width - total_len) {
                    let idx = i as usize
                        % (if tmp_vec.len() > 1 {
                            tmp_vec.len() - 1
                        } else {
                            tmp_vec.len()
                        });
                    tmp_vec[idx] = format!("{} ", tmp_vec[idx]);
                }
                let new_word: String = tmp_vec.join("");
                res.push(new_word);
                total_len = word.len();
                tmp_vec.clear();
                tmp_vec.push(word);
            } else {
                total_len += word.len();
                tmp_vec.push(word);
            }
        }
        let new_word: String = tmp_vec.join(" ");
        res.push(format!("{:<max_width$}", new_word));
        res
    }

    // 28. Find the Index of the First Occurrence in a String
    // once
    pub fn str_str(haystack: String, needle: String) -> i32 {
        haystack.find(needle.as_str()).map_or(-1, |i| i as i32)
    }

    // 6. Zigzag Conversion
    /*
    Input: s = "PAYPALISHIRING", numRows = 3
    Output: "PAHNAPLSIIGYIR"
    P0    A0    H0    N0
    A1 P1 L1 S1 I1 I1 G1
    Y2    I2    R2
    */
    pub fn convert(s: String, num_rows: i32) -> String {
        let mut zigzags: Vec<_> = (0..num_rows)
            .chain((1..num_rows - 1).rev())
            .cycle()
            .zip(s.chars())
            .collect();
        zigzags.sort_by_key(|&(row, _)| row);
        zigzags.into_iter().map(|(_, c)| c).collect()
    }

    // 151. Reverse Words in a String
    pub fn reverse_words(s: String) -> String {
        s.split_whitespace().rev().collect::<Vec<&str>>().join(" ")
    }

    // 14. Longest Common Prefix
    pub fn longest_common_prefix(strs: Vec<String>) -> String {
        strs.into_iter()
            .reduce(|acc, cur| {
                acc.chars()
                    .zip(cur.chars())
                    .take_while(|(a, c)| a == c)
                    .map(|(c, _)| c)
                    .collect()
            })
            .unwrap()
    }

    // 13. Roman to Integer
    // once
    pub fn roman_to_int(s: String) -> i32 {
        let roman_map = HashMap::from([
            ('I', 1),
            ('V', 5),
            ('X', 10),
            ('L', 50),
            ('C', 100),
            ('D', 500),
            ('M', 1000),
        ]);
        let mut curr = roman_map.get(&s.chars().next().unwrap()).unwrap();
        let mut count = 0;
        for c in s.chars().skip(1) {
            let next = roman_map.get(&c).unwrap();
            if curr < next {
                count += curr * -1;
            } else {
                count += curr;
            }
            curr = next;
        }
        count + curr
    }

    // 135. Candy
    pub fn candy(ratings: Vec<i32>) -> i32 {
        let mut candies = vec![1; ratings.len()];
        let mut curr = 1;
        for i in 1..ratings.len() {
            if ratings[i] > ratings[i - 1] {
                curr += 1;
            } else {
                curr = 1;
            }
            candies[i] = curr;
        }
        curr = candies[ratings.len() - 1];
        for i in (0..ratings.len() - 1).rev() {
            if ratings[i] > ratings[i + 1] {
                curr = max(curr + 1, candies[i]);
            } else {
                curr = max(1, candies[i]);
            }
            candies[i] = curr;
        }
        candies.iter().sum()
    }

    // 134. Gas Station
    pub fn can_complete_circuit(gas: Vec<i32>, cost: Vec<i32>) -> i32 {
        let delta: Vec<i32> = gas
            .into_iter()
            .zip(cost.into_iter())
            .map(|(g, c)| g - c)
            .collect();
        let mut index = 0;
        let mut count = 0;
        let mut total = 0;
        for i in 0..delta.len() {
            total += delta[i];
            count += delta[i];
            if count < 0 {
                index = i + 1;
                count = 0;
            }
        }
        if count >= 0 {
            index as i32
        } else {
            -1
        }
    }

    // 42. Trapping Rain Water
    // once
    pub fn trap(height: Vec<i32>) -> i32 {
        let mut trap = vec![0; height.len()];
        let mut h = 0;
        for i in 0..height.len() {
            trap[i] = max(h - height[i], 0);
            h = max(height[i], h);
        }
        h = 0;
        for i in (0..height.len()).rev() {
            trap[i] = min(max(h - height[i], 0), trap[i]);
            h = max(height[i], h);
        }
        trap.iter().sum()
    }

    // 238. Product of Array Except Self
    pub fn product_except_self(nums: Vec<i32>) -> Vec<i32> {
        let len = nums.len();
        let mut ans = vec![1; len];
        let mut curr = 1;
        for i in 0..len {
            ans[i] *= curr;
            curr *= nums[i];
        }
        curr = 1;
        for i in (0..len).rev() {
            ans[i] *= curr;
            curr *= nums[i];
        }
        ans
    }

    // 274. H-Index
    // once
    pub fn h_index(mut citations: Vec<i32>) -> i32 {
        citations.sort();
        let mut count = citations.len() as i32;
        for citation in citations.iter() {
            if *citation >= count {
                break;
            }
            count -= 1;
        }
        count
    }

    // 45. Jump Game II
    pub fn jump(nums: Vec<i32>) -> i32 {
        let mut count = 0;
        let mut max_reach = 0;
        let mut current_end = 0;
        for i in 0..nums.len() - 1 {
            let j = nums[i];
            max_reach = max_reach.max(i + j as usize);
            if i == current_end {
                count += 1;
                current_end = max_reach;
            }
        }
        count
    }

    // 55. Jump Game
    /* There is another way to check the maxium reachable from 0 -> last one,
    if someone is out of the maxium reachable range, it means it is impossible to reach here.
    */
    pub fn can_jump(nums: Vec<i32>) -> bool {
        if nums.len() == 1 {
            return true;
        }
        let mut left = nums.len() - 2;
        let mut right = nums.len() - 2;
        for i in (0..nums.len() - 1).rev() {
            let jump = nums[i] as usize;
            let reach = i + jump;
            if reach > right {
                right = reach;
            }
            if reach >= left {
                left = i;
            }
        }
        left == 0 && right >= nums.len() - 1
    }

    // 122. Best Time to Buy and Sell Stock II
    // once
    pub fn max_profit2(prices: Vec<i32>) -> i32 {
        let mut count = 0;
        let mut buy = prices[0];
        for price in prices.iter().skip(1) {
            if *price > buy {
                count += price - buy;
            }
            buy = *price;
        }
        count
    }

    // 121. Best Time to Buy and Sell Stock
    pub fn max_profit(prices: Vec<i32>) -> i32 {
        let mut buy = prices[0];
        let mut profit = 0;
        for i in 1..prices.len() {
            let prz = prices[i];
            let tmp = prz - buy;
            if tmp > profit {
                profit = tmp;
            }
            if prz < buy {
                buy = prz;
            }
        }
        profit
    }

    // 189. Rotate Arrayshift
    // once
    /*
       let n = nums.len();
       let k = (k as usize) % n; // In case k is greater than n
       nums.reverse();
       nums[..k].reverse();
       nums[k..].reverse();
    */
    pub fn rotate(nums: &mut Vec<i32>, k: i32) {
        if k < 1 {
            return;
        }
        let mut uk = k as usize;
        if uk > nums.len() {
            uk = uk - nums.len();
        }
        nums.reverse();
        let mut i = 0;
        let mut j = uk - 1;
        while i < j {
            let tmp = nums[j];
            nums[j] = nums[i];
            nums[i] = tmp;
            i += 1;
            j -= 1;
        }
        i = uk;
        j = nums.len() - 1;
        while i < j {
            let tmp = nums[j];
            nums[j] = nums[i];
            nums[i] = tmp;
            i += 1;
            j -= 1;
        }
    }

    // 169. Majority Element
    pub fn majority_element(mut nums: Vec<i32>) -> i32 {
        nums.sort();
        let mut index = 0;
        let mut count = 0;
        let mut majority = 0;
        while index < nums.len() {
            if nums[index] == majority {
                count += 1;
            } else {
                if index + count > nums.len() {
                    break;
                }
                if nums[index + count] == nums[index] {
                    majority = nums[index];
                    index += count;
                    continue;
                }
            }
            index += 1;
        }
        majority
    }

    // 86. Partition List
    pub fn partition(mut head: Option<Box<ListNode>>, x: i32) -> Option<Box<ListNode>> {
        let mut l_dummy = ListNode::new(-1);
        let mut r_dummy = ListNode::new(-1);
        let mut l_tail = &mut l_dummy;
        let mut r_tail = &mut r_dummy;

        while let Some(mut curr) = head.take() {
            head = curr.next.take();
            if curr.val < x {
                l_tail.next = Some(curr);
                l_tail = l_tail.next.as_mut().unwrap();
            } else {
                r_tail.next = Some(curr);
                r_tail = r_tail.next.as_mut().unwrap();
            }
        }
        l_tail.next = r_dummy.next;
        l_dummy.next
    }

    // 328. Odd Even Linked List
    pub fn odd_even_list(mut head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        if head.as_ref()?.next.is_none() {
            return head;
        }

        let mut even_head = head.as_mut()?.next.take();
        let mut even_tail = &mut even_head;
        let mut odd_tail = &mut head;

        while even_tail.is_some() && even_tail.as_ref()?.next.is_some() {
            let mut next_odd = even_tail.as_mut()?.next.take();
            let next_even = next_odd.as_mut()?.next.take();
            odd_tail.as_mut()?.next = next_odd;
            odd_tail = &mut odd_tail.as_mut()?.next;
            even_tail.as_mut()?.next = next_even;
            even_tail = &mut even_tail.as_mut()?.next;
        }
        odd_tail.as_mut()?.next = even_head;
        head
    }

    // 80. Remove Duplicates from Sorted Array II
    pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
        nums.dedup();
        if nums.len() <= 2 {
            return nums.len() as i32;
        }
        let mut j = 2;
        for i in 2..nums.len() {
            if nums[j] != nums[i] {
                nums[j] = nums[i];
            }
            if nums[j] != nums[j - 2] {
                j += 1;
            }
        }
        j as i32
    }
}

// 173. Binary Search Tree Iterator
struct BSTIterator {
    stack: Vec<i32>,
}
impl BSTIterator {
    fn new(root: Option<Rc<RefCell<TreeNode>>>) -> Self {
        fn push(root: &TNode, stack: &mut Vec<i32>) {
            match root {
                Some(root) => {
                    let node = root.as_ref().borrow();
                    push(&node.right, stack);
                    stack.push(node.val);
                    push(&node.left, stack);
                }
                None => {}
            }
        }
        let mut stack = Vec::new();
        push(&root, &mut stack);
        Self { stack }
    }

    fn next(&mut self) -> i32 {
        self.stack.pop().unwrap()
    }

    fn has_next(&self) -> bool {
        !self.stack.is_empty()
    }
}

// 155. Min Stack
struct MinStack {
    val_to_min: Vec<(i32, i32)>,
}
impl MinStack {
    fn new() -> Self {
        MinStack {
            val_to_min: Vec::new(),
        }
    }

    fn push(&mut self, val: i32) {
        if let Some((_, pop_min)) = self.val_to_min.last() {
            self.val_to_min.push((val, min(*pop_min, val)));
        } else {
            self.val_to_min.push((val, val));
        }
    }

    fn pop(&mut self) {
        self.val_to_min.pop();
    }

    fn top(&self) -> i32 {
        self.val_to_min.last().unwrap().0
    }

    fn get_min(&self) -> i32 {
        self.val_to_min.last().unwrap().1
    }
}

// 380. Insert Delete GetRandom O(1)
use rand::Rng;

struct RandomizedSet {
    vec: Vec<i32>,
    map: HashMap<i32, usize>,
}
impl RandomizedSet {
    fn new() -> Self {
        RandomizedSet {
            vec: Vec::new(),
            map: HashMap::new(),
        }
    }
    fn insert(&mut self, val: i32) -> bool {
        if self.map.contains_key(&val) {
            return false;
        }
        self.vec.push(val);
        self.map.insert(val, self.vec.len() - 1);
        true
    }
    fn remove(&mut self, val: i32) -> bool {
        if let Some(index) = self.map.remove(&val) {
            let last_element = *self.vec.last().unwrap();
            self.vec.swap_remove(index);
            if index < self.vec.len() {
                self.map.insert(last_element, index);
            }
            true
        } else {
            false
        }
    }
    fn get_random(&self) -> i32 {
        let mut rng = rand::thread_rng();
        let random_index = rng.gen_range(0..self.vec.len());
        self.vec[random_index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn mini_window() {
        match fs::read_to_string("mini_window.txt") {
            Ok(content) => {
                let mut lines = content.lines();
                if let (Some(s), Some(t)) = (lines.next(), lines.next()) {
                    let res = Solution::min_window(s.to_string(), t.to_string());
                    assert_eq!(res, "");
                }
            }
            Err(_) => panic!("File not found"),
        }
    }
}

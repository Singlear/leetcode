use std::{
    cmp::{max, min, Ordering},
    collections::HashMap,
    i32,
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

struct Solution;

#[allow(dead_code)]
impl Solution {
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

    // 148. Sort List
    pub fn sort_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        todo!()
        // let mut dummy = ListNode::new(i32::MIN);
        // let mut head = head.as_mut();
        // while head.is_some() {
        //     let curr = head.as_mut()?;

        //     let mut new_head = &mut dummy;
        //     while new_head.next.is_some() && new_head.next.as_mut()?.val < curr.as_mut().val {
        //         new_head = new_head.next.as_mut()?;
        //     }
        //     head = &mut curr.next.as_mut();
        //     curr.next = new_head.next;
        // }
        // dummy.next
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
        left <= 0 && right >= nums.len() - 1
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
